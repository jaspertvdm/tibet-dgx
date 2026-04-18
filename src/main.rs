// ═══════════════════════════════════════════════════════════════
// tibet-dgx — Zero-Trust DGX CLI
//
// Cross-machine AI inference without NVLink/Infiniband.
// Combines RAM RAID-0 + DIME Aperture + ClusterMux into
// a simple CLI for serving and loading LLMs across machines.
//
// Usage:
//   tibet-dgx serve [--bind 0.0.0.0:4432]
//   tibet-dgx load <gguf-path> [--endpoint 10.0.100.1:4432]
//   tibet-dgx info <gguf-path>
//   tibet-dgx bench <endpoint>
// ═══════════════════════════════════════════════════════════════

use clap::{Parser, Subcommand};
use tibet_trust_kernel::cluster_mux::*;
use tibet_trust_kernel::cluster_transport::{BlockStore, sha256_hex};
use tibet_trust_kernel::llm_mapper::ModelManifest;
use tibet_trust_kernel::ram_raid::RAID_BLOCK_SIZE;
#[cfg(feature = "quic")]
use tibet_trust_kernel::quic_mux::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use std::io::Read;

#[derive(Parser)]
#[command(name = "tibet-dgx")]
#[command(about = "Zero-Trust DGX — Run LLMs across machines without NVLink")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start RAM B server (run on the remote machine)
    Serve {
        /// Bind address
        #[arg(long, default_value = "0.0.0.0:4432")]
        bind: String,

        /// Use QUIC transport (multi-stream, no HOL blocking)
        #[arg(long)]
        quic: bool,
    },

    /// Load a GGUF model into cross-machine RAID-0
    Load {
        /// Path to GGUF model file
        path: String,

        /// Remote RAM B endpoint
        #[arg(long, short, default_value = "127.0.0.1:4432")]
        endpoint: String,

        /// Verify all blocks after loading
        #[arg(long, default_value_t = true)]
        verify: bool,

        /// Use QUIC transport (multi-stream, no HOL blocking)
        #[arg(long)]
        quic: bool,
    },

    /// Show model info and RAID-0 distribution
    Info {
        /// Path to GGUF model file
        path: String,
    },

    /// Benchmark connection to RAM B
    Bench {
        /// Remote RAM B endpoint
        endpoint: String,

        /// Number of test blocks to send
        #[arg(long, default_value_t = 100)]
        blocks: usize,

        /// Use QUIC transport (multi-stream, no HOL blocking)
        #[arg(long)]
        quic: bool,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { bind, quic } => {
            #[cfg(feature = "quic")]
            if quic {
                cmd_serve_quic(&bind).await;
                return;
            }
            #[cfg(not(feature = "quic"))]
            if quic {
                eprintln!("QUIC not available — rebuild with --features quic");
                std::process::exit(1);
            }
            cmd_serve(&bind).await;
        }
        Commands::Load { path, endpoint, verify, quic } => {
            #[cfg(feature = "quic")]
            if quic {
                cmd_load_quic(&path, &endpoint, verify).await;
                return;
            }
            #[cfg(not(feature = "quic"))]
            if quic {
                eprintln!("QUIC not available — rebuild with --features quic");
                std::process::exit(1);
            }
            cmd_load(&path, &endpoint, verify).await;
        }
        Commands::Info { path } => cmd_info(&path),
        Commands::Bench { endpoint, blocks, quic } => {
            #[cfg(feature = "quic")]
            if quic {
                cmd_bench_quic(&endpoint, blocks).await;
                return;
            }
            #[cfg(not(feature = "quic"))]
            if quic {
                eprintln!("QUIC not available — rebuild with --features quic");
                std::process::exit(1);
            }
            cmd_bench(&endpoint, blocks).await;
        }
    }
}

async fn cmd_serve(bind: &str) {
    println!("tibet-dgx: starting RAM B server on {}", bind);
    println!("  Waiting for connections from RAM A...\n");

    let store = Arc::new(BlockStore::new());
    let listener = tokio::net::TcpListener::bind(bind).await
        .unwrap_or_else(|e| { eprintln!("Cannot bind to {}: {}", bind, e); std::process::exit(1); });

    println!("  Listening on {} — ready for RAID-0 block storage", bind);

    loop {
        match listener.accept().await {
            Ok((socket, peer)) => {
                socket.set_nodelay(true).ok();
                let s = store.clone();
                let frames = Arc::new(AtomicU64::new(0));
                println!("  Connection from {}", peer);
                tokio::spawn(async move {
                    let _ = handle_mux_connection(socket, s, "ram-b.aint", frames).await;
                    println!("  Disconnected: {}", peer);
                });
            }
            Err(e) => eprintln!("Accept error: {}", e),
        }
    }
}

async fn cmd_load(path: &str, endpoint: &str, verify: bool) {
    let file_size = std::fs::metadata(path)
        .unwrap_or_else(|e| { eprintln!("Cannot read {}: {}", path, e); std::process::exit(1); })
        .len() as usize;
    let num_blocks = (file_size + RAID_BLOCK_SIZE - 1) / RAID_BLOCK_SIZE;
    let remote_blocks = num_blocks / 2;

    let filename = path.split('/').last().unwrap_or(path);
    println!("tibet-dgx: loading {} ({:.2} GB, {} blocks)",
        filename, file_size as f64 / (1024.0 * 1024.0 * 1024.0), num_blocks);
    println!("  Local (RAM A):  {} blocks ({:.2} GB)", num_blocks - remote_blocks, (num_blocks - remote_blocks) as f64 * 2.0 / 1024.0);
    println!("  Remote (RAM B): {} blocks ({:.2} GB) → {}", remote_blocks, remote_blocks as f64 * 2.0 / 1024.0, endpoint);
    println!();

    let client = Arc::new(ClusterMuxClient::new(endpoint, "tibet-dgx.aint"));
    let rtt = client.ping().await
        .unwrap_or_else(|e| { eprintln!("Cannot connect to RAM B at {}: {}", endpoint, e); std::process::exit(1); });
    println!("  RAM B RTT: {}µs", rtt);

    // Store phase
    let mut file = std::fs::File::open(path).unwrap();
    let mut block_hashes: Vec<String> = Vec::with_capacity(num_blocks);
    let mut local_store: Vec<Vec<u8>> = Vec::new();
    let t0 = Instant::now();
    let mut bytes_remote = 0u64;

    for i in 0..num_blocks {
        let mut buf = vec![0u8; RAID_BLOCK_SIZE];
        let n = file.read(&mut buf).unwrap();
        buf.truncate(n);
        let hash = sha256_hex(&buf);
        block_hashes.push(hash.clone());

        if i % 2 == 0 {
            local_store.push(buf);
        } else {
            client.store_block(i, &buf, &hash, "tibet-dgx", n, i as u64).await.unwrap();
            bytes_remote += n as u64;
        }

        if (i + 1) % 200 == 0 || i == num_blocks - 1 {
            let pct = (i + 1) as f64 / num_blocks as f64 * 100.0;
            let elapsed = t0.elapsed().as_secs_f64().max(0.001);
            print!("\r  Loading: [{:>5.1}%] {}/{} blocks, {:.0} MB/s    ", pct, i + 1, num_blocks, bytes_remote as f64 / 1_000_000.0 / elapsed);
        }
    }
    let store_time = t0.elapsed();
    println!("\n  Stored in {:.1}s ({:.0} MB/s to RAM B)\n", store_time.as_secs_f64(), bytes_remote as f64 / 1_000_000.0 / store_time.as_secs_f64().max(0.001));

    // Verify phase
    if verify {
        print!("  Verifying...");
        let t1 = Instant::now();
        let mut ok = 0u32;
        let mut fail = 0u32;
        let mut local_idx = 0usize;

        for i in 0..num_blocks {
            let data = if i % 2 == 0 {
                let d = local_store[local_idx].clone();
                local_idx += 1;
                d
            } else {
                let (d, _) = client.fetch_block(i, Some(&block_hashes[i]), i as u64).await.unwrap();
                d
            };
            if sha256_hex(&data) == block_hashes[i] { ok += 1; } else { fail += 1; }
        }
        let vt = t1.elapsed();
        println!("\r  Verified: {}/{} OK in {:.1}s ({} failures)    \n", ok, num_blocks, vt.as_secs_f64(), fail);
    }

    let (hits, misses, ratio, saved) = client.hash_cache.stats();
    println!("  Hash cache: {} hits ({:.0}%), {:.1} MB SHA-256 saved", hits, ratio * 100.0, saved as f64 / 1_000_000.0);
    println!("\n  Model loaded. Ready for inference.");
}

fn cmd_info(path: &str) {
    let file_size = std::fs::metadata(path)
        .unwrap_or_else(|e| { eprintln!("Cannot read {}: {}", path, e); std::process::exit(1); })
        .len() as usize;
    let num_blocks = (file_size + RAID_BLOCK_SIZE - 1) / RAID_BLOCK_SIZE;
    let filename = path.split('/').last().unwrap_or(path);

    println!("tibet-dgx: model info");
    println!("  File:     {}", filename);
    println!("  Size:     {:.2} GB ({} bytes)", file_size as f64 / (1024.0 * 1024.0 * 1024.0), file_size);
    println!("  Blocks:   {} × 2MB", num_blocks);
    println!("  RAM A:    {} blocks ({:.2} GB)", (num_blocks + 1) / 2, ((num_blocks + 1) / 2) as f64 * 2.0 / 1024.0);
    println!("  RAM B:    {} blocks ({:.2} GB)", num_blocks / 2, (num_blocks / 2) as f64 * 2.0 / 1024.0);
    println!();

    // Memory scenarios
    let scenarios = [
        ("32 GB (single machine)", 32usize * 1024 / 2),
        ("64 GB (2 machines)", 64usize * 1024 / 2),
        ("96 GB (3 machines)", 96usize * 1024 / 2),
        ("128 GB (4 machines)", 128usize * 1024 / 2),
    ];
    println!("  Memory scenarios:");
    for (name, budget) in &scenarios {
        let pct = (*budget as f64 / num_blocks as f64 * 100.0).min(100.0);
        let fits = *budget >= num_blocks;
        println!("    {}: {:.0}% resident {}", name, pct, if fits { "-- fits!" } else { "" });
    }
}

async fn cmd_bench(endpoint: &str, blocks: usize) {
    println!("tibet-dgx: benchmarking connection to {}", endpoint);

    let client = Arc::new(ClusterMuxClient::new(endpoint, "tibet-dgx.aint"));

    // Ping
    let mut rtts = Vec::new();
    for _ in 0..10 {
        let rtt = client.ping().await.unwrap();
        rtts.push(rtt);
    }
    rtts.sort();
    println!("  Ping RTT: min={}µs, median={}µs, max={}µs", rtts[0], rtts[5], rtts[9]);

    // Store throughput
    let test_data: Vec<u8> = (0..RAID_BLOCK_SIZE).map(|i| (i % 256) as u8).collect();
    let hash = sha256_hex(&test_data);
    let t0 = Instant::now();
    for i in 0..blocks {
        client.store_block(i, &test_data, &hash, "bench", test_data.len(), i as u64).await.unwrap();
    }
    let store_time = t0.elapsed();
    let store_mbps = (blocks as f64 * RAID_BLOCK_SIZE as f64) / 1_000_000.0 / store_time.as_secs_f64();
    println!("  Store: {} blocks in {:.2}s ({:.0} MB/s)", blocks, store_time.as_secs_f64(), store_mbps);

    // Fetch throughput
    let t1 = Instant::now();
    for i in 0..blocks {
        let _ = client.fetch_block(i, Some(&hash), i as u64).await.unwrap();
    }
    let fetch_time = t1.elapsed();
    let fetch_mbps = (blocks as f64 * RAID_BLOCK_SIZE as f64) / 1_000_000.0 / fetch_time.as_secs_f64();
    println!("  Fetch: {} blocks in {:.2}s ({:.0} MB/s)", blocks, fetch_time.as_secs_f64(), fetch_mbps);

    let (hits, misses, ratio, saved) = client.hash_cache.stats();
    println!("  Cache: {} hits ({:.0}%), {} misses", hits, ratio * 100.0, misses);
    println!("  SHA-256 saved: {:.1} MB", saved as f64 / 1_000_000.0);
}

// ═══════════════════════════════════════════════════════════════
// QUIC variants — multi-stream, no head-of-line blocking
// ═══════════════════════════════════════════════════════════════

#[cfg(feature = "quic")]
async fn cmd_serve_quic(bind: &str) {
    println!("tibet-dgx: starting QUIC RAM B server on {}", bind);
    println!("  Transport: QUIC (multi-stream, 0-RTT, no HOL blocking)");
    println!("  Waiting for connections from RAM A...\n");

    let store = Arc::new(BlockStore::new());
    let addr: std::net::SocketAddr = bind.parse()
        .unwrap_or_else(|e| { eprintln!("Invalid bind address {}: {}", bind, e); std::process::exit(1); });
    let server = QuicMuxServer::new(addr, "ram-b.aint", store);

    server.serve().await
        .unwrap_or_else(|e| { eprintln!("QUIC server error: {}", e); std::process::exit(1); });
}

#[cfg(feature = "quic")]
async fn cmd_load_quic(path: &str, endpoint: &str, verify: bool) {
    let file_size = std::fs::metadata(path)
        .unwrap_or_else(|e| { eprintln!("Cannot read {}: {}", path, e); std::process::exit(1); })
        .len() as usize;
    let num_blocks = (file_size + RAID_BLOCK_SIZE - 1) / RAID_BLOCK_SIZE;
    let remote_blocks = num_blocks / 2;

    let filename = path.split('/').last().unwrap_or(path);
    println!("tibet-dgx: loading {} ({:.2} GB, {} blocks) via QUIC",
        filename, file_size as f64 / (1024.0 * 1024.0 * 1024.0), num_blocks);
    println!("  Transport: QUIC (multi-stream, no HOL blocking)");
    println!("  Local (RAM A):  {} blocks ({:.2} GB)", num_blocks - remote_blocks, (num_blocks - remote_blocks) as f64 * 2.0 / 1024.0);
    println!("  Remote (RAM B): {} blocks ({:.2} GB) -> {}", remote_blocks, remote_blocks as f64 * 2.0 / 1024.0, endpoint);
    println!();

    let client = Arc::new(QuicMuxClient::new(endpoint, "tibet-dgx.aint"));
    let rtt = client.ping().await
        .unwrap_or_else(|e| { eprintln!("Cannot connect to RAM B at {} (QUIC): {}", endpoint, e); std::process::exit(1); });
    println!("  RAM B RTT: {}us (QUIC)", rtt);

    // Store phase
    let mut file = std::fs::File::open(path).unwrap();
    let mut block_hashes: Vec<String> = Vec::with_capacity(num_blocks);
    let mut local_store: Vec<Vec<u8>> = Vec::new();
    let t0 = Instant::now();
    let mut bytes_remote = 0u64;

    for i in 0..num_blocks {
        let mut buf = vec![0u8; RAID_BLOCK_SIZE];
        let n = file.read(&mut buf).unwrap();
        buf.truncate(n);
        let hash = sha256_hex(&buf);
        block_hashes.push(hash.clone());

        if i % 2 == 0 {
            local_store.push(buf);
        } else {
            client.store_block(i, &buf, &hash, "tibet-dgx", n, i as u64).await.unwrap();
            bytes_remote += n as u64;
        }

        if (i + 1) % 200 == 0 || i == num_blocks - 1 {
            let pct = (i + 1) as f64 / num_blocks as f64 * 100.0;
            let elapsed = t0.elapsed().as_secs_f64().max(0.001);
            print!("\r  Loading: [{:>5.1}%] {}/{} blocks, {:.0} MB/s (QUIC)    ", pct, i + 1, num_blocks, bytes_remote as f64 / 1_000_000.0 / elapsed);
        }
    }
    let store_time = t0.elapsed();
    println!("\n  Stored in {:.1}s ({:.0} MB/s to RAM B via QUIC)\n", store_time.as_secs_f64(), bytes_remote as f64 / 1_000_000.0 / store_time.as_secs_f64().max(0.001));

    // Verify phase — use parallel batch fetch (QUIC advantage!)
    if verify {
        print!("  Verifying (parallel QUIC streams)...");
        let t1 = Instant::now();
        let mut ok = 0u32;
        let mut fail = 0u32;
        let mut local_idx = 0usize;

        // Batch fetch all remote blocks at once
        let remote_requests: Vec<(usize, u64)> = (0..num_blocks)
            .filter(|i| i % 2 != 0)
            .map(|i| (i, i as u64))
            .collect();

        let remote_results = client.fetch_batch_parallel(&remote_requests).await.unwrap();
        let mut remote_map: std::collections::HashMap<usize, Vec<u8>> = remote_results.into_iter()
            .map(|(idx, data, _)| (idx, data))
            .collect();

        for i in 0..num_blocks {
            let data = if i % 2 == 0 {
                let d = local_store[local_idx].clone();
                local_idx += 1;
                d
            } else {
                remote_map.remove(&i).unwrap()
            };
            if sha256_hex(&data) == block_hashes[i] { ok += 1; } else { fail += 1; }
        }
        let vt = t1.elapsed();
        println!("\r  Verified: {}/{} OK in {:.1}s ({} failures, parallel QUIC)    \n", ok, num_blocks, vt.as_secs_f64(), fail);
    }

    let (hits, _misses, ratio, saved) = client.hash_cache.stats();
    println!("  Hash cache: {} hits ({:.0}%), {:.1} MB SHA-256 saved", hits, ratio * 100.0, saved as f64 / 1_000_000.0);
    println!("\n  Model loaded via QUIC. Ready for inference.");
}

#[cfg(feature = "quic")]
async fn cmd_bench_quic(endpoint: &str, blocks: usize) {
    println!("tibet-dgx: benchmarking QUIC connection to {}", endpoint);
    println!("  Transport: QUIC (multi-stream, no HOL blocking)\n");

    let client = Arc::new(QuicMuxClient::new(endpoint, "tibet-dgx.aint"));

    // Ping
    let mut rtts = Vec::new();
    for _ in 0..10 {
        let rtt = client.ping().await.unwrap();
        rtts.push(rtt);
    }
    rtts.sort();
    println!("  Ping RTT: min={}us, median={}us, max={}us", rtts[0], rtts[5], rtts[9]);

    // Store throughput (sequential)
    let test_data: Vec<u8> = (0..RAID_BLOCK_SIZE).map(|i| (i % 256) as u8).collect();
    let hash = sha256_hex(&test_data);
    let t0 = Instant::now();
    for i in 0..blocks {
        client.store_block(i, &test_data, &hash, "bench", test_data.len(), i as u64).await.unwrap();
    }
    let store_time = t0.elapsed();
    let store_mbps = (blocks as f64 * RAID_BLOCK_SIZE as f64) / 1_000_000.0 / store_time.as_secs_f64();
    println!("  Store (sequential): {} blocks in {:.2}s ({:.0} MB/s)", blocks, store_time.as_secs_f64(), store_mbps);

    // Fetch throughput (sequential)
    let t1 = Instant::now();
    for i in 0..blocks {
        let _ = client.fetch_block(i, Some(&hash), i as u64).await.unwrap();
    }
    let fetch_time = t1.elapsed();
    let fetch_mbps = (blocks as f64 * RAID_BLOCK_SIZE as f64) / 1_000_000.0 / fetch_time.as_secs_f64();
    println!("  Fetch (sequential): {} blocks in {:.2}s ({:.0} MB/s)", blocks, fetch_time.as_secs_f64(), fetch_mbps);

    // Fetch throughput (PARALLEL — the QUIC advantage!)
    let requests: Vec<(usize, u64)> = (0..blocks).map(|i| (i, i as u64)).collect();
    let t2 = Instant::now();
    let results = client.fetch_batch_parallel(&requests).await.unwrap();
    let parallel_time = t2.elapsed();
    let parallel_mbps = (blocks as f64 * RAID_BLOCK_SIZE as f64) / 1_000_000.0 / parallel_time.as_secs_f64();
    println!("  Fetch (parallel):   {} blocks in {:.2}s ({:.0} MB/s)  <-- QUIC multi-stream!", blocks, parallel_time.as_secs_f64(), parallel_mbps);

    let speedup = fetch_time.as_secs_f64() / parallel_time.as_secs_f64();
    println!("\n  Parallel speedup:   {:.1}x faster than sequential", speedup);

    let (hits, misses, ratio, saved) = client.hash_cache.stats();
    println!("  Cache: {} hits ({:.0}%), {} misses", hits, ratio * 100.0, misses);
    println!("  SHA-256 saved: {:.1} MB", saved as f64 / 1_000_000.0);
    println!("  Streams opened: {}", client.streams_opened.load(Ordering::Relaxed));
}
