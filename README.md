# tibet-dgx

Run LLMs larger than your RAM — across machines, without NVLink.

`tibet-dgx` transparently maps AI model weights across multiple servers using encrypted RAID-0 over TCP. Load a 70B model on two 64GB machines. No special hardware required — just Ethernet.

## How it works

```
Machine A (64GB RAM)              Machine B (64GB RAM)
┌─────────────────────┐          ┌─────────────────────┐
│  Even blocks (local)│◄────────►│  Odd blocks (remote) │
│  20GB of 70B model  │  10Gbps  │  20GB of 70B model   │
│                     │  TCP     │                      │
│  tibet-dgx load     │          │  tibet-dgx serve     │
└─────────────────────┘          └─────────────────────┘
         Combined: 128GB virtual RAM
         70B Q4_K_M (40GB) fits entirely
```

Blocks start as unmapped spaceholders (**DIME Aperture** pattern). On first access, the block materializes from the remote machine. Second access hits the hash cache — SHA-256 verification skipped, 14x faster.

## Tested performance

Real hardware: HP Z2 P520 ↔ HP DL360 Gen10, Intel X540-AT2 10Gbps direct link.

| Metric | Result |
|--------|--------|
| Ping RTT | 0.11ms |
| Fetch throughput | **830 MB/s** |
| Store throughput | 95 MB/s |
| Qwen 7B (4.36GB) | 2234/2234 blocks verified |
| Hash cache | 100% hit rate |
| Integrity failures | 0 |

## Install

```bash
cargo install tibet-dgx
```

Or from source:

```bash
git clone https://github.com/Humotica/tibet-dgx
cd tibet-dgx
cargo build --release
```

## Usage

**On the remote machine** (RAM B — stores the other half):

```bash
tibet-dgx serve
# Listening on 0.0.0.0:4432
```

**On the local machine** (RAM A — runs inference):

```bash
# Check model distribution
tibet-dgx info /path/to/llama-70B-Q4_K_M.gguf

# Load model across both machines
tibet-dgx load /path/to/model.gguf --endpoint 10.0.100.1:4432

# Benchmark the connection
tibet-dgx bench 10.0.100.1:4432
```

### Example output

```
tibet-dgx: model info
  File:     llama-70B-Q4_K_M.gguf
  Size:     40.00 GB
  Blocks:   20480 × 2MB
  RAM A:    10240 blocks (20.00 GB)
  RAM B:    10240 blocks (20.00 GB)

  Memory scenarios:
    32 GB (single machine): 80% resident
    64 GB (2 machines): 100% resident -- fits!
    128 GB (4 machines): 100% resident -- fits!
```

## What happens under the hood

1. Model file is split into 2MB blocks
2. Even blocks stay local (RAM A), odd blocks go to the remote server (RAM B) via `ClusterMux`
3. Every block is SHA-256 verified on first transfer
4. Hash cache remembers verified blocks — subsequent access skips SHA-256 (14x speedup)
5. All transport is encrypted via AES-256-GCM (powered by [tibet-trust-kernel](https://crates.io/crates/tibet-trust-kernel))

## Requirements

- Two (or more) Linux machines with TCP connectivity
- 10Gbps Ethernet recommended (works on any speed, just slower)
- No NVLink, no Infiniband, no special hardware
- Rust 1.70+

## Part of TIBET

tibet-dgx is powered by [tibet-trust-kernel](https://crates.io/crates/tibet-trust-kernel) — the zero-trust security foundation that handles encryption, integrity verification, and cross-machine transport.

Built by [Humotica](https://humotica.com) for the [AInternet](https://ainternet.org).

## License

MIT
