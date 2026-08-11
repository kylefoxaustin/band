# BAND: Bandwidth Assessment for Native DDR

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Introduction

BAND (Bandwidth Assessment for Native DDR) is a portable memory-bandwidth
measurement tool. It runs the four classic [STREAM](https://www.cs.virginia.edu/stream/)
kernels — **Copy, Scale, Add, Triad** — and reports sustained memory bandwidth
in decimal GB/s, directly comparable to STREAM.C output.

The hard part of measuring "DDR bandwidth" portably is that true memory‑controller
counters live behind privileged, per‑SoC interfaces (Intel uncore IMC, AMD µProf,
ARM PMU, Apple `powermetrics`) that differ on every platform. BAND's answer is to
**measure at several levels and always tell you which level produced each number**:

| Tier | Backend | Requires | What it measures |
|------|---------|----------|------------------|
| **0** | NumPy | nothing (Python + NumPy) | bandwidth **achievable from Python** |
| **1** | Native | a C compiler | bandwidth **achievable from compiled code** (≈ hardware achievable) |
| **2** | DRAM counters | Linux `perf` + privileges | **actual DRAM traffic** at the memory controller |

Higher tiers are more accurate but less portable. BAND auto‑uses whatever is
available and **degrades gracefully** — on a bare machine you still get Tier 0,
on a machine with a compiler you also get Tier 1, and with privileges you get the
ground‑truth Tier 2 number.

In addition, a **working‑set sweep** runs a single‑core kernel across array sizes
from sub‑L1 up past L3. The plateau at large sizes is a *measured* estimate of
per‑core DRAM bandwidth — no assumptions about cache size required.

## Why this is more honest than "Python ≈ STREAM"

Earlier versions of BAND tried to convince you that NumPy results matched STREAM.C.
That comparison is fragile for two real reasons:

1. **NumPy does not emit non‑temporal (streaming) stores.** A logical "write" of N
   bytes can cost a hidden read‑for‑ownership, so real DRAM traffic exceeds the
   byte count STREAM reports. Whether that matters depends on the CPU and compiler.
2. **A naive NumPy Triad allocates a temporary array**, so it measures allocator
   churn, not memory bandwidth. (The previous version "fixed" this by *doubling*
   the Triad number — which simply hid the bug.)

BAND now computes the Triad in place with a cache‑resident temporary (no large
intermediate allocation), so Tier 0 Triad is a real bandwidth figure rather than
an artifact. And rather than pretend Python equals C, BAND shows Tier 0 **and**
Tier 1 side by side so you can see the gap directly.

## Getting Started

### Prerequisites

- Python 3.7+
- NumPy (required)
- psutil (optional — used only for the total‑memory line in system info)
- A C compiler (optional — enables Tier 1; `cc`, `gcc`, or `clang`)
- Linux `perf` + privileges (optional — enables Tier 2)

```bash
git clone https://github.com/kylefoxaustin/band.git
cd band
pip install -r requirements.txt
chmod +x band.py
```

## How to Run

```bash
./band.py                       # default: 2 GB total, all CPUs (capped at 8)
./band.py --size 4 --threads 8  # larger working set, 8 threads
./band.py --dram-counters       # also attempt Tier 2 (needs perf + privileges)
```

### Command‑line options

```
--size FLOAT          Total memory in GB across all three arrays (default: 2.0)
--iterations INT      Timed iterations per operation (default: 7)
--threads INT         Worker threads (default: all available CPUs, capped at 8)
--no-pin              Do not pin threads to CPUs
--no-numpy            Skip Tier 0
--no-native           Skip Tier 1
--dram-counters       Attempt Tier 2 (DRAM counters via perf)
--dram-driver CHOICE  Workload driving the Tier 2 window: auto|native|numpy
                      (auto = native if available, else numpy)
--no-sweep            Skip the working-set sweep
--verbose             Show the compiler command and extra diagnostics
--stream-file PATH    Compare against a STREAM.C output file
--peak-mts FLOAT      Memory transfer rate (e.g. 6000 for DDR5-6000)
--channels INT        Populated memory channels  (with --peak-mts: % of peak)
```

### Reading the output

> **The numbers below are ILLUSTRATIVE — they show the shape of the output, not a
> measurement.** They are not from a recorded run on a known machine, so they are not
> comparable to your results and must not be cited. Bandwidth figures are meaningless
> without the CPU, memory configuration and thread count that produced them; when you
> report your own, report those alongside.

```
[Tier 0] NumPy  (achievable from Python)
  Copy   72.51   Scale 97.28   Add 54.72   Triad 70.81  (GB/s, median)

[Tier 1] Native  (achievable from compiled C + OpenMP)
  Copy   78.06   Scale 79.22   Add 78.37   Triad 78.09  (GB/s, best-of-reps)

Working-set sweep (single core, Copy kernel)
   512.0 KB   153.55  ########################################   <- in L2
     2.0 MB    61.92  ################                           <- exceeds L2
   256.0 MB    52.07  ##############                             <- DRAM plateau
  Estimated per-core DRAM bandwidth (plateau): ~52 GB/s
```

A few things worth noting in real output:

- Each result shows **median, min, max, and CV%** (coefficient of variation).
  A `<- high variance` flag appears when CV > 5%, which usually means CPU
  frequency scaling, thermal throttling, or a busy machine — pin the clocks and
  close other workloads for stable numbers.
- When the Tier 1 native kernel is bus‑bound, **all four operations converge to
  roughly the same GB/s** — that is the correct signature of saturated memory
  bandwidth. Tier 0 spreads out more because it also reflects NumPy's per‑operation
  efficiency.
- The sweep makes the cache hierarchy visible: bandwidth peaks while the working
  set fits in L1/L2, then steps down to the DRAM plateau.

## Tier 2: measuring true DRAM traffic (Linux)

Tier 2 reads the CPU's uncore IMC counters via `perf` while a sustained workload
runs, giving the **actual bytes moved across the memory bus** — including the
read‑for‑ownership traffic the logical byte count misses.

```bash
# perf must be allowed to read uncore counters:
sudo sysctl kernel.perf_event_paranoid=0      # or run band.py under sudo
./band.py --dram-counters
```

BAND probes several IMC event sets and uses the first the kernel actually returns
numbers for:

- `uncore_imc_free_running/data_read/` + `data_write/` (client / Raptor Lake etc.)
- `unc_m_cas_count_rd` + `unc_m_cas_count_wr` (per‑controller CAS counts)
- older `uncore_imc/cas_count_*` and `uncore_imc/data_*` spellings

perf reports some counters pre‑scaled (e.g. in MiB) and others as raw cache‑line
counts; BAND reads perf's unit column and converts accordingly (raw counts ×64 B).
While a saturating multi‑threaded Triad runs in‑process, perf counts DRAM traffic
system‑wide over a fixed window, so the figure is true bus traffic — it includes
read‑for‑ownership and can legitimately differ from the logical Tier 0/1 numbers.
If no event set works, or privileges are missing, Tier 2 is skipped with an
explanation.

By default the counting window is driven by the **native** kernel (`--dram-driver
auto`), which sustains traffic closest to the hardware ceiling; use
`--dram-driver numpy` to measure the DRAM traffic of a NumPy workload instead.
A useful side observation: when the native kernel's measured DRAM bandwidth
roughly equals its *logical* Tier 1 number, the compiler is emitting non‑temporal
(streaming) stores — the write bypasses cache, so there is no read‑for‑ownership
inflation.

> Tier 2 is currently Linux/`perf`‑only and has been verified on Intel Raptor Lake.
> On other platforms/microarchitectures it falls back gracefully rather than
> reporting a guessed number.

## Comparing with STREAM.C

The included `setup_stream.sh` downloads, compiles, and runs the reference
STREAM benchmark:

```bash
./setup_stream.sh                          # writes stream_results.txt
./band.py --stream-file stream_results.txt
```

BAND parses the STREAM table and prints a side‑by‑side comparison of STREAM.C,
BAND's native (Tier 1), and NumPy (Tier 0) results, with each tier's share of
STREAM.C. Units are decimal throughout (1 GB/s = 1000 MB/s), so the percentages
are apples‑to‑apples.

## Units and methodology

- **Decimal units**, matching STREAM: 1 GB = 1e9 bytes; bandwidth = bytes / time / 1e9.
- **`time.perf_counter()`** for Tier 0; Tier 1 times inside C with `omp_get_wtime`,
  excluding all Python overhead and reporting best‑of‑reps.
- **Logical traffic counted per element**: Copy/Scale = 16 B (1 read + 1 write),
  Add/Triad = 24 B (2 reads + 1 write) — identical to STREAM's accounting.
- **Thread pinning + parallel first‑touch** so pages are allocated on the NUMA
  node that uses them; NumPy ufuncs release the GIL, so threading yields real
  parallelism for large arrays.
- The default `--size` (2 GB) far exceeds any current CPU cache, so the steady
  bandwidth reflects DRAM, not cache. Use the sweep to confirm you've exceeded
  cache on your hardware.

## Performance tips

- **Scan thread counts** (`--threads 1,2,4,8,…` across runs) to find where
  bandwidth saturates — it is often well below your core count, set by the number
  of memory channels.
- **Match threads to memory channels** rather than to cores.
- **Pin CPU frequency** (disable turbo / set the `performance` governor) and close
  other memory‑heavy processes to drop the CV%.
- Pass `--peak-mts` and `--channels` to see your result as a percentage of
  theoretical peak, e.g. `--peak-mts 6000 --channels 2` for dual‑channel DDR5‑6000
  (= 96 GB/s peak).

## Attestation

Maintained by Kyle Fox ([@kylefoxaustin](https://github.com/kylefoxaustin)).
Intended for educational and performance‑measurement purposes. Contributions,
bug reports, and feature requests are welcome.

## License

MIT — see the LICENSE file.
