#!/usr/bin/env python3
"""
BAND: Bandwidth Assessment for Native DDR

A portable memory-bandwidth measurement tool.

BAND measures sustained memory bandwidth using the four classic STREAM
kernels (Copy, Scale, Add, Triad). It does so through a *tiered* set of
backends and reports which tier produced each number, so you always know
how close the figure is to true DRAM bandwidth:

  Tier 0  NumPy        Always available. Pure Python + NumPy, no compiler,
                       no privileges. Measures the bandwidth *achievable
                       from Python*.
  Tier 1  Native       Used when a C compiler is present. Compiles a small
                       STREAM-style kernel at runtime and runs it with
                       OpenMP. Measures bandwidth *achievable from native
                       code* -- typically close to the hardware peak.
  Tier 2  DRAM counters Used when privileged hardware counters are available
                       (Linux uncore IMC via perf). Measures *actual DRAM
                       traffic* at the memory controller, independent of
                       read-for-ownership / non-temporal-store effects.

In addition, a working-set sweep runs a single-core kernel across sizes from
sub-L1 up past L3 and reports the bandwidth curve. The high-size plateau is a
direct, *measured* estimate of per-core DRAM bandwidth -- no assumptions about
cache size required.

Units follow the STREAM convention: bandwidth is reported in decimal GB/s
(1 GB = 1e9 bytes), so figures are directly comparable to STREAM.C output.
"""

import argparse
import ctypes
import hashlib
import os
import platform
import shutil
import statistics
import subprocess
import tempfile
import threading
from datetime import datetime
from time import perf_counter, sleep

import numpy as np

try:
    import psutil
except ImportError:  # psutil is optional; only used for system info
    psutil = None


# --------------------------------------------------------------------------
# Constants and operation definitions
# --------------------------------------------------------------------------

GB = 1_000_000_000          # decimal gigabyte (STREAM convention)
ELEM_BYTES = 8              # float64

# STREAM kernels. Array roles follow stream.c exactly:
#   Copy : c = a              (read a, write c)
#   Scale: b = scalar * c     (read c, write b)
#   Add  : c = a + b          (read a, read b, write c)
#   Triad: a = b + scalar*c   (read b, read c, write a)
#
# "bytes" is the logical traffic STREAM counts per element (reads + writes).
OPERATIONS = {
    "Copy":  {"bytes": 2 * ELEM_BYTES},
    "Scale": {"bytes": 2 * ELEM_BYTES},
    "Add":   {"bytes": 3 * ELEM_BYTES},
    "Triad": {"bytes": 3 * ELEM_BYTES},
}
OP_ORDER = ["Copy", "Scale", "Add", "Triad"]

SCALAR = 3.0
# Per-chunk temporary for the NumPy Triad. Sized to stay resident in L2 so the
# temporary's traffic does not leak to DRAM (keeps logical == real traffic).
NUMPY_TRIAD_CHUNK = 32 * 1024  # elements -> 256 KiB


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0


def available_cpus():
    """Logical CPUs this process may run on (respects cgroup/affinity)."""
    if hasattr(os, "sched_getaffinity"):
        try:
            return sorted(os.sched_getaffinity(0))
        except OSError:
            pass
    n = os.cpu_count() or 1
    return list(range(n))


def summarize(bandwidths):
    """Reduce a list of per-iteration GB/s figures to summary stats."""
    if not bandwidths:
        return None
    vals = list(bandwidths)
    mean = statistics.fmean(vals)
    stdev = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return {
        "median": statistics.median(vals),
        "mean": mean,
        "min": min(vals),
        "max": max(vals),
        "stdev": stdev,
        "cv": (stdev / mean * 100.0) if mean else 0.0,
        "n": len(vals),
        "raw": vals,
    }


# --------------------------------------------------------------------------
# System information
# --------------------------------------------------------------------------

def detect_caches():
    """Return a list of (level, type, size_bytes) from sysfs (Linux)."""
    caches = []
    base = "/sys/devices/system/cpu/cpu0/cache"
    try:
        for entry in sorted(os.listdir(base)):
            d = os.path.join(base, entry)
            try:
                level = int(open(os.path.join(d, "level")).read().strip())
                ctype = open(os.path.join(d, "type")).read().strip()
                raw = open(os.path.join(d, "size")).read().strip()
            except (OSError, ValueError):
                continue
            mult = 1
            if raw.endswith("K"):
                mult, raw = 1024, raw[:-1]
            elif raw.endswith("M"):
                mult, raw = 1024 * 1024, raw[:-1]
            try:
                caches.append((level, ctype, int(raw) * mult))
            except ValueError:
                continue
    except OSError:
        pass
    return caches


def get_system_info():
    info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "system": platform.system(),
        "architecture": platform.machine(),
        "cpu_count": os.cpu_count() or 1,
        "available_cpus": len(available_cpus()),
    }
    if psutil is not None:
        info["memory_gb"] = psutil.virtual_memory().total / (1024 ** 3)

    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass

    info["caches"] = detect_caches()
    return info


def largest_cache_bytes(caches):
    return max((sz for _, _, sz in caches), default=0)


# --------------------------------------------------------------------------
# Tier 0: NumPy backend
# --------------------------------------------------------------------------

def _numpy_kernel(op, a, b, c, tmp):
    """Run one STREAM kernel on the given (per-thread) arrays, in place."""
    if op == "Copy":
        np.copyto(c, a)
    elif op == "Scale":
        np.multiply(c, SCALAR, out=b)
    elif op == "Add":
        np.add(a, b, out=c)
    elif op == "Triad":
        # a = b + scalar*c, computed chunk-wise with a cache-resident temp so
        # no large intermediate array is allocated (which would otherwise
        # dominate the measurement -- the classic NumPy-Triad pitfall).
        n = a.shape[0]
        step = tmp.shape[0]
        for i in range(0, n, step):
            j = min(i + step, n)
            k = j - i
            np.multiply(c[i:j], SCALAR, out=tmp[:k])
            np.add(b[i:j], tmp[:k], out=a[i:j])
    else:
        raise ValueError(f"unknown op {op}")


class NumpyPool:
    """Persistent pool of pinned worker threads sharing barrier-synced work.

    Threads are created once, pinned to distinct CPUs, and each allocates and
    first-touches its own array slice (so pages land on the local NUMA node).
    `run_once` triggers one synchronized pass and returns the wall-clock time.
    """

    def __init__(self, total_elems, threads, pin=True):
        self.threads = threads
        self.pin = pin and hasattr(os, "sched_setaffinity")
        self.cpus = available_cpus()

        base = total_elems // threads
        rem = total_elems - base * threads
        self.slice_sizes = [base + (1 if i < rem else 0) for i in range(threads)]
        self.total_elems = total_elems

        self._cmd = None
        self._stop = False
        self._alloc_err = [None] * threads
        self._start = threading.Barrier(threads + 1)
        self._end = threading.Barrier(threads + 1)
        self._ready = threading.Barrier(threads + 1)

        self._workers = []
        for tid in range(threads):
            t = threading.Thread(target=self._worker, args=(tid,), daemon=True)
            t.start()
            self._workers.append(t)
        self._ready.wait()  # wait for all allocations to complete
        err = next((e for e in self._alloc_err if e), None)
        if err:
            raise err

    def _worker(self, tid):
        if self.pin and self.cpus:
            try:
                os.sched_setaffinity(0, {self.cpus[tid % len(self.cpus)]})
            except OSError:
                pass
        try:
            n = self.slice_sizes[tid]
            a = np.ones(n, dtype=np.float64)
            b = np.full(n, 2.0, dtype=np.float64)
            c = np.zeros(n, dtype=np.float64)
            tmp = np.empty(min(n, NUMPY_TRIAD_CHUNK) or 1, dtype=np.float64)
        except Exception as exc:  # noqa: BLE001 - report allocation failures
            self._alloc_err[tid] = exc
            self._ready.wait()
            return
        self._ready.wait()

        while True:
            self._start.wait()
            if self._stop:
                return
            _numpy_kernel(self._cmd, a, b, c, tmp)
            self._end.wait()

    def run_once(self, op):
        self._cmd = op
        t0 = perf_counter()
        self._start.wait()
        self._end.wait()
        return perf_counter() - t0

    def close(self):
        self._stop = True
        try:
            self._start.wait()
        except threading.BrokenBarrierError:
            pass


def measure_numpy(op, total_elems, threads, iterations, pin=True):
    """Return summary stats (GB/s) for one op via the NumPy backend."""
    pool = NumpyPool(total_elems, threads, pin=pin)
    try:
        pool.run_once(op)  # warm-up (not recorded)
        bw = []
        for _ in range(iterations):
            elapsed = pool.run_once(op)
            if elapsed > 0:
                bytes_moved = OPERATIONS[op]["bytes"] * total_elems
                bw.append(bytes_moved / elapsed / GB)
    finally:
        pool.close()
    return summarize(bw)


# --------------------------------------------------------------------------
# Tier 1: Native (runtime-compiled C) backend
# --------------------------------------------------------------------------

_NATIVE_SOURCE = r"""
#include <stdlib.h>
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct { double *a, *b, *c; long n; } band_state;

static double now(void) {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}

band_state* band_init(long n) {
    band_state* s = (band_state*)malloc(sizeof(band_state));
    if (!s) return NULL;
    s->n = n;
    s->a = (double*)aligned_alloc(64, (size_t)n * sizeof(double));
    s->b = (double*)aligned_alloc(64, (size_t)n * sizeof(double));
    s->c = (double*)aligned_alloc(64, (size_t)n * sizeof(double));
    if (!s->a || !s->b || !s->c) { return s; }
    /* Parallel first-touch so pages land on the node that will use them. */
    #pragma omp parallel for
    for (long i = 0; i < n; i++) { s->a[i] = 1.0; s->b[i] = 2.0; s->c[i] = 0.0; }
    return s;
}

void band_free(band_state* s) {
    if (!s) return;
    free(s->a); free(s->b); free(s->c); free(s);
}

/* op: 0=Copy 1=Scale 2=Add 3=Triad. Returns the minimum elapsed time (s). */
double band_run(band_state* s, int op, int reps) {
    const double scalar = 3.0;
    long n = s->n;
    double *a = s->a, *b = s->b, *c = s->c;
    double best = 1e300;
    for (int r = 0; r < reps; r++) {
        double t0 = now();
        switch (op) {
        case 0:
            #pragma omp parallel for
            for (long i = 0; i < n; i++) c[i] = a[i];
            break;
        case 1:
            #pragma omp parallel for
            for (long i = 0; i < n; i++) b[i] = scalar * c[i];
            break;
        case 2:
            #pragma omp parallel for
            for (long i = 0; i < n; i++) c[i] = a[i] + b[i];
            break;
        case 3:
            #pragma omp parallel for
            for (long i = 0; i < n; i++) a[i] = b[i] + scalar * c[i];
            break;
        }
        double dt = now() - t0;
        if (dt < best) best = dt;
    }
    return best;
}

void band_set_threads(int t) {
#ifdef _OPENMP
    omp_set_num_threads(t);
#else
    (void)t;
#endif
}
"""


def _find_compiler():
    for cc in ("cc", "gcc", "clang"):
        path = shutil.which(cc)
        if path:
            return path
    return None


class NativeBackend:
    """Compiles and loads the native STREAM kernel; None if unavailable."""

    def __init__(self, lib_path, threads):
        self._lib = ctypes.CDLL(lib_path)
        self._lib.band_init.restype = ctypes.c_void_p
        self._lib.band_init.argtypes = [ctypes.c_long]
        self._lib.band_free.argtypes = [ctypes.c_void_p]
        self._lib.band_run.restype = ctypes.c_double
        self._lib.band_run.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self._lib.band_set_threads.argtypes = [ctypes.c_int]
        self._threads = threads
        self._lib.band_set_threads(threads)

    @classmethod
    def build(cls, threads, openmp=True, verbose=False):
        cc = _find_compiler()
        if not cc:
            return None, "no C compiler found (looked for cc/gcc/clang)"

        base_flags = ["-O3", "-fPIC", "-shared", "-funroll-loops"]
        # -march=native helps the compiler emit wide SIMD / streaming stores.
        march = ["-march=native"]
        omp = ["-fopenmp"] if openmp else []

        cache_dir = os.path.join(tempfile.gettempdir(), "band_native")
        os.makedirs(cache_dir, exist_ok=True)
        key = hashlib.sha1(
            (_NATIVE_SOURCE + cc + " ".join(base_flags + march + omp)
             + platform.platform()).encode()
        ).hexdigest()[:16]
        lib_path = os.path.join(cache_dir, f"band_{key}.so")
        src_path = os.path.join(cache_dir, f"band_{key}.c")

        if not os.path.exists(lib_path):
            with open(src_path, "w") as f:
                f.write(_NATIVE_SOURCE)
            # Try with -march=native first, then without (some toolchains/VMs
            # reject it), then without OpenMP.
            attempts = [
                [cc, *base_flags, *march, *omp, src_path, "-o", lib_path],
                [cc, *base_flags, *omp, src_path, "-o", lib_path],
                [cc, *base_flags, src_path, "-o", lib_path],
            ]
            last_err = ""
            for cmd in attempts:
                try:
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    last_err = str(exc)
                    continue
                if r.returncode == 0 and os.path.exists(lib_path):
                    if verbose:
                        print(f"  (compiled with: {' '.join(cmd)})")
                    break
                last_err = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "unknown error"
            else:
                return None, f"compilation failed: {last_err}"

        try:
            return cls(lib_path, threads), None
        except OSError as exc:
            return None, f"could not load compiled library: {exc}"

    def measure(self, op, per_array_elems, reps):
        op_idx = {"Copy": 0, "Scale": 1, "Add": 2, "Triad": 3}[op]
        state = self._lib.band_init(ctypes.c_long(per_array_elems))
        if not state:
            raise MemoryError("native band_init failed (out of memory?)")
        try:
            best = self._lib.band_run(ctypes.c_void_p(state), op_idx, reps)
        finally:
            self._lib.band_free(ctypes.c_void_p(state))
        if best <= 0:
            return None
        bw = OPERATIONS[op]["bytes"] * per_array_elems / best / GB
        # Native backend reports best-of-reps as a single robust figure.
        return summarize([bw])

    def make_driver(self, op, per_array_elems):
        """Return a driver_fn(stop_event) that runs `op` until stopped.

        Used to drive the Tier 2 counting window with the native kernel, which
        sustains memory traffic closer to the hardware ceiling than NumPy.
        """
        op_idx = {"Copy": 0, "Scale": 1, "Add": 2, "Triad": 3}[op]
        lib = self._lib
        nthreads = self._threads

        def driver(stop):
            # OpenMP's thread-count ICV is per-thread; set it here so the kernel
            # runs multi-threaded on this (spawned) driver thread, not on the
            # single thread inherited from OMP_NUM_THREADS=1.
            lib.band_set_threads(nthreads)
            state = lib.band_init(ctypes.c_long(per_array_elems))
            if not state:
                return
            try:
                while not stop.is_set():
                    lib.band_run(ctypes.c_void_p(state), op_idx, 8)
            finally:
                lib.band_free(ctypes.c_void_p(state))

        return driver


# --------------------------------------------------------------------------
# Tier 2: DRAM counters (Linux uncore IMC via perf)
# --------------------------------------------------------------------------

# Candidate IMC event sets, probed in order. Names vary by microarchitecture:
# client (desktop/laptop) free-running counters, per-controller CAS counts, and
# older generic spellings. The first set perf actually returns numbers for wins.
_PERF_EVENT_SETS = [
    ("imc-free-running", ["uncore_imc_free_running/data_read/",
                          "uncore_imc_free_running/data_write/"]),
    ("imc-cas", ["unc_m_cas_count_rd", "unc_m_cas_count_wr"]),
    ("imc-cas-slash", ["uncore_imc/cas_count_read/", "uncore_imc/cas_count_write/"]),
    ("imc-data-slash", ["uncore_imc/data_reads/", "uncore_imc/data_writes/"]),
]

# perf may report a pre-scaled unit (e.g. "MiB") or a raw event count. Map the
# unit to bytes; an empty/unknown unit means a raw CAS-style count (64 B/line).
_UNIT_BYTES = {"B": 1, "KiB": 1024, "MiB": 1024 ** 2, "GiB": 1024 ** 3,
               "KB": 1e3, "MB": 1e6, "GB": 1e9}


def _counter_to_bytes(value, unit):
    unit = (unit or "").strip()
    if unit in _UNIT_BYTES:
        return value * _UNIT_BYTES[unit]
    return value * 64  # raw cache-line count


def _perf_available():
    if platform.system() != "Linux":
        return False, "DRAM counters supported on Linux only"
    if not shutil.which("perf"):
        return False, "perf not installed"
    try:
        paranoid = int(open("/proc/sys/kernel/perf_event_paranoid").read().strip())
    except (OSError, ValueError):
        paranoid = None
    if paranoid is not None and paranoid > 0 and os.geteuid() != 0:
        return False, (f"perf_event_paranoid={paranoid} blocks uncore counters; "
                       "rerun with sudo or set it to 0")
    return True, None


def _parse_perf_bytes(stderr, events):
    """Sum DRAM bytes from perf -x, output for the given event names."""
    total, found = 0.0, 0
    for line in stderr.splitlines():
        parts = line.split(",")
        if len(parts) < 3 or parts[2] not in events:
            continue
        try:
            val = float(parts[0].replace(" ", ""))
        except ValueError:
            continue  # "<not supported>" / "<not counted>"
        total += _counter_to_bytes(val, parts[1])
        found += 1
    return total, found


def _probe_perf_events():
    """Find an IMC event set perf actually returns numeric values for."""
    for label, events in _PERF_EVENT_SETS:
        cmd = ["perf", "stat", "-a", "-x", ",", "-e", ",".join(events),
               "--", "sleep", "0.3"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.TimeoutExpired):
            continue
        _, found = _parse_perf_bytes(r.stderr, events)
        if found:
            return label, events
    return None


def measure_dram_counters_live(driver_fn, duration, event_set):
    """Count system-wide DRAM traffic while a workload saturates memory.

    `driver_fn(stop_event)` runs a saturating workload in this process until
    `stop_event` is set. perf counts all DRAM traffic system-wide over a fixed
    window, so the result reflects true bus traffic (including read-for-ownership).
    """
    label, events = event_set
    stop = threading.Event()

    th = threading.Thread(target=driver_fn, args=(stop,), daemon=True)
    th.start()
    sleep(0.3)  # reach steady state before the counting window opens

    cmd = ["perf", "stat", "-a", "-x", ",", "-e", ",".join(events),
           "--", "sleep", str(duration)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        stop.set()
        th.join(timeout=5)
        return None, f"perf run failed: {exc}"

    stop.set()
    th.join(timeout=5)

    total_bytes, found = _parse_perf_bytes(r.stderr, events)
    if found == 0:
        return None, "no counter values parsed from perf output"
    return {"bandwidth": total_bytes / duration / GB, "label": label,
            "total_gb": total_bytes / GB}, None


# --------------------------------------------------------------------------
# Working-set sweep
# --------------------------------------------------------------------------

def working_set_sweep(max_array_bytes, reps_target_bytes=128 * 1024 * 1024):
    """Single-core Copy bandwidth across array sizes (cache hierarchy curve).

    Returns a list of (array_bytes, gb_per_s). The plateau at large sizes is a
    measured estimate of single-core DRAM bandwidth.
    """
    saved_affinity = None
    if hasattr(os, "sched_setaffinity") and available_cpus():
        try:
            saved_affinity = os.sched_getaffinity(0)
            os.sched_setaffinity(0, {available_cpus()[0]})
        except OSError:
            saved_affinity = None

    max_elems = max(max_array_bytes // ELEM_BYTES, 1024)
    a = np.ones(max_elems, dtype=np.float64)
    c = np.empty(max_elems, dtype=np.float64)

    results = []
    size = 2 * 1024  # start at 2 KiB per array
    while size * ELEM_BYTES <= max_array_bytes:
        n = size
        reps = max(3, int(reps_target_bytes / (n * ELEM_BYTES)))
        av, cv = a[:n], c[:n]
        np.copyto(cv, av)  # warm
        best = 1e300
        for _ in range(reps):
            t0 = perf_counter()
            np.copyto(cv, av)
            dt = perf_counter() - t0
            if dt < best:
                best = dt
        if best > 0:
            bw = OPERATIONS["Copy"]["bytes"] * n / best / GB
            results.append((n * ELEM_BYTES, bw))
        size *= 2

    if saved_affinity is not None:
        try:
            os.sched_setaffinity(0, saved_affinity)
        except OSError:
            pass
    return results


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def print_op_table(title, results, note=None):
    print(f"\n{title}")
    print("-" * len(title))
    if note:
        print(f"  {note}")
    print(f"  {'Operation':<8} {'GB/s':>9} {'min':>9} {'max':>9} {'cv%':>6}")
    for op in OP_ORDER:
        s = results.get(op)
        if not s:
            continue
        warn = "  <- high variance" if s["cv"] > 5.0 else ""
        print(f"  {op:<8} {s['median']:>9.2f} {s['min']:>9.2f} "
              f"{s['max']:>9.2f} {s['cv']:>6.1f}{warn}")


def print_sweep(results, caches):
    if not results:
        return
    print("\nWorking-set sweep (single core, Copy kernel)")
    print("-" * 43)
    print("  Bandwidth vs per-array size. Plateau at large sizes ~= per-core")
    print("  DRAM bandwidth; peaks at small sizes show cache bandwidth.")
    cache_sizes = sorted(sz for lvl, t, sz in caches if t != "Instruction")
    print(f"\n  {'array size':>11} {'GB/s':>9}  bar")
    peak = max(bw for _, bw in results) if results else 1.0
    for size, bw in results:
        bar = "#" * int(round(bw / peak * 40))
        marker = ""
        for sz in cache_sizes:
            if size > sz >= size // 2:
                marker = f"  <~ exceeds {human_bytes(sz)} cache"
        print(f"  {human_bytes(size):>11} {bw:>9.2f}  {bar}{marker}")
    # Plateau estimate: mean of the largest few points.
    tail = [bw for _, bw in results[-3:]]
    if tail:
        print(f"\n  Estimated per-core DRAM bandwidth (plateau): "
              f"{statistics.fmean(tail):.2f} GB/s")


def parse_stream_file(filename):
    """Parse STREAM.C output; returns ({op: MB_per_s}, ok)."""
    results = {"Copy": None, "Scale": None, "Add": None, "Triad": None}
    try:
        with open(filename) as f:
            content = f.readlines()
    except OSError as exc:
        print(f"Error reading STREAM file {filename}: {exc}")
        return results, False

    start = -1
    for i, line in enumerate(content):
        if "Function" in line and "Best Rate MB/s" in line:
            start = i + 1
            break
    if start == -1:
        print(f"Warning: could not find results table in {filename}")
        return results, False

    for line in content[start:start + 4]:
        parts = line.split()
        if len(parts) >= 2 and parts[0].rstrip(":") in results:
            try:
                results[parts[0].rstrip(":")] = float(parts[1])
            except ValueError:
                pass
    ok = any(v is not None for v in results.values())
    return results, ok


def print_stream_comparison(numpy_res, native_res, stream_mb):
    print("\nComparison vs STREAM.C (decimal MB/s)")
    print("-" * 37)

    def cell(v, ref):
        if v is None:
            return f"{'-':>13}"
        if ref:
            return f"{v:.0f} ({v / ref * 100:.0f}%)".rjust(13)
        return f"{v:.0f}".rjust(13)

    print(f"  {'Operation':<8} {'STREAM.C':>13} {'Native':>13} {'NumPy':>13}")
    for op in OP_ORDER:
        sc = stream_mb.get(op)
        nat = native_res.get(op)["median"] * 1000 if native_res.get(op) else None
        npv = numpy_res.get(op)["median"] * 1000 if numpy_res.get(op) else None
        print(f"  {op:<8} {(f'{sc:.0f}' if sc else '-'):>13}"
              f"{cell(nat, sc)}{cell(npv, sc)}")
    print("  (percentages are share of STREAM.C; 1 GB/s = 1000 MB/s, decimal)")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BAND: Bandwidth Assessment for Native DDR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--size", type=float, default=2.0,
                        help="Total memory in GB used per test")
    parser.add_argument("--iterations", type=int, default=7,
                        help="Timed iterations per operation")
    parser.add_argument("--threads", type=int, default=None,
                        help="Threads (default: all available CPUs, capped at 8)")
    parser.add_argument("--no-pin", action="store_true",
                        help="Disable pinning threads to CPUs")

    parser.add_argument("--no-numpy", action="store_true", help="Skip Tier 0 (NumPy)")
    parser.add_argument("--no-native", action="store_true", help="Skip Tier 1 (native)")
    parser.add_argument("--dram-counters", action="store_true",
                        help="Attempt Tier 2 (DRAM counters via perf; needs privileges)")
    parser.add_argument("--dram-driver", choices=("auto", "native", "numpy"),
                        default="auto",
                        help="Workload that drives the Tier 2 counting window "
                             "(auto: native if available, else numpy)")
    parser.add_argument("--no-sweep", action="store_true",
                        help="Skip the working-set sweep")
    parser.add_argument("--verbose", action="store_true",
                        help="Show compiler command and extra diagnostics")

    parser.add_argument("--stream-file", type=str, default=None,
                        help="STREAM.C output file to compare against")

    parser.add_argument("--peak-mts", type=float, default=None,
                        help="Memory transfer rate in MT/s (e.g. 6000 for DDR5-6000) "
                             "to report %% of theoretical peak")
    parser.add_argument("--channels", type=int, default=None,
                        help="Number of populated memory channels (for peak calc)")

    args = parser.parse_args()

    # Keep NumPy's own threading off so our threads/OpenMP control parallelism.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    avail = available_cpus()
    threads = args.threads if args.threads else min(len(avail), 8)
    pin = not args.no_pin

    # --size is the total footprint across all three arrays (a, b, c), so the
    # per-array element count is size / 3 / 8. Both tiers use the same value,
    # giving identical working sets for a fair comparison.
    per_array_elems = max(int(args.size * GB / 3 / ELEM_BYTES), 1 << 20)
    total_elems = max(per_array_elems, threads * NUMPY_TRIAD_CHUNK)
    per_array_native = per_array_elems

    info = get_system_info()
    print("BAND: Bandwidth Assessment for Native DDR")
    print("=" * 43)
    print(f"System:    {info['system']} {info['architecture']}")
    if "cpu_model" in info:
        print(f"Processor: {info['cpu_model']}")
    print(f"CPUs:      {info['cpu_count']} ({info['available_cpus']} available)")
    if "memory_gb" in info:
        print(f"Memory:    {info['memory_gb']:.1f} GiB")
    if info["caches"]:
        cache_str = ", ".join(
            f"L{lvl}{('i' if t == 'Instruction' else 'd' if t == 'Data' else '')}="
            f"{human_bytes(sz)}" for lvl, t, sz in info["caches"])
        print(f"Caches:    {cache_str}")
    print(f"Test size: {args.size:.2f} GB total | threads: {threads} | "
          f"pin: {'on' if pin else 'off'} | iters: {args.iterations}")

    theoretical_peak = None
    if args.peak_mts and args.channels:
        # GB/s = MT/s * 8 bytes/transfer * channels / 1000 (decimal)
        theoretical_peak = args.peak_mts * 8 * args.channels / 1000.0
        print(f"Theoretical peak: {theoretical_peak:.1f} GB/s "
              f"({args.peak_mts:.0f} MT/s x {args.channels} ch)")

    numpy_results = {}
    native_results = {}
    native_backend = None  # shared by Tier 1 and Tier 2

    # ---- Tier 0: NumPy ----
    if not args.no_numpy:
        print("\n[Tier 0] NumPy  (achievable from Python)")
        for op in OP_ORDER:
            print(f"  measuring {op}...", end="", flush=True)
            try:
                s = measure_numpy(op, total_elems, threads, args.iterations, pin=pin)
                numpy_results[op] = s
                print(f" {s['median']:.2f} GB/s")
            except MemoryError:
                print(" out of memory (reduce --size)")
                break
        print_op_table("Tier 0 (NumPy) results", numpy_results)

    # ---- Tier 1: Native ----
    if not args.no_native:
        print("\n[Tier 1] Native  (achievable from compiled C + OpenMP)")
        native_backend, err = NativeBackend.build(threads, verbose=args.verbose)
        if native_backend is None:
            print(f"  skipped: {err}")
        else:
            native_reps = max(args.iterations + 3, 10)
            for op in OP_ORDER:
                print(f"  measuring {op}...", end="", flush=True)
                try:
                    s = native_backend.measure(op, per_array_native, native_reps)
                    native_results[op] = s
                    print(f" {s['median']:.2f} GB/s")
                except MemoryError:
                    print(" out of memory (reduce --size)")
                    break
            print_op_table(
                "Tier 1 (Native) results", native_results,
                note="best-of-reps; -O3 -march=native, may use SIMD/streaming stores")

    # ---- Working-set sweep ----
    if not args.no_sweep:
        llc = largest_cache_bytes(info["caches"]) or 32 * 1024 * 1024
        max_array = min(max(llc * 8, 64 * 1024 * 1024), 512 * 1024 * 1024)
        sweep = working_set_sweep(max_array)
        print_sweep(sweep, info["caches"])

    # ---- Tier 2: DRAM counters ----
    if args.dram_counters:
        print("\n[Tier 2] DRAM counters  (measured traffic at memory controller)")
        ok, why = _perf_available()
        if not ok:
            print(f"  skipped: {why}")
        else:
            print("  probing uncore IMC events...", end="", flush=True)
            event_set = _probe_perf_events()
            if not event_set:
                print(" none accepted on this CPU; skipped")
            else:
                print(f" using {event_set[0]} ({', '.join(event_set[1])})")

                # Pick the workload that drives the counting window. The native
                # kernel sustains traffic closest to the hardware ceiling; NumPy
                # is the portable fallback.
                want = args.dram_driver
                if want in ("auto", "native") and native_backend is None:
                    nb, _ = NativeBackend.build(threads, verbose=False)
                    native_backend = nb
                use_native = (want != "numpy") and native_backend is not None
                if want == "native" and native_backend is None:
                    print("  (native driver requested but unavailable; using NumPy)")
                    use_native = False

                pool = None
                if use_native:
                    driver_fn = native_backend.make_driver("Triad", per_array_native)
                    driver_label = "native"
                else:
                    pool = NumpyPool(total_elems, threads, pin=pin)
                    driver_label = "NumPy"

                    def driver_fn(stop, _pool=pool):
                        while not stop.is_set():
                            _pool.run_once("Triad")
                try:
                    res, err = measure_dram_counters_live(driver_fn, 3.0, event_set)
                finally:
                    if pool is not None:
                        pool.close()
                if err:
                    print(f"  skipped: {err}")
                else:
                    print(f"  measured DRAM bandwidth: {res['bandwidth']:.2f} GB/s "
                          f"(saturating {driver_label} Triad workload)")
                    print("  note: real bus traffic incl. read-for-ownership, so it")
                    print("        can exceed the logical Tier 0/1 figures.")

    # ---- STREAM.C comparison ----
    if args.stream_file:
        stream_mb, ok = parse_stream_file(args.stream_file)
        if ok:
            print_stream_comparison(numpy_results, native_results, stream_mb)
        else:
            print(f"\nSTREAM.C comparison skipped (could not parse {args.stream_file})")

    # ---- Summary ----
    print("\n" + "=" * 43)
    print("Summary (Triad, the most representative kernel)")
    if numpy_results.get("Triad"):
        print(f"  Tier 0 NumPy : {numpy_results['Triad']['median']:.2f} GB/s")
    if native_results.get("Triad"):
        print(f"  Tier 1 Native: {native_results['Triad']['median']:.2f} GB/s")
    if theoretical_peak:
        best = max((r["Triad"]["median"] for r in (native_results, numpy_results)
                    if r.get("Triad")), default=0)
        if best:
            print(f"  Best Triad is {best/theoretical_peak*100:.0f}% of "
                  f"theoretical peak ({theoretical_peak:.1f} GB/s)")
    print("\nTier meaning: Tier 0 = what Python can reach; Tier 1 = what native")
    print("code can reach (~hardware achievable); Tier 2 = true DRAM traffic.")


if __name__ == "__main__":
    main()
