#!/usr/bin/env python3
"""
BAND: Bandwidth Assessment for Native DDR

A memory bandwidth measurement tool for Python that provides results
comparable to the STREAM benchmark. Includes optimized implementations
for various memory operations with a focus on performance.
"""

import numpy as np
import time
import argparse
import os
import multiprocessing
import threading
import psutil
import platform
from datetime import datetime
import gc
import sys

class BandwidthTest:
    """Base class for bandwidth tests"""

    def __init__(self, name, size_gb=1, threads=None, iterations=3, chunk_size_kb=None):
        self.name = name
        self.size_gb = size_gb
        self.iterations = iterations
        self.threads = threads if threads is not None else min(multiprocessing.cpu_count(), 4)

        # Calculate per-thread array sizes
        self.thread_size_gb = self.size_gb / self.threads
        self.elements_per_thread = int(self.thread_size_gb * 1024**3 / 8)  # 8 bytes per double

        # Set chunk size - allow override from parameter
        if chunk_size_kb:
            self.chunk_size = chunk_size_kb * 1024  # Convert KB to bytes
        else:
            # Default chunk size - can be overridden in subclasses
            self.chunk_size = 1024 * 1024  # 1MB chunks

        # Results storage
        self.results = []

    def _setup_data(self):
        """Create data arrays for each thread"""
        # Use a list to track arrays for each thread
        thread_data = []

        # Initialize arrays with proper values for the test
        for _ in range(self.threads):
            # Create arrays for each test
            a = np.ones(self.elements_per_thread, dtype=np.float64)
            b = np.ones(self.elements_per_thread, dtype=np.float64) * 2.0
            c = np.zeros(self.elements_per_thread, dtype=np.float64)
            thread_data.append((a, b, c))

        return thread_data

    def run(self):
        """Run the benchmark with multiple threads"""
        print(f"Running {self.name} test with {self.threads} threads, " +
              f"{self.size_gb:.1f} GB total memory...")

        # Force garbage collection before starting
        gc.collect()

        # Prepare arrays
        thread_data = self._setup_data()

        # Warmup to ensure memory is allocated
        self._warmup(thread_data)

        # Run multiple iterations
        times = []
        for i in range(self.iterations):
            print(f"  Iteration {i+1}/{self.iterations}...", end="", flush=True)

            # Start threads
            threads = []

            # Let system settle for consistency
            time.sleep(0.1)
            gc.collect()

            start_time = time.time()

            for t in range(self.threads):
                thread = threading.Thread(
                    target=self._run_thread,
                    args=(thread_data[t], t)
                )
                thread.daemon = True
                threads.append(thread)
                thread.start()

            # Wait for threads to complete
            for thread in threads:
                thread.join()

            elapsed = time.time() - start_time

            # Calculate bandwidth
            total_bytes_processed = self._get_bytes_processed() * self.threads
            bandwidth_gb_sec = total_bytes_processed / elapsed / (1024**3)

            times.append(bandwidth_gb_sec)
            print(f" {bandwidth_gb_sec:.2f} GB/s")

        # Calculate statistics - exclude first iteration if we have multiple
        if len(times) > 1:
            times = times[1:]

        if times:
            self.results = {
                "min": min(times),
                "max": max(times),
                "mean": sum(times) / len(times),
                "raw": times
            }

            print(f"  {self.name} result: {self.results['mean']:.2f} GB/s " +
                  f"(min: {self.results['min']:.2f}, max: {self.results['max']:.2f})")

        return self.results

    def _warmup(self, thread_data):
        """Warm up memory access to ensure it's mapped"""
        for arrays in thread_data:
            a, b, c = arrays

            # Touch every 1MB to ensure pages are mapped
            stride = 131072  # 1MB worth of doubles
            for i in range(0, len(a), stride):
                a[i] = 1.0
                b[i] = 2.0
                c[i] = 0.0

    def _run_thread(self, arrays, thread_id):
        """Override in subclasses"""
        pass

    def _get_bytes_processed(self):
        """Return bytes processed in one thread"""
        return 0


class StreamCopy(BandwidthTest):
    """STREAM Copy test: c = a"""

    def __init__(self, *args, **kwargs):
        super().__init__("Py-STREAM Copy", *args, **kwargs)

    def _run_thread(self, arrays, thread_id):
        a, _, c = arrays

        # Use NumPy's optimized copy
        np.copyto(c, a)

    def _get_bytes_processed(self):
        # Read from a, write to c = 2 operations
        return 2 * 8 * self.elements_per_thread  # 2 arrays × 8 bytes × elements


class StreamScale(BandwidthTest):
    """STREAM Scale test: b = scalar × c"""

    def __init__(self, *args, **kwargs):
        super().__init__("Py-STREAM Scale", *args, **kwargs)

    def _run_thread(self, arrays, thread_id):
        _, b, c = arrays
        scalar = 3.0

        # Full vectorized operation
        np.multiply(c, scalar, out=b)

    def _get_bytes_processed(self):
        # Read from c, write to b = 2 operations
        return 2 * 8 * self.elements_per_thread


class StreamAdd(BandwidthTest):
    """STREAM Add test: c = a + b"""

    def __init__(self, *args, **kwargs):
        super().__init__("Py-STREAM Add", *args, **kwargs)
        # Default optimal chunk size - can be overridden by command line
        if not kwargs.get('chunk_size_kb'):
            self.chunk_size = 4 * 1024 * 1024  # 4MB chunks

    def _run_thread(self, arrays, thread_id):
        a, b, c = arrays

        # Process in optimally sized chunks
        for i in range(0, len(c), self.chunk_size):
            end = min(i + self.chunk_size, len(c))
            np.add(a[i:end], b[i:end], out=c[i:end])

    def _get_bytes_processed(self):
        # Read from a and b, write to c = 3 operations
        return 3 * 8 * self.elements_per_thread


class StreamTriad(BandwidthTest):
    """STREAM Triad test: a = b + scalar × c"""

    def __init__(self, *args, **kwargs):
        super().__init__("Py-STREAM Triad", *args, **kwargs)
        # Default optimal chunk size - can be overridden by command line
        if not kwargs.get('chunk_size_kb'):
            self.chunk_size = 4 * 1024 * 1024  # 4MB chunks

    def _run_thread(self, arrays, thread_id):
        a, b, c = arrays
        scalar = 3.0

        # Process in chunks for better cache behavior
        for i in range(0, len(a), self.chunk_size):
            end = min(i + self.chunk_size, len(a))

            # Two separate operations to avoid temporary array creation
            chunk_c = c[i:end] * scalar
            a[i:end] = b[i:end] + chunk_c

    def _get_bytes_processed(self):
        # Read from b and c, write to a, with scalar multiplication = 3 operations
        return 3 * 8 * self.elements_per_thread


class ChunkedTriad(BandwidthTest):
    """Chunked Triad implementation with explicit temporaries"""

    def __init__(self, *args, **kwargs):
        super().__init__("Py-Chunked Triad", *args, **kwargs)
        # Default optimal chunk size - can be overridden by command line
        if not kwargs.get('chunk_size_kb'):
            self.chunk_size = 512 * 1024  # 512KB chunks

    def _run_thread(self, arrays, thread_id):
        a, b, c = arrays
        scalar = 3.0

        # Create a reusable temporary array
        temp = np.empty(self.chunk_size, dtype=np.float64)

        # Process in small chunks with reused temporary
        for i in range(0, len(a), self.chunk_size):
            end = min(i + self.chunk_size, len(a))
            chunk_size = end - i

            # First compute scalar * c into temp
            np.multiply(c[i:end], scalar, out=temp[:chunk_size])

            # Then add b and store in a
            np.add(b[i:end], temp[:chunk_size], out=a[i:end])

    def _get_bytes_processed(self):
        # Read from b and c, write to a, with scalar multiplication = 3 operations
        return 3 * 8 * self.elements_per_thread


class CombinedTriad(BandwidthTest):
    """Combined operation Triad: a = b + scalar * c"""

    def __init__(self, *args, **kwargs):
        super().__init__("Py-Combined Triad", *args, **kwargs)
        # Default optimal chunk size - can be overridden by command line
        if not kwargs.get('chunk_size_kb'):
            self.chunk_size = 8 * 1024 * 1024  # 8MB chunks

    def _run_thread(self, arrays, thread_id):
        a, b, c = arrays
        scalar = 3.0

        # Process in chunks for better performance
        for i in range(0, len(a), self.chunk_size):
            end = min(i + self.chunk_size, len(a))

            # Use numpy's ability to combine operations
            # By using a single expression, NumPy can optimize better
            a[i:end] = b[i:end] + scalar * c[i:end]

    def _get_bytes_processed(self):
        # Read from b and c, write to a, with scalar multiplication = 3 operations
        return 3 * 8 * self.elements_per_thread


class MemcpyTest(BandwidthTest):
    """Memory copy test similar to MBW"""

    def __init__(self, *args, **kwargs):
        super().__init__("Py-MEMCPY", *args, **kwargs)

    def _run_thread(self, arrays, thread_id):
        a, b, _ = arrays

        # Use numpy's built-in copy for maximum performance
        np.copyto(b, a)

    def _get_bytes_processed(self):
        # Read from a, write to b = 2 operations
        return 2 * 8 * self.elements_per_thread


def get_system_info():
    """Collect system information"""
    info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "system": platform.system(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": multiprocessing.cpu_count(),
        "memory_gb": psutil.virtual_memory().total / (1024**3)
    }

    # Try to get CPU model from /proc/cpuinfo
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if 'model name' in line:
                    info["cpu_model"] = line.split(':')[1].strip()
                    break
    except:
        pass

    # Try to get memory information
    try:
        if platform.system() == "Linux":
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        mem_kb = int(line.split()[1])
                        info["memory_total_kb"] = mem_kb
                        break
    except:
        pass

    return info


def parse_stream_file(filename):
    """
    Parse a STREAM benchmark output file to extract benchmark results.

    Args:
        filename (str): Path to the file containing STREAM output

    Returns:
        dict: Dictionary with benchmark results for Copy, Scale, Add, and Triad operations
        bool: Success status of parsing operation
    """
    results = {
        "Copy": None,
        "Scale": None,
        "Add": None,
        "Triad": None
    }

    try:
        with open(filename, 'r') as f:
            content = f.readlines()

        # Find the results table
        table_start = -1
        for i, line in enumerate(content):
            if "Function" in line and "Best Rate MB/s" in line:
                table_start = i + 1
                break

        if table_start == -1:
            print(f"Warning: Could not find results table in {filename}")
            return results, False

        # Parse the next 4 lines for Copy, Scale, Add, Triad results
        for i in range(4):
            if table_start + i < len(content):
                line = content[table_start + i]
                parts = line.split()

                # Expected format: "Copy: 25698.6 0.062431 0.062260 0.062579"
                if len(parts) >= 2 and parts[0].rstrip(':') in results:
                    op_name = parts[0].rstrip(':')
                    try:
                        results[op_name] = float(parts[1])
                    except ValueError:
                        print(f"Warning: Could not parse value for {op_name}: {parts[1]}")

        # Verify we found all values
        missing = [op for op, val in results.items() if val is None]
        if missing:
            print(f"Warning: Could not find results for: {', '.join(missing)}")
            if len(missing) == len(results):  # If all values are missing
                return results, False

        return results, True

    except Exception as e:
        print(f"Error parsing STREAM file {filename}: {str(e)}")
        return results, False

def calculate_effective_bandwidth(results):
    """
    Calculate effective bandwidth scores for different application types
    """
    # General application weights based on instruction mix research
    general_weights = {
        "Py-STREAM Copy": 0.40,  # 40% weight for copy operations
        "Py-STREAM Scale": 0.25,  # 25% weight for scale operations
        "Py-STREAM Add": 0.15,   # 15% weight for add operations
        "Py-STREAM Triad": 0.20  # 20% weight for triad operations
    }

    # LLM-specific weights based on memory access patterns in LLMs
    llm_weights = {
        "Py-STREAM Copy": 0.70,  # 70% weight for copy operations (read-heavy)
        "Py-STREAM Scale": 0.20,  # 20% weight for scale operations
        "Py-STREAM Add": 0.05,   # 5% weight for add operations
        "Py-STREAM Triad": 0.05  # 5% weight for triad operations
    }

    # Get benchmark results (falling back to 0 if test not run)
    copy_bw = results.get("Py-STREAM Copy", {}).get("mean", 0)
    scale_bw = results.get("Py-STREAM Scale", {}).get("mean", 0)
    add_bw = results.get("Py-STREAM Add", {}).get("mean", 0)
    triad_bw = results.get("Py-STREAM Triad", {}).get("mean", 0)

    # Calculate general application effective bandwidth
    general_bw = (
        general_weights["Py-STREAM Copy"] * copy_bw +
        general_weights["Py-STREAM Scale"] * scale_bw +
        general_weights["Py-STREAM Add"] * add_bw +
        general_weights["Py-STREAM Triad"] * triad_bw
    )

    # Calculate LLM-specific effective bandwidth
    llm_bw = (
        llm_weights["Py-STREAM Copy"] * copy_bw +
        llm_weights["Py-STREAM Scale"] * scale_bw +
        llm_weights["Py-STREAM Add"] * add_bw +
        llm_weights["Py-STREAM Triad"] * triad_bw
    )

    # Calculate with doubled Triad (to match STREAM.C implementation)
    doubled_triad = triad_bw * 2  # Assume Python Triad is ~50% of C Triad

    general_bw_adjusted = (
        general_weights["Py-STREAM Copy"] * copy_bw +
        general_weights["Py-STREAM Scale"] * scale_bw +
        general_weights["Py-STREAM Add"] * add_bw +
        general_weights["Py-STREAM Triad"] * doubled_triad
    )

    llm_bw_adjusted = (
        llm_weights["Py-STREAM Copy"] * copy_bw +
        llm_weights["Py-STREAM Scale"] * scale_bw +
        llm_weights["Py-STREAM Add"] * add_bw +
        llm_weights["Py-STREAM Triad"] * doubled_triad
    )

    return {
        "general": general_bw,
        "llm": llm_bw,
        "general_adjusted": general_bw_adjusted,
        "llm_adjusted": llm_bw_adjusted,
        "weights": {
            "general": general_weights,
            "llm": llm_weights
        }
    }
def calculate_effective_bandwidth(results):
    """
    Calculate effective bandwidth scores for different application types
    """
    # General application weights based on instruction mix research
    general_weights = {
        "Py-STREAM Copy": 0.40,  # 40% weight for copy operations
        "Py-STREAM Scale": 0.25,  # 25% weight for scale operations
        "Py-STREAM Add": 0.15,   # 15% weight for add operations
        "Py-STREAM Triad": 0.20  # 20% weight for triad operations
    }

    # LLM-specific weights based on memory access patterns in LLMs
    llm_weights = {
        "Py-STREAM Copy": 0.70,  # 70% weight for copy operations (read-heavy)
        "Py-STREAM Scale": 0.20,  # 20% weight for scale operations
        "Py-STREAM Add": 0.05,   # 5% weight for add operations
        "Py-STREAM Triad": 0.05  # 5% weight for triad operations
    }

    # Get benchmark results (falling back to 0 if test not run)
    copy_bw = results.get("Py-STREAM Copy", {}).get("mean", 0)
    scale_bw = results.get("Py-STREAM Scale", {}).get("mean", 0)
    add_bw = results.get("Py-STREAM Add", {}).get("mean", 0)
    triad_bw = results.get("Py-STREAM Triad", {}).get("mean", 0)

    # Calculate general application effective bandwidth
    general_bw = (
        general_weights["Py-STREAM Copy"] * copy_bw +
        general_weights["Py-STREAM Scale"] * scale_bw +
        general_weights["Py-STREAM Add"] * add_bw +
        general_weights["Py-STREAM Triad"] * triad_bw
    )

    # Calculate LLM-specific effective bandwidth
    llm_bw = (
        llm_weights["Py-STREAM Copy"] * copy_bw +
        llm_weights["Py-STREAM Scale"] * scale_bw +
        llm_weights["Py-STREAM Add"] * add_bw +
        llm_weights["Py-STREAM Triad"] * triad_bw
    )

    # Calculate with doubled Triad (to match STREAM.C implementation)
    doubled_triad = triad_bw * 2  # Assume Python Triad is ~50% of C Triad

    general_bw_adjusted = (
        general_weights["Py-STREAM Copy"] * copy_bw +
        general_weights["Py-STREAM Scale"] * scale_bw +
        general_weights["Py-STREAM Add"] * add_bw +
        general_weights["Py-STREAM Triad"] * doubled_triad
    )

    llm_bw_adjusted = (
        llm_weights["Py-STREAM Copy"] * copy_bw +
        llm_weights["Py-STREAM Scale"] * scale_bw +
        llm_weights["Py-STREAM Add"] * add_bw +
        llm_weights["Py-STREAM Triad"] * doubled_triad
    )

    return {
        "general": general_bw,
        "llm": llm_bw,
        "general_adjusted": general_bw_adjusted,
        "llm_adjusted": llm_bw_adjusted,
        "weights": {
            "general": general_weights,
            "llm": llm_weights
        }
    }


def main():
    parser = argparse.ArgumentParser(description="BAND: Bandwidth Assessment for Native DDR")
    parser.add_argument("--size", type=float, default=4.0,
                        help="Size in GB for each test (default: 4 GB)")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Number of iterations per test (default: 3)")
    parser.add_argument("--threads", type=int,
                        default=min(multiprocessing.cpu_count(), 4),
                        help=f"Number of threads (default: min(CPU count, 4))")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="Chunk size in KB for operations (default: varies by test)")
    parser.add_argument("--triad-only", action="store_true",
                        help="Run only the triad tests for optimization experiments")
    parser.add_argument("--best", action="store_true",
                        help="Run only the best implementation for each operation")
    parser.add_argument("--compare", action="store_true",
                        help="Compare to C STREAM benchmark results")
    parser.add_argument("--c-stream-triad", type=float, default=19.98,
                        help="C STREAM Triad result for comparison (default: 19.98 GB/s)")
    parser.add_argument("--stream-file", type=str, default=None,
                        help="Path to STREAM benchmark output file for comparison")
    parser.add_argument("--enable-chunking", action="store_true",
                        help="Enable chunked implementations that may benefit from CPU cache")
    args = parser.parse_args()

    # Try to set process priority higher for more consistent results
    try:
        if platform.system() == "Windows":
            import psutil
            p = psutil.Process(os.getpid())
            p.nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            os.nice(-10)  # Lower value means higher priority on Unix
    except:
        print("Warning: Could not set process priority")

    # Disable numpy threading to avoid oversubscription
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # Get STREAM results from file if provided
    stream_results = {"Copy": None, "Scale": None, "Add": None, "Triad": None}
    stream_file_valid = False

    if args.stream_file:
        print(f"\nReading STREAM benchmark results from {args.stream_file}")
        stream_results, stream_file_valid = parse_stream_file(args.stream_file)

        if stream_file_valid:
            print("STREAM-C results detected:")
            for op, val in stream_results.items():
                if val is not None:
                    print(f"  {op}: {val:.2f} MB/s")
        else:
            print("ERROR: Failed to parse STREAM benchmark file. Continuing without comparison.")
            print("If you want to compare with STREAM-C, please check your file format or use --c-stream-triad option.")

    # Use file-provided Triad value if available and valid, otherwise use command-line argument
    c_stream_triad = stream_results["Triad"] if (stream_file_valid and stream_results["Triad"] is not None) else args.c_stream_triad

    # Print system information
    info = get_system_info()
    print("\nBAND: Bandwidth Assessment for Native DDR")
    print("-" * 40)
    print(f"System: {info['system']} {info['architecture']}")
    if 'cpu_model' in info:
        print(f"Processor: {info['cpu_model']}")
    print(f"CPU Cores: {info['cpu_count']}")
    print(f"Memory: {info['memory_gb']:.1f} GB")
    print(f"Test Size: {args.size:.1f} GB per test")
    print(f"Threads: {args.threads}")
    print(f"Iterations: {args.iterations}")
    if args.chunk_size:
        print(f"Chunk Size: {args.chunk_size} KB")
    if args.enable_chunking:
        print("Cache optimization: Enabled (using chunked implementations)")
    print("")

    # Common kwargs for all tests
    test_kwargs = {
        'size_gb': args.size,
        'threads': args.threads,
        'iterations': args.iterations,
        'chunk_size_kb': args.chunk_size
    }

    # Adjust test selection based on arguments
    if args.triad_only:
        if args.best:
            if args.enable_chunking:
                tests = [
                    ChunkedTriad(**test_kwargs)  # Best with chunking enabled
                ]
            else:
                tests = [
                    StreamTriad(**test_kwargs)  # Standard STREAM implementation
                ]
        else:
            tests = [
                StreamTriad(**test_kwargs),  # Always include standard implementation
            ]
            if args.enable_chunking:
                # Only add chunked implementations if explicitly enabled
                tests.extend([
                    ChunkedTriad(**test_kwargs),
                    CombinedTriad(**test_kwargs)
                ])
    elif args.best:
        # Use only the best implementation for each operation
        tests = [
            StreamCopy(**test_kwargs),      # Best Copy
            StreamScale(**test_kwargs),     # Best Scale
            StreamAdd(**test_kwargs),       # Best Add
        ]
        if args.enable_chunking:
            tests.append(ChunkedTriad(**test_kwargs))  # Best Triad with chunking
        else:
            tests.append(StreamTriad(**test_kwargs))   # Standard STREAM Triad
    else:
        # Default test selection - standard STREAM implementations
        tests = [
            StreamCopy(**test_kwargs),
            StreamScale(**test_kwargs),
            StreamAdd(**test_kwargs),
            StreamTriad(**test_kwargs),
            MemcpyTest(**test_kwargs)
        ]

        # Add chunked implementations only if explicitly enabled
        if args.enable_chunking:
            tests.extend([
                ChunkedTriad(**test_kwargs),
                CombinedTriad(**test_kwargs)
            ])

    # Run all tests
    results = {}
    for test in tests:
        result = test.run()
        results[test.name] = result
        print("")  # Add spacing between tests

    # Print summary
    print("\nResults Summary")
    print("-" * 14)
    for test_name, result in results.items():
        if result:
            print(f"{test_name}: {result['mean']:.2f} GB/s")

    # Compare triad implementations if available
    if "Py-STREAM Triad" in results and args.enable_chunking:
        base_triad = results["Py-STREAM Triad"]["mean"]
        print("\nTriad Implementation Comparison (vs Py-STREAM Triad):")
        for test_name, result in results.items():
            if test_name != "Py-STREAM Triad" and "Triad" in test_name:
                triad_result = result["mean"]
                if base_triad > 0:
                    improvement = (triad_result / base_triad - 1) * 100
                    print(f"  {test_name}: {triad_result:.2f} GB/s ({improvement:+.1f}%)")

    # Compare to C STREAM results if requested
    if args.compare or args.stream_file:
        if not args.compare and not stream_file_valid:
            # Skip comparison if neither option is usable
            print("\nC STREAM comparison skipped due to invalid file.")
        else:
            print("\nPython vs C Comparison:")

            # Compare all operations if we have valid file data
            if stream_file_valid:
                # Map test names to STREAM-C operation names
                operation_map = {
                    "Py-STREAM Copy": "Copy",
                    "Py-STREAM Scale": "Scale",
                    "Py-STREAM Add": "Add",
                    "Py-STREAM Triad": "Triad",
                    "Py-Chunked Triad": "Triad",
                    "Py-Combined Triad": "Triad",
                    "Py-MEMCPY": "Copy"
                }

                for test_name, result in results.items():
                    if test_name in operation_map and stream_results[operation_map[test_name]] is not None:
                        stream_value = stream_results[operation_map[test_name]]  # Already in MB/s
                        python_value = result["mean"] * 1024.0  # Convert GB/s to MB/s
                        ratio = python_value / stream_value * 100
                        print(f"  {test_name}: {python_value:.2f} MB/s ({ratio:.1f}% of STREAM.C {operation_map[test_name]} @ {stream_value:.2f} MB/s)")
            elif c_stream_triad > 0:
                # Only compare Triad with command-line value
                best_triad = 0
                best_name = ""
                for test_name, result in results.items():
                    if "Triad" in test_name and result["mean"] > best_triad:
                        best_triad = result["mean"]
                        best_name = test_name

                if best_triad > 0:
                    # Convert GB/s to MB/s for display
                    best_triad_mb = best_triad * 1024.0
                    ratio = best_triad_mb / c_stream_triad * 100
                    print(f"  Best Python Triad ({best_name}) achieves {ratio:.1f}% of STREAM.C Triad performance")
                    print(f"  {best_name}: {best_triad_mb:.2f} MB/s vs STREAM.C Triad: {c_stream_triad:.2f} MB/s")

    # Calculate effective bandwidth metrics
    if "Py-STREAM Triad" in results:
        bandwidth_scores = calculate_effective_bandwidth(results)

        print("\nBAND Effective Bandwidth Metrics:")
        print("-" * 35)
        print("Py-STREAM results:")
        print(f"  - General application bandwidth score: {bandwidth_scores['general']:.2f} GB/s")
        print(f"  - LLM bandwidth score:                {bandwidth_scores['llm']:.2f} GB/s")

        print("\nPy-STREAM with doubled Triad (to match STREAM.C):")
        print(f"  - General application bandwidth score: {bandwidth_scores['general_adjusted']:.2f} GB/s")
        print(f"  - LLM bandwidth score:                {bandwidth_scores['llm_adjusted']:.2f} GB/s")

        # Explanation of calculations
        print("\nCalculation Explanation:")
        print("  General score = (0.40 × Copy) + (0.25 × Scale) + (0.15 × Add) + (0.20 × Triad)")
        print("  LLM score     = (0.70 × Copy) + (0.20 × Scale) + (0.05 × Add) + (0.05 × Triad)")
        print("  * Adjusted scores use doubled Triad values to approximate STREAM.C performance")
        print("  * Weightings based on instruction mix analysis of typical applications")




if __name__ == "__main__":
    main()
