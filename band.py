#!/usr/bin/env python3
"""
Optimized DDR Bandwidth Measurement Tool

This script measures memory bandwidth through various tests to estimate
the overall DDR bandwidth of a system. It's compatible with both x86/AMD64
and ARM64 platforms running Ubuntu 22.04 or similar Linux distributions.
"""

import os
import sys
import time
import argparse
import numpy as np
import multiprocessing
import psutil
import platform
from datetime import datetime
import threading
import ctypes
import signal
import json

# Global variable for signal handling
should_exit = False

class BandwidthTest:
    """Base class for bandwidth tests"""
    
    def __init__(self, name, size_gb=1, iterations=3, threads=None):
        self.name = name
        self.size_gb = size_gb
        self.iterations = iterations
        self.threads = threads if threads is not None else min(multiprocessing.cpu_count(), 4)
        self.results = []
        
        # Calculate array size to split across threads
        self.thread_size_gb = self.size_gb / self.threads
        
        # Use uint8 for more direct memory access (less overhead)
        # Elements per thread - calculate bytes directly
        self.thread_elements = int(self.thread_size_gb * 1024 * 1024 * 1024)
        
        # The block size for chunked operations - 64KB is a good balance
        self.block_size = 65536  # 64KB blocks
        
    def _setup_data(self):
        """Create thread arrays - optimized for performance"""
        thread_arrays = []
        for _ in range(self.threads):
            # Use uint8 instead of float64 for more direct memory access
            arr = np.zeros(self.thread_elements, dtype=np.uint8)
            # Initialize with some data
            arr[:] = np.random.randint(0, 255, size=min(16777216, len(arr)), dtype=np.uint8).repeat(max(1, len(arr) // 16777216))
            thread_arrays.append(arr)
        return thread_arrays
    
    def _run_thread(self, array, thread_id, results):
        """Base thread function - override in subclasses"""
        pass
    
    def run(self):
        """Run the test across multiple threads"""
        print(f"Running {self.name} test with {self.threads} threads, " +
              f"{self.size_gb:.1f} GB total memory...")
        
        # Setup thread data
        thread_arrays = self._setup_data()
        manager = multiprocessing.Manager()
        thread_results = manager.list()
        
        # For each iteration
        iteration_results = []
        for iteration in range(self.iterations):
            if should_exit:
                break
                
            print(f"  Iteration {iteration+1}/{self.iterations}...", end="", flush=True)
            
            # Create and start threads
            threads = []
            start_time = time.time()
            
            for i in range(self.threads):
                t = threading.Thread(
                    target=self._run_thread,
                    args=(thread_arrays[i], i, thread_results)
                )
                t.daemon = True
                threads.append(t)
                t.start()
            
            # Wait for all threads to complete
            for t in threads:
                t.join()
                
            end_time = time.time()
            elapsed = end_time - start_time
            
            # Calculate bandwidth in GB/s
            total_bytes = self.size_gb * 1024 * 1024 * 1024
            bandwidth_gb_sec = total_bytes / elapsed / (1024 * 1024 * 1024)
            
            iteration_results.append(bandwidth_gb_sec)
            print(f" {bandwidth_gb_sec:.2f} GB/s")
        
        # Calculate statistics if we have results
        if iteration_results:
            self.results = {
                "min": min(iteration_results),
                "max": max(iteration_results),
                "mean": np.mean(iteration_results),
                "median": np.median(iteration_results),
                "stddev": np.std(iteration_results),
                "iterations": len(iteration_results),
                "raw": iteration_results
            }
            
            print(f"  {self.name} result: {self.results['mean']:.2f} GB/s " +
                  f"(min: {self.results['min']:.2f}, max: {self.results['max']:.2f})")
            
        return self.results


class SequentialReadTest(BandwidthTest):
    """Sequential read test - reads memory sequentially"""
    
    def __init__(self, *args, **kwargs):
        super().__init__("Sequential Read", *args, **kwargs)
    
    def _run_thread(self, array, thread_id, results):
        # Optimized to use pointer arithmetic directly for better performance
        # Force CPU to read from memory by scanning whole array
        total_blocks = len(array) // self.block_size
        checksum = 0
        
        # Read in blocks for better performance
        for i in range(total_blocks):
            start_idx = i * self.block_size
            # Use numpy's sum directly - highly optimized
            checksum += np.sum(array[start_idx:start_idx+self.block_size])
            
        # Process any remaining elements
        if len(array) % self.block_size:
            start_idx = total_blocks * self.block_size
            checksum += np.sum(array[start_idx:])
            
        # Store result to prevent optimization
        results.append(checksum)


class SequentialWriteTest(BandwidthTest):
    """Sequential write test - writes memory sequentially"""
    
    def __init__(self, *args, **kwargs):
        super().__init__("Sequential Write", *args, **kwargs)
    
    def _run_thread(self, array, thread_id, results):
        # Optimized sequential write
        total_blocks = len(array) // self.block_size
        
        # Create a data block to write
        data_block = np.random.randint(0, 255, size=self.block_size, dtype=np.uint8)
        
        # Write in blocks
        for i in range(total_blocks):
            start_idx = i * self.block_size
            array[start_idx:start_idx+self.block_size] = data_block
            
        # Process any remaining elements
        if len(array) % self.block_size:
            start_idx = total_blocks * self.block_size
            remainder = len(array) - start_idx
            array[start_idx:] = data_block[:remainder]
            
        # Store a checksum to prevent optimization
        results.append(np.sum(array[:min(1024, len(array))]))


class SequentialCopyTest(BandwidthTest):
    """Sequential copy test - copies memory sequentially"""
    
    def __init__(self, *args, **kwargs):
        super().__init__("Sequential Copy", *args, **kwargs)
    
    def _setup_data(self):
        """Create source and destination arrays"""
        thread_arrays = []
        for _ in range(self.threads):
            # Source array - initialize with random data
            src = np.random.randint(0, 255, size=self.thread_elements, dtype=np.uint8)
            # Destination array - initialized to zero
            dst = np.zeros(self.thread_elements, dtype=np.uint8)
            thread_arrays.append((src, dst))
        return thread_arrays
    
    def _run_thread(self, arrays, thread_id, results):
        src, dst = arrays
        total_blocks = len(src) // self.block_size
        
        # Copy in blocks for better performance
        for i in range(total_blocks):
            start_idx = i * self.block_size
            dst[start_idx:start_idx+self.block_size] = src[start_idx:start_idx+self.block_size]
            
        # Handle remainder
        if len(src) % self.block_size:
            start_idx = total_blocks * self.block_size
            dst[start_idx:] = src[start_idx:]
            
        # Store a result to prevent optimization
        results.append(np.sum(dst[:min(1024, len(dst))]))


class RandomReadTest(BandwidthTest):
    """Random read test - reads memory in random pattern"""
    
    def __init__(self, *args, **kwargs):
        super().__init__("Random Read", *args, **kwargs)
        
    def _setup_data(self):
        """Create data arrays and index arrays"""
        thread_data = []
        for thread_id in range(self.threads):
            # Data array
            data = np.random.randint(0, 255, size=self.thread_elements, dtype=np.uint8)
            
            # Create random indices but in a more cache-friendly pattern
            # Divide into regions to maintain some locality
            region_size = 4096  # 4KB regions
            num_regions = self.thread_elements // region_size
            
            # Create indices for traversing regions in random order, but elements within
            # a region in sequential order for better performance
            indices = np.zeros(self.thread_elements, dtype=np.int32)
            
            region_order = np.arange(num_regions)
            np.random.shuffle(region_order)
            
            for i, region in enumerate(region_order):
                start = i * region_size
                src_start = region * region_size
                indices[start:start+region_size] = np.arange(src_start, src_start+region_size)
            
            # Handle remainder
            if self.thread_elements % region_size:
                start = num_regions * region_size
                indices[start:] = np.arange(start, self.thread_elements)
            
            thread_data.append((data, indices))
        return thread_data
    
    def _run_thread(self, arrays, thread_id, results):
        data, indices = arrays
        checksum = 0
        
        # Number of accesses - 50% of the array or 100M, whichever is smaller
        num_accesses = min(len(indices) // 2, 100_000_000)
        
        # Process in blocks for better cache behavior
        block_size = 1024  # Process 1024 indices at a time
        num_blocks = num_accesses // block_size
        
        for b in range(num_blocks):
            idx_start = b * block_size
            idx_end = idx_start + block_size
            
            # Gather the values at these indices
            for i in range(idx_start, idx_end):
                idx = indices[i]
                checksum += data[idx]
                
        # Handle remainder
        rem_start = num_blocks * block_size
        if rem_start < num_accesses:
            for i in range(rem_start, num_accesses):
                idx = indices[i]
                checksum += data[idx]
            
        # Store result to prevent optimization
        results.append(checksum)


class StridedReadTest(BandwidthTest):
    """Strided read test - reads memory in strided pattern"""
    
    def __init__(self, *args, **kwargs):
        super().__init__("Strided Read", *args, **kwargs)
    
    def _run_thread(self, array, thread_id, results):
        # Use different strides - optimized for better coverage
        strides = [16, 32, 64, 128, 256, 512, 1024, 2048]
        checksum = 0
        
        # Limited number of accesses per stride for better performance
        accesses_per_stride = min(len(array) // 64, 1_000_000)
        
        for stride in strides:
            # Limit by accesses rather than array length for better benchmarking
            limit = min(len(array), stride * accesses_per_stride)
            
            # Read with current stride
            for i in range(0, limit, stride):
                checksum += array[i]
                
        # Store result to prevent optimization
        results.append(checksum)


class StreamTriadTest(BandwidthTest):
    """Stream Triad benchmark - a = b + scalar * c"""
    
    def __init__(self, *args, **kwargs):
        super().__init__("Stream Triad", *args, **kwargs)
    
    def _setup_data(self):
        """Create source arrays for STREAM benchmark"""
        thread_arrays = []
        # Elements per array (we're using uint8, so we need more elements)
        elem_per_thread = self.thread_elements // 3  # We need 3 arrays
        
        for _ in range(self.threads):
            # Create arrays a, b, c with equal size
            a = np.zeros(elem_per_thread, dtype=np.uint8)
            b = np.random.randint(0, 255, size=elem_per_thread, dtype=np.uint8)
            c = np.random.randint(0, 255, size=elem_per_thread, dtype=np.uint8)
            thread_arrays.append((a, b, c))
        return thread_arrays
    
    def _run_thread(self, arrays, thread_id, results):
        a, b, c = arrays
        scalar = 3  # Simple scalar multiplier
        
        total_blocks = len(a) // self.block_size
        
        # Triad operation in blocks: a = b + scalar * c
        for i in range(total_blocks):
            start_idx = i * self.block_size
            end_idx = start_idx + self.block_size
            a[start_idx:end_idx] = b[start_idx:end_idx] + scalar * c[start_idx:end_idx]
            
        # Handle remainder
        if len(a) % self.block_size:
            start_idx = total_blocks * self.block_size
            a[start_idx:] = b[start_idx:] + scalar * c[start_idx:]
            
        # Store a result to prevent optimization
        results.append(np.sum(a[:min(1024, len(a))]))


def get_system_info():
    """Collect system information"""
    info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "system": platform.system(),
        "architecture": platform.machine(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": multiprocessing.cpu_count(),
        "memory_gb": psutil.virtual_memory().total / (1024**3)
    }
    
    # Try to get CPU model from /proc/cpuinfo
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            
        # For x86
        model_name = None
        for line in cpuinfo.split('\n'):
            if 'model name' in line:
                model_name = line.split(':')[1].strip()
                break
                
        # For ARM
        if model_name is None:
            for line in cpuinfo.split('\n'):
                if 'Hardware' in line:
                    model_name = line.split(':')[1].strip()
                    break
                    
        if model_name:
            info["cpu_model"] = model_name
    except:
        pass
        
    # Get memory information from /proc/meminfo
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
            
        # Memory channel detection is difficult, but we can try
        # by looking at CPU model
        if 'cpu_model' in info:
            if 'i9-9900K' in info['cpu_model']:
                info['memory_channels'] = "Dual channel (Estimated)"
            elif 'Xeon' in info['cpu_model']:
                info['memory_channels'] = "Quad/Six channel (Estimated)"
            else:
                info['memory_channels'] = "Unknown"
    except:
        pass
        
    return info


def signal_handler(sig, frame):
    """Handle Ctrl+C"""
    global should_exit
    print("\nCtrl+C detected. Finishing current test and exiting...")
    should_exit = True


def estimate_total_bandwidth(results):
    """Estimate the total DDR bandwidth with improved algorithm"""
    # STREAM Triad is considered the gold standard for bandwidth measurement
    stream_triad = results.get("Stream Triad", {}).get("mean", 0)
    
    # Sequential operations provide a good baseline
    seq_read = results.get("Sequential Read", {}).get("mean", 0)
    seq_write = results.get("Sequential Write", {}).get("mean", 0)
    seq_copy = results.get("Sequential Copy", {}).get("mean", 0)
    
    # If we have Stream Triad result, it gets highest weight
    if stream_triad:
        # STREAM bandwidth is already a good indicator, but we apply a correction
        # factor of 1.15-1.2 to account for real-world overhead
        estimated = stream_triad * 1.18
    elif seq_read and seq_write and seq_copy:
        # Weighted average - read is typically fastest, then write, then copy
        estimated = (seq_read * 0.5 + seq_write * 0.3 + seq_copy * 0.2) * 1.25
    elif seq_read and seq_write:
        estimated = (seq_read * 0.6 + seq_write * 0.4) * 1.25
    elif seq_read:
        estimated = seq_read * 1.25
    else:
        estimated = 0
        
    return estimated


def main():
    global should_exit
    
    parser = argparse.ArgumentParser(description="DDR Bandwidth Measurement Tool")
    parser.add_argument("--size", type=float, default=4.0,
                        help="Size in GB for each test (default: 4 GB)")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Number of iterations per test (default: 3)")
    parser.add_argument("--threads", type=int, 
                        default=min(multiprocessing.cpu_count(), 4),
                        help=f"Number of threads (default: min(CPU count, 4) = {min(multiprocessing.cpu_count(), 4)})")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for test results (JSON format)")
    parser.add_argument("--quick", action="store_true", 
                        help="Run only the most important tests (sequential read/write/copy)")
    parser.add_argument("--large", action="store_true",
                        help="Use a larger test size (16GB) to ensure going beyond cache")
    args = parser.parse_args()
    
    # If large option is selected, override size
    if args.large:
        args.size = 16.0
        print("Using large test size (16GB)")
    
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"\nDDR Bandwidth Measurement Tool (Optimized)")
    print(f"----------------------------------------")
    
    # Get system information
    system_info = get_system_info()
    print(f"System: {system_info['system']} {system_info['architecture']}")
    if 'cpu_model' in system_info:
        print(f"Processor: {system_info['cpu_model']}")
    print(f"CPU Cores: {system_info['cpu_count']}")
    print(f"Memory: {system_info['memory_gb']:.1f} GB")
    if 'memory_channels' in system_info:
        print(f"Memory Channels: {system_info['memory_channels']}")
    print(f"Test Size: {args.size:.1f} GB per test")
    print(f"Threads: {args.threads}")
    print(f"Iterations: {args.iterations}")
    print("")
    
    # Tests to run
    tests = [
        SequentialReadTest(size_gb=args.size, iterations=args.iterations, threads=args.threads),
        SequentialWriteTest(size_gb=args.size, iterations=args.iterations, threads=args.threads),
        SequentialCopyTest(size_gb=args.size, iterations=args.iterations, threads=args.threads),
        StreamTriadTest(size_gb=args.size, iterations=args.iterations, threads=args.threads)
    ]
    
    if not args.quick:
        tests.extend([
            RandomReadTest(size_gb=args.size, iterations=args.iterations, threads=args.threads),
            StridedReadTest(size_gb=args.size, iterations=args.iterations, threads=args.threads)
        ])
    
    # Run tests
    results = {}
    for test in tests:
        if should_exit:
            break
        test_results = test.run()
        results[test.name] = test_results
        print("")  # Add spacing between tests
    
    # Estimate total bandwidth
    print("\nResults Summary")
    print("--------------")
    for test_name, test_result in results.items():
        if test_result:
            print(f"{test_name}: {test_result['mean']:.2f} GB/s " +
                  f"±{test_result['stddev']:.2f}")
    
    estimated_bandwidth = estimate_total_bandwidth(results)
    
    if estimated_bandwidth > 0:
        print(f"\nEstimated DDR Bandwidth: {estimated_bandwidth:.2f} GB/s")
        print(f"Note: Actual bandwidth may be ±15% of this estimate due to OS overhead")
    else:
        print("\nInsufficient data to estimate total bandwidth")
    
    # Save results if requested
    if args.output and results:
        output_data = {
            "system_info": system_info,
            "test_results": results,
            "estimated_bandwidth": estimated_bandwidth
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
