# BAND: Bandwidth Assessment for Native DDR

![Python](https://img.shields.io/badge/python-3.6+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Introduction

BAND (Bandwidth Assessment for Native DDR) is a Python-based memory bandwidth measurement tool designed to provide results comparable to the industry-standard STREAM benchmark. It offers optimized implementations for various memory operations with a focus on performance, allowing Python developers to evaluate memory bandwidth in their environments.

## Why BAND was Created

Traditional memory bandwidth benchmarks like STREAM are written in C and require compilation for each platform. BAND was created to provide an easy-to-use Python alternative that:

1. Requires no compilation step (just Python + NumPy)
2. Produces results comparable to the C-based STREAM benchmark
3. Provides multiple optimized implementations to explore memory bandwidth characteristics
4. Offers a simple, cross-platform way to estimate DDR memory bandwidth
5. Helps Python developers understand memory performance constraints in data-intensive applications

BAND is particularly useful for:
- Data scientists working with large NumPy arrays
- Python developers optimizing memory-bound applications
- Performance engineers comparing memory subsystems across platforms
- System administrators evaluating Python performance on different hardware

## Getting Started

### Prerequisites

- Python 3.6 or higher
- Dependencies listed in requirements.txt:
  - NumPy: For efficient array operations
  - psutil: For system information collection

### Installation

```bash
# Clone the repository
git clone https://github.com/kylefoxaustin/band.git
cd band

# Install required packages using requirements.txt
pip install -r requirements.txt

# Make the script executable
chmod +x band.py
```

## How to Run

Basic usage:

```bash
./band.py
```

This will run all tests with default settings (4GB total memory, using up to 4 threads).

### Command Line Options

```bash
./band.py --size 2.0 --threads 8 --iterations 5
```

You can customize the execution with the following options:

```
--size FLOAT           Size in GB for each test (default: 4 GB)
--iterations INT       Number of iterations per test (default: 3)
--threads INT          Number of threads (default: min(CPU count, 4))
--chunk-size INT       Chunk size in KB for operations (default: varies by test)
--triad-only           Run only the triad tests for optimization experiments
--best                 Run only the best implementation for each operation
--compare              Compare to C STREAM benchmark results
--c-stream-triad FLOAT C STREAM Triad result for comparison (default: 19.98 GB/s)
--stream-file PATH     Path to a file containing the output from a STREAM-C benchmark run
--enable-chunking      Enable chunked implementations that may benefit from CPU cache
```

## Explanation of CLI Options

- **--size**: Total memory size to use for testing. Larger values provide more accurate results but require more RAM. Recommended to use at least 2-4 GB for meaningful results.

- **--iterations**: Number of times each test is run. The first iteration is considered a warm-up and excluded from final results.

- **--threads**: Number of threads to use. Defaults to the minimum of available CPU cores or 4. More threads can help utilize multi-channel memory systems.

- **--chunk-size**: Size of data chunks processed in each iteration, measured in KB. Different chunk sizes can significantly impact performance due to cache effects.

- **--triad-only**: Run only the Triad tests, which combine read, write and arithmetic operations. Useful for focusing on the most comprehensive memory test.

- **--best**: Run only the best implementation for each operation type, which is useful for maximum performance measurement with minimal testing time.

- **--compare**: Enable comparison with C STREAM benchmark results using default reference values.

- **--c-stream-triad**: Specify your C STREAM Triad result in GB/s for direct comparison.

- **--stream-file**: Path to a file containing the output from a STREAM-C benchmark run. BAND will parse this file to extract all benchmark values for comprehensive comparison.

- **--enable-chunking**: Enable cache-optimized implementations that use smaller chunk sizes for better cache utilization. By default, BAND uses standard implementations that match STREAM's approach of measuring pure memory bandwidth.

## Effective Bandwidth Metrics

When run, BAND provides composite bandwidth metrics that estimate real-world performance for different application types by weighting individual test results:

### General Application Bandwidth Score

This score estimates memory bandwidth for typical applications based on analysis of instruction mixes across various workloads:

```
General Score = (0.40 × Copy) + (0.25 × Scale) + (0.15 × Add) + (0.20 × Triad)
```

The weightings are derived from research on instruction frequencies in common applications, where:
- Copy operations represent about 40% of memory accesses
- Scale operations (multiply by scalar) represent about 25% 
- Add operations represent about 15%
- Complex operations like Triad represent about 20%

### LLM Bandwidth Score

This specialized metric estimates memory bandwidth for Large Language Model inference workloads, which have a distinct access pattern dominated by reads:

```
LLM Score = (0.90 × Copy) + (0.05 × Scale) + (0.025 × Add) + (0.025 × Triad)
```

LLMs are extremely read-heavy during inference, as they must retrieve vast numbers of parameters from memory while performing relatively fewer complex operations.

### Adjusted Scores

BAND also provides "adjusted" versions of both metrics where the Triad value is doubled to approximate the performance gap between Python and C implementations. This adjustment accounts for the observation that Python Triad implementations typically achieve around 50% of the performance of equivalent C implementations due to interpreter overhead.  An explanation of why this is the case is in the section ## Why Python Triad Performance Differs from STREAM.C

### Example Output

```
BAND Effective Bandwidth Metrics:
-----------------------------------
Py-STREAM results:
  - General application bandwidth score: 19.91 GB/s
  - LLM bandwidth score:                24.31 GB/s

Py-STREAM with doubled Triad (to match STREAM.C):
  - General application bandwidth score: 21.95 GB/s
  - LLM bandwidth score:                24.56 GB/s

Calculation Explanation:
  General score = (0.40 × Copy) + (0.25 × Scale) + (0.15 × Add) + (0.20 × Triad)
  LLM score     = (0.90 × Copy) + (0.05 × Scale) + (0.025 × Add) + (0.025 × Triad)
  * Adjusted scores use doubled Triad values to approximate STREAM.C performance
  * Weightings based on instruction mix analysis of typical applications
```

These metrics provide more meaningful estimations of how memory bandwidth will affect real application performance compared to looking at individual test results in isolation.

## Memory Bandwidth Measurement Approaches

BAND offers two approaches to measuring memory bandwidth:

### Standard Mode (Default)

By default, BAND uses implementations that closely follow the STREAM benchmark philosophy:

- Focus on measuring sustained memory bandwidth
- Use large array sizes that exceed cache capacity
- Minimize cache effects to get a true measure of memory subsystem performance

This mode is most useful for:
- Comparing Python performance to STREAM.C benchmark results
- Evaluating true memory bandwidth limitations
- Hardware performance comparisons

### Cache-Optimized Mode (with --enable-chunking)

When the `--enable-chunking` flag is used, BAND includes additional implementations that are optimized for cache utilization:

- Py-Chunked Triad: Uses smaller chunk sizes (512KB by default) with reused temporary arrays
- Py-Combined Triad: Uses NumPy's expression optimization with moderate chunk sizes

This mode is useful for:
- Understanding potential performance with optimized code
- Exploring cache effects on performance
- Developing cache-friendly NumPy code

The cache-optimized implementations often outperform the standard STREAM implementations by significant margins (typically 20-50%), showing the importance of cache-friendly coding in Python.

## Understanding Cache Optimization and Chunking

BAND offers an optional cache optimization mode via the `--enable-chunking` flag. Understanding how this works can help you decide when to use it and interpret the results.

### How Chunking Works

The ChunkedTriad implementation (enabled with `--enable-chunking`) processes data in small chunks (512KB by default) using a reusable temporary array. Here's how it works:

1. **Data Segmentation**: Instead of processing the entire large arrays at once, the algorithm divides them into manageable chunks
2. **Temporary Array Reuse**: It creates a single small temporary array that gets reused for each chunk
3. **Sequential Processing**: Each chunk is processed completely before moving to the next

Standard STREAM implementations intentionally use very large arrays to measure sustained memory bandwidth without cache benefits. In contrast, chunking takes advantage of cache behavior to optimize performance.

### Cache Utilization in ChunkedTriad

When using 512KB chunks on a system with, for example, a 1MB L2 cache:

- The working set (portions of arrays a, b, c, and the temporary array) is approximately 2MB total
- This exceeds the L2 cache size, so not all data remains cache-resident
- However, the temporary array can remain entirely in cache
- Between 25-50% of the data from the main arrays might remain in cache between operations
- The rest will be evicted and reloaded as needed

This improved cache utilization explains why the chunked implementation typically outperforms the standard implementation by 30-50% or more.

### When to Use Chunking

Use standard mode (default):
- When you want to measure true memory subsystem bandwidth
- For direct comparison with STREAM benchmark results
- To evaluate hardware memory performance
- For comparing different systems' memory bandwidth

Use chunking mode (`--enable-chunking`):
- When you want to see the potential performance of cache-optimized code
- To understand how much performance is left "on the table" with naive implementations
- For developing algorithms that will work with large datasets
- To experiment with different chunk sizes for your specific CPU architecture

### Experimenting with Chunk Sizes

You can customize the chunk size with the `--chunk-size` parameter (in KB):
```bash
./band.py --enable-chunking --chunk-size 256
```

Different CPUs have different cache sizes and behaviors. Experimenting with chunk sizes can reveal the optimal size for your specific hardware:

- Values smaller than your L1 cache (typically 32-64KB per core) might show best performance
- Values that fit in L2 cache (typically 256KB-1MB per core) often perform well
- Values that exceed L3 cache (typically 2-32MB shared) will approach standard STREAM performance

Observing how performance changes with different chunk sizes can provide insights into your CPU's cache hierarchy and help you optimize real-world NumPy code.

## Comparing with STREAM-C Results

BAND offers built-in functionality to compare its results with the industry-standard STREAM benchmark written in C. This allows you to evaluate how the Python implementation performs relative to native code.

### Obtaining STREAM-C Results

You can use the included `setup_stream.sh` script to download, compile, and run the STREAM benchmark:

```bash
# Download, compile and run STREAM-C
./setup_stream.sh
```

The script will:
1. Download the STREAM benchmark source code
2. Compile it with appropriate optimizations
3. Run the benchmark
4. Save the results to `stream_results.txt`

Alternatively, you can manually compile and run STREAM:

```bash
# Download and compile STREAM
git clone https://github.com/jeffhammond/STREAM.git
cd STREAM
gcc -O3 -fopenmp -DSTREAM_ARRAY_SIZE=100000000 -DNTIMES=10 stream.c -o stream_omp

# Run STREAM-C with multiple threads
export OMP_NUM_THREADS=4  # Adjust based on your system
./stream_omp > stream_results.txt
```

### Automated Comparison with STREAM-C Results

To compare BAND results with the STREAM-C results:

```bash
./band.py --stream-file stream_results.txt
```

This approach provides:
- Comprehensive comparison using all measured operations (Copy, Scale, Add, Triad)
- Automatic unit conversion (BAND reports in GB/s, STREAM-C in MB/s)
- Fair comparison using standard STREAM mode by default

To include cache-optimized implementations in the comparison:

```bash
./band.py --stream-file stream_results.txt --enable-chunking
```

## Example Results

Below is an example of running BAND with STREAM-C comparison:

```
Reading STREAM benchmark results from stream_results.txt
STREAM-C results detected:
  Copy: 26224.70 MB/s
  Scale: 18053.20 MB/s
  Add: 20061.90 MB/s
  Triad: 20014.90 MB/s

BAND: Bandwidth Assessment for Native DDR
----------------------------------------
System: Linux x86_64
Processor: Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz
CPU Cores: 16
Memory: 62.7 GB
Test Size: 4.0 GB per test
Threads: 4
Iterations: 3

Running Py-STREAM Copy test with 4 threads, 4.0 GB total memory...
  Iteration 1/3... 26.45 GB/s
  Iteration 2/3... 26.63 GB/s
  Iteration 3/3... 26.56 GB/s
  Py-STREAM Copy result: 26.60 GB/s (min: 26.56, max: 26.63)

[Additional test results omitted for brevity]

Results Summary
--------------
Py-STREAM Copy: 26.60 GB/s
Py-STREAM Scale: 17.42 GB/s
Py-STREAM Add: 19.14 GB/s
Py-STREAM Triad: 10.22 GB/s
Py-MEMCPY: 26.46 GB/s

Python vs C Comparison:
  Py-STREAM Copy: 27238.40 MB/s (103.9% of STREAM.C Copy @ 26224.70 MB/s)
  Py-STREAM Scale: 17837.08 MB/s (98.8% of STREAM.C Scale @ 18053.20 MB/s)
  Py-STREAM Add: 19599.36 MB/s (97.7% of STREAM.C Add @ 20061.90 MB/s)
  Py-STREAM Triad: 10465.28 MB/s (52.3% of STREAM.C Triad @ 20014.90 MB/s)
  Py-MEMCPY: 27093.74 MB/s (103.3% of STREAM.C Copy @ 26224.70 MB/s)
```

When running with `--enable-chunking`, you'll see additional results:

```
Results Summary
--------------
Py-STREAM Copy: 26.60 GB/s
Py-STREAM Scale: 17.42 GB/s
Py-STREAM Add: 19.14 GB/s
Py-STREAM Triad: 10.22 GB/s
Py-Chunked Triad: 14.64 GB/s
Py-Combined Triad: 10.78 GB/s
Py-MEMCPY: 26.46 GB/s

Triad Implementation Comparison (vs Py-STREAM Triad):
  Py-Chunked Triad: 14.64 GB/s (+43.3%)
  Py-Combined Triad: 10.78 GB/s (+5.5%)

Python vs C Comparison:
  [Same output as above plus the chunked implementations]
  Py-Chunked Triad: 14991.36 MB/s (74.9% of STREAM.C Triad @ 20014.90 MB/s)
  Py-Combined Triad: 11034.72 MB/s (55.1% of STREAM.C Triad @ 20014.90 MB/s)
```

## Understanding the Results

BAND provides several performance metrics:

1. **Individual test results**: Raw bandwidth for each implementation in GB/s
2. **Triad implementation comparison**: When using `--enable-chunking`, shows how different Triad implementations compare
3. **Python vs C comparison**: Shows how each Python implementation compares to its C counterpart in MB/s and percentage

Typically:
- Py-STREAM Copy and Py-MEMCPY operations in Python achieve 90-105% of STREAM.C performance
- Py-STREAM Scale operations achieve 85-100% of STREAM.C performance
- Py-STREAM Add operations achieve 85-98% of STREAM.C performance
- Standard Py-STREAM Triad operations achieve 50-60% of STREAM.C performance
- Cache-optimized Py-Chunked Triad implementations can achieve 70-85% of STREAM.C performance

The lower performance of standard Triad operations highlights NumPy's overhead in complex operations, while the significant improvement with cache optimization demonstrates the importance of cache-friendly code in Python.

## Why Python Triad Performance Differs from STREAM.C

Users may notice that the standard Py-STREAM Triad implementation typically achieves only 50-60% of the performance measured by STREAM.C, while simpler operations like Copy and Scale achieve 90-105% of STREAM.C performance. This significant difference in the Triad operation is worth explaining:

### Reasons for Lower Triad Performance

1. **Operation Complexity**: The Triad operation (`a = b + scalar × c`) is more complex than Copy or Scale, involving multiple arrays, a multiplication, and an addition.

2. **NumPy Overhead**: For complex operations, Python's NumPy introduces several layers of overhead:
   - Type checking and array shape validation
   - Temporary array creation for intermediate results
   - Python interpreter overhead for managing multiple operations
   - Memory allocation and garbage collection costs

3. **Memory Access Patterns**: The Triad operation requires reading from two arrays (`b` and `c`) and writing to a third (`a`), creating more complex memory access patterns that expose inefficiencies in Python's memory management.

4. **Optimization Limitations**: C compilers can perform low-level optimizations that aren't available to NumPy, such as:
   - Instruction-level parallelism
   - Register allocation optimization
   - Loop unrolling and vectorization tuned to the specific CPU architecture

### Implications

This performance gap is an intrinsic characteristic when comparing Python to compiled C code for memory-intensive operations. It's not a flaw in the benchmark but rather an accurate reflection of the trade-offs between Python's ease of use and C's performance.

The Chunked Triad implementation (available with `--enable-chunking`) partially mitigates these issues through better cache utilization, achieving 70-85% of STREAM.C performance, demonstrating that algorithmic improvements can significantly narrow the gap.

This behavior is precisely why BAND is valuable: it helps Python developers understand the memory bandwidth limitations they might encounter in real-world applications and the potential gains from cache-friendly programming techniques.

## Performance Optimization Options

To achieve the best memory bandwidth results, consider trying:

1. **Experiment with thread count** 
   - Match the number of threads to your CPU's memory channels for optimal results
   - Try powers of 2: `--threads 1`, `--threads 2`, `--threads 4`, `--threads 8`

2. **Try different chunk sizes**
   - Smaller chunks (16-128KB) may work better on systems with small caches (with `--enable-chunking`)
   - Larger chunks (1-8MB) often work better on server-class hardware 
   - Example: `--chunk-size 512` or `--chunk-size 4096`

3. **Optimize for your workload**
   - Use `--triad-only` to focus on the most comprehensive test
   - Compare standard vs cache-optimized implementations with `--enable-chunking`

4. **System-level optimizations**
   - Run with elevated process priority
   - Disable CPU frequency scaling
   - Close other memory-intensive applications
   - Try setting process affinity to specific NUMA nodes if applicable

5. **Memory configurations**
   - Test with various memory configurations (dual vs. single channel)
   - Compare DIMM speeds and configurations if possible

## Automated STREAM Setup

For convenience, BAND includes a shell script to download, compile, and run the STREAM benchmark:

```bash
./setup_stream.sh
```

The script will:
1. Check for required dependencies (gcc, git)
2. Download the STREAM source code
3. Compile it with appropriate optimizations
4. Run the benchmark with your system's CPU core count
5. Save the results to `stream_results.txt`

After running this script, you can use the STREAM results with BAND:

```bash
./band.py --stream-file stream_results.txt
```

## Attestation

Maintained by Kyle Fox ([@kylefoxaustin](https://github.com/kylefoxaustin)).

This project is intended for educational and performance measurement purposes.
Contributions, bug reports, and feature requests are welcome.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
