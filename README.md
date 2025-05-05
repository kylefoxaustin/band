# BAND: Bandwidth Assessment for NumPy and DDR

![Python](https://img.shields.io/badge/python-3.6+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Introduction

BAND (Bandwidth Assessment for NumPy and DDR) is a Python-based memory bandwidth measurement tool designed to provide results comparable to the industry-standard STREAM benchmark. It offers optimized implementations for various memory operations with a focus on performance, allowing Python developers to evaluate memory bandwidth in their environments.

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

## Comparing with STREAM-C Results

BAND offers built-in functionality to compare its results with the industry-standard STREAM benchmark written in C. This allows you to evaluate how the Python implementation performs relative to native code.

### Obtaining STREAM-C Results

To compare with STREAM-C:

1. **Download and compile STREAM**:
   ```bash
   # Clone the STREAM repository or download from https://www.cs.virginia.edu/stream/
   git clone https://github.com/jeffhammond/STREAM.git
   cd STREAM
   
   # Compile with OpenMP support
   gcc -O3 -fopenmp -DSTREAM_ARRAY_SIZE=100000000 -DNTIMES=10 stream.c -o stream_omp
   ```

2. **Run STREAM-C**:
   ```bash
   # Set thread count (adjust based on your system)
   export OMP_NUM_THREADS=4
   
   # Run the benchmark
   ./stream_omp
   ```

3. **Note the results**: Look for the "Function" table in the output, which will show bandwidth for Copy, Scale, Add, and Triad operations.

Example STREAM-C output:
```
Function    Best Rate MB/s  Avg time     Min time     Max time
Copy:           25698.6     0.062431     0.062260     0.062579
Scale:          17981.4     0.089938     0.088981     0.092308
Add:            19979.9     0.120457     0.120121     0.122574
Triad:          19976.2     0.120263     0.120143     0.120580
```

### Automated Comparison with STREAM-C Results

For more accurate and convenient comparison, BAND can automatically read STREAM-C results from an output file:

1. **Save STREAM-C results to a file**:
   ```bash
   # Run STREAM-C and save output to a file
   export OMP_NUM_THREADS=4 && ./stream_omp > stream_results.txt
   ```

2. **Run BAND with the results file**:
   ```bash
   ./band.py --stream-file stream_results.txt
   ```

This approach provides several advantages:
- More accurate comparisons using all measured operations (Copy, Scale, Add, Triad)
- No need to manually extract and input values
- Ensures you're comparing with results from the same system

BAND will automatically parse the STREAM-C output file, extract the relevant benchmark values, and use them for comparison. The comparison output will show how each BAND operation compares to its STREAM-C counterpart.

## How to Use Comparison Results

When you run BAND with the `--compare` option, the tool will output something like this at the end of the results:

```
Best Python Triad (Chunked Triad) achieves 82.9% of C STREAM Triad performance
```

This comparison tells you:

1. **Which implementation performed best**: In this case, "Chunked Triad" was the fastest
2. **Relative performance**: The Python implementation achieved 82.9% of the C implementation's performance

This information is valuable for:

- **Development decisions**: Understanding the performance trade-offs of using Python vs C
- **Hardware evaluation**: Testing how well Python NumPy utilizes different memory subsystems
- **Optimization opportunities**: Seeing which Python implementations come closest to C performance

Remember that the comparison is only as accurate as the STREAM-C value you provide. For meaningful results, you should:

1. Run STREAM-C and BAND on the same hardware
2. Use similar memory sizes and thread counts
3. Run both tests with minimal system load
4. Specify the exact STREAM-C Triad value using `--c-stream-triad`

## Performance Optimization Options

To achieve the best memory bandwidth results, consider trying:

1. **Experiment with thread count** 
   - Match the number of threads to your CPU's memory channels for optimal results
   - Try powers of 2: `--threads 1`, `--threads 2`, `--threads 4`, `--threads 8`

2. **Try different chunk sizes**
   - Smaller chunks (16-128KB) may work better on systems with small caches
   - Larger chunks (1-8MB) often work better on server-class hardware 
   - Example: `--chunk-size 512` or `--chunk-size 4096`

3. **Optimize for your workload**
   - Use `--triad-only` to focus on the most comprehensive test
   - Compare the standard Triad vs ChunkedTriad implementations

4. **System-level optimizations**
   - Run with elevated process priority
   - Disable CPU frequency scaling
   - Close other memory-intensive applications
   - Try setting process affinity to specific NUMA nodes if applicable

5. **Memory configurations**
   - Test with various memory configurations (dual vs. single channel)
   - Compare DIMM speeds and configurations if possible

## Example Results

```
BAND: Bandwidth Assessment for NumPy and DDR
----------------------------------------
System: Linux x86_64
Processor: AMD Ryzen 9 5900X 12-Core Processor
CPU Cores: 16
Memory: 32.0 GB
Test Size: 4.0 GB per test
Threads: 4
Iterations: 3

Running STREAM Copy test with 4 threads, 4.0 GB total memory...
  Iteration 1/3... 21328.55 GB/s
  Iteration 2/3... 21356.82 GB/s
  Iteration 3/3... 21345.67 GB/s
  STREAM Copy result: 21351.25 GB/s (min: 21345.67, max: 21356.82)

Running STREAM Scale test with 4 threads, 4.0 GB total memory...
  Iteration 1/3... 14862.21 GB/s
  Iteration 2/3... 14891.45 GB/s
  Iteration 3/3... 14878.33 GB/s
  STREAM Scale result: 14884.89 GB/s (min: 14862.21, max: 14891.45)

Running STREAM Add test with 4 threads, 4.0 GB total memory...
  Iteration 1/3... 16345.78 GB/s
  Iteration 2/3... 16382.12 GB/s
  Iteration 3/3... 16367.45 GB/s
  STREAM Add result: 16374.79 GB/s (min: 16345.78, max: 16382.12)

Running STREAM Triad test with 4 threads, 4.0 GB total memory...
  Iteration 1/3... 16278.34 GB/s
  Iteration 2/3... 16305.67 GB/s
  Iteration 3/3... 16297.21 GB/s
  STREAM Triad result: 16301.44 GB/s (min: 16278.34, max: 16305.67)

Running Chunked Triad test with 4 threads, 4.0 GB total memory...
  Iteration 1/3... 16512.45 GB/s
  Iteration 2/3... 16567.89 GB/s
  Iteration 3/3... 16545.32 GB/s
  Chunked Triad result: 16556.61 GB/s (min: 16512.45, max: 16567.89)

Results Summary
--------------
STREAM Copy: 21351.25 GB/s
STREAM Scale: 14884.89 GB/s
STREAM Add: 16374.79 GB/s
STREAM Triad: 16301.44 GB/s
Chunked Triad: 16556.61 GB/s

Triad Implementation Comparison (vs STREAM Triad):
  Chunked Triad: 16556.61 GB/s (+1.6%)

Best Python Triad (Chunked Triad) achieves 82.9% of C STREAM Triad performance
```

## Attestation

Maintained by Kyle Fox ([@kylefoxaustin](https://github.com/kylefoxaustin)).

This project is intended for educational and performance measurement purposes.
Contributions, bug reports, and feature requests are welcome.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
