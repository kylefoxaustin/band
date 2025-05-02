# DDR Bandwidth Measurement Tool

A cross-platform tool for measuring and estimating DDR memory bandwidth on both x86/AMD64 and ARM64 systems running Linux (tested on Ubuntu 22.04).

## Overview

This tool performs a series of memory operations to estimate the available DDR bandwidth on your system. It measures:

- Sequential read bandwidth
- Sequential write bandwidth
- Sequential copy bandwidth
- Random read bandwidth
- Strided read bandwidth

The results are combined to provide an overall estimate of your system's DDR bandwidth with an expected accuracy of ±15%.

## Features

- Multi-threaded testing to maximize bandwidth utilization
- Multiple test patterns to simulate different memory access scenarios
- Statistical analysis (min, max, mean, standard deviation)
- Works on both x86/AMD64 and ARM64 platforms
- Compatible with Linux (tested on Ubuntu 22.04)
- JSON output for further analysis

## Requirements

- Python 3.6 or higher
- NumPy library
- psutil library

## Installation

1. Install the required Python packages:

```bash
pip install numpy psutil
```

2. Make the script executable:

```bash
chmod +x ddr_bandwidth.py
```

## Usage

Basic usage:

```bash
./ddr_bandwidth.py
```

Custom configuration:

```bash
./ddr_bandwidth.py --size 2 --threads 8 --iterations 5 --output results.json
```

### Command Line Options

- `--size`: Size in GB for each test (default: 1 GB)
- `--iterations`: Number of iterations per test (default: 3)
- `--threads`: Number of threads to use (default: min(CPU count, 4))
- `--output`: Output file for test results in JSON format
- `--quick`: Run only the most important tests (sequential read/write/copy)

## Example Output

```
DDR Bandwidth Measurement Tool
-------------------------------
System: Linux x86_64
Processor: Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz
CPU Cores: 16
Memory: 32.0 GB
Test Size: 1.0 GB per test
Threads: 4
Iterations: 3

Running Sequential Read test with 4 threads, 1.0 GB total memory...
  Iteration 1/3... 24.18 GB/s
  Iteration 2/3... 24.56 GB/s
  Iteration 3/3... 24.42 GB/s
  Sequential Read result: 24.39 GB/s (min: 24.18, max: 24.56)

Running Sequential Write test with 4 threads, 1.0 GB total memory...
  Iteration 1/3... 18.32 GB/s
  Iteration 2/3... 18.45 GB/s
  Iteration 3/3... 18.26 GB/s
  Sequential Write result: 18.34 GB/s (min: 18.26, max: 18.45)

Running Sequential Copy test with 4 threads, 1.0 GB total memory...
  Iteration 1/3... 16.78 GB/s
  Iteration 2/3... 16.85 GB/s
  Iteration 3/3... 16.71 GB/s
  Sequential Copy result: 16.78 GB/s (min: 16.71, max: 16.85)

Running Random Read test with 4 threads, 1.0 GB total memory...
  Iteration 1/3... 5.23 GB/s
  Iteration 2/3... 5.26 GB/s
  Iteration 3/3... 5.24 GB/s
  Random Read result: 5.24 GB/s (min: 5.23, max: 5.26)

Running Strided Read test with 4 threads, 1.0 GB total memory...
  Iteration 1/3... 9.45 GB/s
  Iteration 2/3... 9.38 GB/s
  Iteration 3/3... 9.42 GB/s
  Strided Read result: 9.42 GB/s (min: 9.38, max: 9.45)

Results Summary
--------------
Sequential Read: 24.39 GB/s ±0.16
Sequential Write: 18.34 GB/s ±0.08
Sequential Copy: 16.78 GB/s ±0.06
Random Read: 5.24 GB/s ±0.01
Strided Read: 9.42 GB/s ±0.03

Estimated DDR Bandwidth: 25.43 GB/s
Note: Actual bandwidth may be ±15% of this estimate due to OS overhead
```

## Understanding the Results

The tool provides several bandwidth measurements:

- **Sequential Read**: Measures how quickly data can be read from memory in a sequential pattern
- **Sequential Write**: Measures how quickly data can be written to memory in a sequential pattern
- **Sequential Copy**: Measures combined read and write operations when copying memory
- **Random Read**: Measures memory access performance with random access patterns
- **Strided Read**: Measures memory access with non-contiguous patterns

The **Estimated DDR Bandwidth** represents the tool's best estimate of your system's actual DDR bandwidth, taking into account all test results and applying correction factors based on empirical testing.

## How It Works

The tool works by:

1. Allocating large blocks of memory
2. Performing various memory operations (read, write, copy)
3. Measuring the time taken to complete these operations
4. Calculating bandwidth in GB/s based on the amount of data processed and time elapsed
5. Running multiple iterations to ensure statistical reliability
6. Applying correction factors to estimate real-world bandwidth

## Limitations

- Results can be affected by system load and background processes
- Memory bandwidth is shared across all CPU cores, so maximum bandwidth may not be achievable in practice
- The operating system introduces overhead that can affect measurements
- Estimated bandwidth has an expected accuracy of approximately ±15%

## Using with JAWS

This tool complements the JAWS memory consumer tool by helping you:

1. Establish a baseline for your system's memory bandwidth
2. Compare performance before and during JAWS memory consumption
3. Measure the impact of different memory access patterns on bandwidth
4. Verify that JAWS is effectively simulating bandwidth constraints

## License

MIT License

## Troubleshooting

- If you get memory allocation errors, reduce the `--size` parameter
- If results seem inconsistent, increase the `--iterations` parameter
- For the most accurate results, close other applications and minimize system activity during testing
- On systems with NUMA architecture, results may vary depending on memory allocation across nodes
