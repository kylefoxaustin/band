#!/bin/bash
#
# setup_stream.sh - Download, compile and run the STREAM benchmark
# For use with BAND (Bandwidth Assessment for NumPy and DDR)
#
# This script:
# 1. Checks for required dependencies
# 2. Downloads STREAM from GitHub
# 3. Compiles with appropriate optimizations
# 4. Runs the benchmark with all available cores
# 5. Saves the results to a file
#

# Stop on error
set -e

# Print current directory
echo "Working in current directory: $(pwd)"

# Check for required dependencies
check_dependencies() {
    echo "Checking for required dependencies..."

    # Check for gcc
    if ! command -v gcc &> /dev/null; then
        echo "ERROR: gcc compiler not found. Please install gcc."
        echo "On Ubuntu/Debian: sudo apt-get install build-essential"
        echo "On Fedora/RHEL: sudo dnf install gcc"
        exit 1
    fi

    # Check for git (optional, we can download directly if not available)
    if ! command -v git &> /dev/null; then
        USE_GIT=false
        # Check for curl or wget
        if command -v curl &> /dev/null; then
            DOWNLOAD_CMD="curl -L -o stream.c"
        elif command -v wget &> /dev/null; then
            DOWNLOAD_CMD="wget -O stream.c"
        else
            echo "ERROR: Neither git, curl, nor wget found. Please install one of them."
            echo "On Ubuntu/Debian: sudo apt-get install git"
            echo "On Fedora/RHEL: sudo dnf install git"
            exit 1
        fi
    else
        USE_GIT=true
    fi
}

# Download STREAM source code
download_stream() {
    echo "Downloading STREAM benchmark..."

    # Create a temporary directory
    TEMP_DIR="stream_temp"

    if [ "$USE_GIT" = true ]; then
        # Clone the repo
        if [ -d "$TEMP_DIR" ]; then
            rm -rf "$TEMP_DIR"
        fi

        git clone --depth 1 https://github.com/jeffhammond/STREAM.git "$TEMP_DIR"
        cp "$TEMP_DIR/stream.c" .
        rm -rf "$TEMP_DIR"
    else
        # Direct download
        STREAM_URL="https://raw.githubusercontent.com/jeffhammond/STREAM/master/stream.c"
        $DOWNLOAD_CMD "$STREAM_URL"
    fi

    # Verify the download
    if [ ! -f "stream.c" ]; then
        echo "ERROR: Failed to download stream.c"
        exit 1
    fi
}

# Compile STREAM
compile_stream() {
    echo "Compiling STREAM benchmark..."

    # Detect number of CPU cores for optimal thread settings
    if command -v nproc &> /dev/null; then
        NUM_CORES=$(nproc)
    elif [ -f /proc/cpuinfo ]; then
        NUM_CORES=$(grep -c processor /proc/cpuinfo)
    else
        NUM_CORES=4  # Default if we can't detect
    fi

    echo "Detected $NUM_CORES CPU cores"

    # Compile with OpenMP support and optimizations
    gcc -O3 -fopenmp -DSTREAM_ARRAY_SIZE=100000000 -DNTIMES=10 stream.c -o stream_omp

    if [ ! -f "stream_omp" ]; then
        echo "ERROR: Compilation failed"
        exit 1
    fi

    echo "STREAM benchmark compiled successfully as 'stream_omp'"
    echo "To run the benchmark with all CPU cores, execute:"
    echo "export OMP_NUM_THREADS=$NUM_CORES && ./stream_omp"
}

# Run STREAM benchmark
run_stream() {
    echo "Running benchmark now with $NUM_CORES threads..."
    export OMP_NUM_THREADS=$NUM_CORES
    ./stream_omp | tee stream_results.txt

    echo "Results saved to stream_results.txt"
    echo "To use these results with BAND, run:"
    echo "./band.py --stream-file stream_results.txt"
}

# Main execution
check_dependencies
download_stream
compile_stream
run_stream

echo "STREAM benchmark setup complete."
