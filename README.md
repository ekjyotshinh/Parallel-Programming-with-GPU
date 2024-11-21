# CUDA Parallel Programming Projects

This repository contains several CUDA-based projects designed to showcase the power of parallel programming using GPUs. Each project focuses on a different aspect of parallel computation, with emphasis on optimizing memory usage, performance, and computational efficiency.

## Projects

### 1. **VectorAdd**
   - **Objective:**  
     Introduces the CUDA API by implementing vector addition.  
     Key steps include:
     - Memory allocation on the device.
     - Transferring data from the host to the device.
     - Launching the kernel.
     - Transferring results back to the host.

### 2. **BasicMatrixMultiplication**
   - **Objective:**  
     Implements a basic matrix multiplication routine using CUDA.  
     The project involves:
     - Setting the dimensions for the product matrix.
     - Allocating device memory.
     - Transferring data to/from the device.
     - Launching the kernel to perform matrix multiplication.

### 3. **TiledMatrixMultiply**
   - **Objective:**  
     Builds upon the basic matrix multiplication implementation by introducing a tiled approach using shared memory.  
     The project focuses on optimizing matrix multiplication for better performance by utilizing shared memory, which involves:
     - Managing shared memory allocation.
     - Efficient data transfer between global and shared memory.
     - Kernel execution optimizations for performance.

### 4. **Convolution**
   - **Objective:**  
     Implements a tiled image convolution using both shared and constant memory in CUDA.  
     The main tasks include:
     - Defining a fixed 5x5 convolution mask.
     - Processing an arbitrarily sized image.
     - Managing memory efficiently.
     - Handling boundary conditions for edge pixels.
     - Transferring the image and mask data to/from the device.

### 5. **Histogram**
   - **Objective:**  
     Implements a histogram calculation using parallel programming in CUDA.  
     Key points:
     - Calculates a histogram of input data (such as an array of values).
     - Uses shared memory to store partial histograms.
     - Reduces global memory writes by minimizing contention and optimizing memory access patterns.
     - Uses atomic operations to update the global histogram in parallel.

### 6. **ListScan**
   - **Objective:**  
     Implements parallel prefix sum (or scan) in CUDA.  
     Key tasks include:
     - Computing the prefix sum of a list of numbers in parallel.
     - Using a divide and conquer approach, splitting the list, performing local scans, and combining results.
     - Leveraging shared memory to store intermediate results for efficient processing.

## Installation

To run the projects, ensure that you have the following prerequisites:

- CUDA toolkit installed.
- A compatible NVIDIA GPU.
- A Linux or Windows machine with a CUDA-compatible driver.

