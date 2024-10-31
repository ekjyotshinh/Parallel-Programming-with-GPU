# Parallel-Programming-with-GPU


### Project Descriptions

1. **VectorAdd**
   - **Objective:** This project introduces the CUDA API by implementing vector addition. Key steps include memory allocation on the device, transferring data from the host to the device, launching the kernel, and transferring the results back to the host.

2. **BasicMatrixMultiplication**
   - **Objective:** This project focuses on implementing a basic matrix multiplication routine using CUDA. The project involves setting the dimensions for the product matrix, allocating device memory, transferring data, launching the kernel, and returning the results.

3. **TiledMatrixMultiply**
   - **Objective:** This project builds upon the previous matrix multiplication implementation by introducing a tiled approach using shared memory. The aim is to optimize matrix multiplication for performance by utilizing shared memory, which requires additional steps for managing shared memory allocation, data transfer between global and shared memory, and kernel execution. The aim is to enhance performance compared to the basic matrix multiplication implementation.

4. **Convolution**
    - **Objective:** This project aims to implement a tiled image convolution using both shared and constant memory in CUDA. The main tasks include defining a fixed 5x5 convolution mask, processing an arbitrarily sized image, and managing memory effectively. Key steps involve allocating device memory for the image and mask, transferring data from the host to the device, launching the convolution kernel, and returning the results to the host while ensuring proper boundary handling for edge pixels.