#include <cuda_runtime.h> 
#include <device_launch_parameters.h> 
#include <wb.h>

#define TILE_WIDTH 16 	//do not change this value

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B


__global__ void matrixMultiplyShared(float* A, float* B, float* C,
    int numARows, int numAColumns,
    int numBColumns) {
    // Calculate the row index of C and the corresponding elements in A and B
    int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // Shared memory for the tiles of A and B
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    // Accumulator for the element C[Row][Col]
    float pValue = 0.0f;

    // Loop over all tiles in the row of A and the column of B
    for (int p = 0; p < (numAColumns + TILE_WIDTH - 1) / TILE_WIDTH; ++p) {

        // Load the tile of A into shared memory
        if (Row < numARows && (p * TILE_WIDTH + threadIdx.x) < numAColumns) {
            As[threadIdx.y][threadIdx.x] = A[Row * numAColumns + (p * TILE_WIDTH + threadIdx.x)];
        }
        else {
            As[threadIdx.y][threadIdx.x] = 0.0f;  // Padding with zeros
        }

        // Load the tile of B into shared memory
        if (Col < numBColumns && (p * TILE_WIDTH + threadIdx.y) < numAColumns) {
            Bs[threadIdx.y][threadIdx.x] = B[(p * TILE_WIDTH + threadIdx.y) * numBColumns + Col];
        }
        else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;  // Padding with zeros
        }

        // Synchronize to ensure all threads have loaded their tiles before computation
        __syncthreads();

        // Perform matrix multiplication for the tile
        for (int k = 0; k < TILE_WIDTH; ++k) {
            pValue += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // Synchronize to ensure that all computations are done before loading the next tile
        __syncthreads();
    }

    // Write the computed value to the output matrix C if within bounds
    if (Row < numARows && Col < numBColumns) {
        C[Row * numBColumns + Col] = pValue;
    }
}


int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)
  
  hostC = NULL;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows; // Rows of A
  numCColumns = numBColumns; // Columns of B
  //@@ Allocate the hostC matrix
  hostC = (float*)malloc(numCRows * numCColumns * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here

  wbCheck(cudaMalloc((void**)&deviceA, numARows * numAColumns * sizeof(float)));
  wbCheck(cudaMalloc((void**)&deviceB, numBRows * numBColumns * sizeof(float)));
  wbCheck(cudaMalloc((void**)&deviceC, numCRows * numCColumns * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");


  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here

  wbCheck(cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  dim3 dimGrid((numCColumns - 1) / TILE_WIDTH + 1, (numCRows - 1) / TILE_WIDTH + 1);


  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here

  matrixMultiplyShared <<<dimGrid, dimBlock >>> (deviceA, deviceB, deviceC, numARows, numAColumns, numBColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  wbCheck(cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");


  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here

  wbCheck(cudaFree(deviceA));
  wbCheck(cudaFree(deviceB));
  wbCheck(cudaFree(deviceC));
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
