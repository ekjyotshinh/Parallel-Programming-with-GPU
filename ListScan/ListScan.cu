#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <wb.h>

#define BLOCK_SIZE 512 

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float* input, float* output, float* aux, int len) {
	//@@ useing the workefficient version of the parallel scan
	//@@ Also make sure to store the block sum to the aux array 
	__shared__ float XY[2 * BLOCK_SIZE];
	int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len) XY[threadIdx.x] = input[i];
	if (i + blockDim.x < len)
		XY[threadIdx.x + blockDim.x] = input[i + blockDim.x];

	for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2)
	{
		__syncthreads();
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index < 2 * BLOCK_SIZE)
			XY[index] += XY[index - stride];
	}

	for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
		__syncthreads();
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index + stride < 2 * BLOCK_SIZE) {
			XY[index + stride] += XY[index];
		}
	}
	__syncthreads();
	if (i < len) output[i] = XY[threadIdx.x];

	if (i + blockDim.x < len) {
		output[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
	}


	if (aux && threadIdx.x == 0)
	{
		aux[blockIdx.x] = XY[2 * BLOCK_SIZE - 1];
	}

}

__global__ void addScannedBlockSums(float* input, float* aux, int len) {
	//@@ kernel to add scanned block sums to all values of the scanned blocks
	int i = threadIdx.x;
	int s = blockIdx.x * 2 * BLOCK_SIZE;
	__shared__ float XY;
	if (blockIdx.x > 0)
	{
		if (i == 0)
		{
			XY = input[blockIdx.x - 1];
		}
		__syncthreads();

		if (s + i < len)
		{
			aux[s + i] = aux[s + i] + XY;
		}
		if (s + BLOCK_SIZE + i < len)
		{
			aux[s + i + BLOCK_SIZE] = aux[s + i + BLOCK_SIZE] + XY;
		}
	}

}

int main(int argc, char** argv) {
	wbArg_t args;
	float* hostInput;  // The input 1D list
	float* hostOutput; // The output 1D list
	float* deviceInput;
	float* deviceOutput;
	float* deviceAuxArray, * deviceAuxScannedArray;
	int numElements; // number of elements in the input/output list

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput = (float*)wbImport(wbArg_getInputFile(args, 0), &numElements);
	hostOutput = (float*)malloc(numElements * sizeof(float));
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The number of input elements in the input is ",
		numElements);

	wbTime_start(GPU, "Allocating device memory.");
	//@@ Allocate device memory
	int size = sizeof(float) * numElements;
	cudaMalloc((void**)&deviceInput, size);
	cudaMalloc((void**)&deviceOutput, size);

	int sizeAux = sizeof(float) * ((numElements - 1) / BLOCK_SIZE + 1);
	cudaMalloc((void**)&deviceAuxArray, sizeAux);
	cudaMalloc((void**)&deviceAuxScannedArray, sizeAux);


	wbTime_stop(GPU, "Allocating device memory.");

	wbTime_start(GPU, "Clearing output device memory.");
	wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
	wbTime_stop(GPU, "Clearing output device memory.");

	wbTime_start(GPU, "Copying input host memory to device.");
	//@@ Copy input host memory to device	
	cudaMemcpy(deviceInput, hostInput, size, cudaMemcpyHostToDevice);
	cudaMemset(deviceAuxArray, 0, sizeAux);
	cudaMemset(deviceAuxScannedArray, 0, sizeAux);
	cudaMemset(deviceOutput, 0, size);


	wbTime_stop(GPU, "Copying input host memory to device.");

	//@@ Initialize the grid and block dimensions here
	dim3 dimBlock(BLOCK_SIZE, 1);
	dim3 dimGrid((numElements - 1) / BLOCK_SIZE + 1, 1, 1);


	wbTime_start(Compute, "Performing CUDA computation");

	scan << <dimGrid, dimBlock >> > (deviceInput, deviceOutput, deviceAuxArray, numElements);

	if (numElements > BLOCK_SIZE)
	{
		scan << <(1, 1, 1), dimBlock >> > (deviceAuxArray, deviceAuxScannedArray, NULL, ((numElements - 1) / BLOCK_SIZE + 1));

		addScannedBlockSums << <dimGrid, dimBlock >> > (deviceAuxScannedArray, deviceOutput, numElements);
	}

	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output device memory to host");

	//@@ Copy results from device to host	
	cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);

	wbTime_stop(Copy, "Copying output device memory to host");

	wbTime_start(GPU, "Freeing device memory");
	//@@ Deallocate device memory
	cudaFree(deviceOutput);
	cudaFree(deviceInput);
	cudaFree(deviceAuxScannedArray);
	cudaFree(deviceAuxArray);
	wbTime_stop(GPU, "Freeing device memory");

	wbSolution(args, hostOutput, numElements);

	free(hostInput);
	free(hostOutput);

	return 0;
}