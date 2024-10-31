#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <wb.h>

#define MASK_WIDTH  5
#define MASK_RADIUS MASK_WIDTH / 2
#define TILE_WIDTH 16
#define SHARED_MEMORY_WIDTH (TILE_WIDTH + MASK_WIDTH - 1)
#define CLAMP(x) (min(max((x), 0.0), 1.0))

__global__ void convolution(float* inputImage, const float* __restrict__ convolutionMask, float* outputImage,
    int numChannels, int imageWidth, int imageHeight) {
    __shared__ float sharedMemory[SHARED_MEMORY_WIDTH][SHARED_MEMORY_WIDTH];
    int channel;

    for (channel = 0; channel < numChannels; channel++) {
        // First batch loading
        int destIndex = threadIdx.y * TILE_WIDTH + threadIdx.x;
        int destY = destIndex / SHARED_MEMORY_WIDTH, destX = destIndex % SHARED_MEMORY_WIDTH;
        int srcY = blockIdx.y * TILE_WIDTH + destY - MASK_RADIUS;
        int srcX = blockIdx.x * TILE_WIDTH + destX - MASK_RADIUS;
        int srcIndex = (srcY * imageWidth + srcX) * numChannels + channel;

        if (srcY >= 0 && srcY < imageHeight && srcX >= 0 && srcX < imageWidth) {
            sharedMemory[destY][destX] = inputImage[srcIndex];
        } else {
            sharedMemory[destY][destX] = 0;
        }

        // Second batch loading
        destIndex = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
        destY = destIndex / SHARED_MEMORY_WIDTH;
        destX = destIndex % SHARED_MEMORY_WIDTH;
        srcY = blockIdx.y * TILE_WIDTH + destY - MASK_RADIUS;
        srcX = blockIdx.x * TILE_WIDTH + destX - MASK_RADIUS;
        srcIndex = (srcY * imageWidth + srcX) * numChannels + channel;

        if (destY < SHARED_MEMORY_WIDTH) {
            if (srcY >= 0 && srcY < imageHeight && srcX >= 0 && srcX < imageWidth) {
                sharedMemory[destY][destX] = inputImage[srcIndex];
            } else {
                sharedMemory[destY][destX] = 0;
            }
        }
        __syncthreads();  // Wait for all threads to finish loading

        float accumulatedValue = 0;
        int y, x;
        // Perform convolution
        for (y = 0; y < MASK_WIDTH; y++) {
            for (x = 0; x < MASK_WIDTH; x++) {
                accumulatedValue += sharedMemory[threadIdx.y + y][threadIdx.x + x] * convolutionMask[y * MASK_WIDTH + x];
            }
        }

        // Compute output pixel location
        int outputY = blockIdx.y * TILE_WIDTH + threadIdx.y;
        int outputX = blockIdx.x * TILE_WIDTH + threadIdx.x;

        if (outputY < imageHeight && outputX < imageWidth) {
            outputImage[(outputY * imageWidth + outputX) * numChannels + channel] = CLAMP(accumulatedValue);
        }
        __syncthreads();  // Synchronize threads
    }
}

int main(int argc, char* argv[]) {
    wbArg_t arg;
    int maskRows;
    int maskColumns;
    int numChannels;
    int imageWidth;
    int imageHeight;
    char* inputImageFile;
    char* inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float* hostInputImageData;
    float* hostOutputImageData;
    float* hostMaskData;
    float* deviceInputImageData;
    float* deviceOutputImageData;
    float* deviceMaskData;

    arg = wbArg_read(argc, argv);  // Parse the input arguments

    inputImageFile = wbArg_getInputFile(arg, 0);
    inputMaskFile = wbArg_getInputFile(arg, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float*)wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == MASK_WIDTH);  // Mask height is fixed to 5
    assert(maskColumns == MASK_WIDTH);  // Mask width is fixed to 5

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    numChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, numChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void**)&deviceInputImageData, imageWidth * imageHeight * numChannels * sizeof(float));
    cudaMalloc((void**)&deviceOutputImageData, imageWidth * imageHeight * numChannels * sizeof(float));
    cudaMalloc((void**)&deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");

    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
        hostInputImageData,
        imageWidth * imageHeight * numChannels * sizeof(float),
        cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
        hostMaskData,
        maskRows * maskColumns * sizeof(float),
        cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");

    wbTime_start(Compute, "Doing the computation on the GPU");
    dim3 gridDimensions(ceil((float)imageWidth / TILE_WIDTH), ceil((float)imageHeight / TILE_WIDTH));
    dim3 blockDimensions(TILE_WIDTH, TILE_WIDTH, 1);
    convolution <<<gridDimensions, blockDimensions >>> (deviceInputImageData, deviceMaskData, deviceOutputImageData,
        numChannels, imageWidth, imageHeight);
    wbTime_stop(Compute, "Doing the computation on the GPU");

    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
        deviceOutputImageData,
        imageWidth * imageHeight * numChannels * sizeof(float),
        cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(arg, outputImage);

    // Cleanup
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
