
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <random>
#include <windows.h>
#include <time.h> 

static const double VALUES_MIN = -1.0;
static const double VALUES_MAX = 1.0;
static const size_t CUDA_BLOCK_SIZE = 32;
static const size_t MATRIX_SIZES_TO_TEST[3] = { 500, 1000, 1500 };

double* fill_matrix_rnd(size_t matrix_size) {

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> distrib(VALUES_MIN, VALUES_MAX);
	size_t element_count = matrix_size * matrix_size;
	double* result = new double[element_count];
	for (size_t i = 0; i < element_count; ++i) {
		result[i] = distrib(gen);
	}
	return result;
}

void cpu_mul(double* matrix_A, double* matrix_B, double* result, size_t matrix_size) {
	for (size_t i = 0; i < matrix_size; ++i) {
		for (size_t j = 0; j < matrix_size; ++j) {
			size_t ind = i * matrix_size + j;
			result[ind] = 0.0;

			for (size_t k = 0; k < matrix_size; ++k) {
				result[ind] += matrix_A[i * matrix_size + k] * matrix_B[k * matrix_size + j];
			}
		}
	}
}

double process_on_cpu(double* matrix_A, double* matrix_B, double* result, size_t matrix_size) {
	clock_t start = clock();

	cpu_mul(matrix_A, matrix_B, result, matrix_size);

	clock_t end = clock();
	return static_cast<double>(end - start) / CLOCKS_PER_SEC;
}


__global__ void mul_on_gpu_kernel(double* a, double* b, double* result, size_t matrix_size) {
	size_t i = blockDim.y * blockIdx.y + threadIdx.y;
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= matrix_size  || j >= matrix_size)
		return;

	size_t ind = i * matrix_size + j;
	result[ind] = 0;

	for (size_t k = 0; k < matrix_size; ++k) {
		result[ind] += a[i * matrix_size + k] * b[k * matrix_size + j];
	}
}


double process_on_gpu(double* matrix_A, double* matrix_B, double* result, size_t matrix_size) {
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start, 0);

	double* gpu_mem_A;
	double* gpu_mem_B;
	double* gpu_mem_res;
	size_t bytes_count = matrix_size * matrix_size * sizeof(double);
	cudaMalloc((void**)&gpu_mem_A, bytes_count);
	cudaMalloc((void**)&gpu_mem_B, bytes_count);
	cudaMalloc((void**)&gpu_mem_res, bytes_count);

	cudaMemcpy(gpu_mem_A, matrix_A, bytes_count, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_mem_B, matrix_B, bytes_count, cudaMemcpyHostToDevice);

	dim3 cuda_threads(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE);
	dim3 cuda_blocks((matrix_size + cuda_threads.x - 1) / cuda_threads.x, (matrix_size + cuda_threads.y - 1) / cuda_threads.y);

	mul_on_gpu_kernel<<<cuda_blocks, cuda_threads >>> (gpu_mem_A, gpu_mem_B, gpu_mem_res, matrix_size);

	cudaMemcpy(result, gpu_mem_res, bytes_count, cudaMemcpyDeviceToHost);

	cudaFree(gpu_mem_A);
	cudaFree(gpu_mem_B);
	cudaFree(gpu_mem_res);

	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	float elapsed_time;
	cudaEventElapsedTime(&elapsed_time, start, end);

	cudaEventDestroy(start);
	cudaEventDestroy(end);

	return elapsed_time / 1000.0f;
}

double get_max_diff(double* matrix_A, double* matrix_B, size_t matrix_size) {
	size_t element_count = matrix_size * matrix_size;
	double result = 0.0;
	for (size_t i = 0; i < element_count; ++i) {

		result = std::max(result, std::fabs(matrix_A[i] - matrix_B[i]));
	}

	return result;
}

int main(int argc, char* argv[]) {

	for (size_t matrix_size : MATRIX_SIZES_TO_TEST) {
		double* matrix_A = fill_matrix_rnd(matrix_size);
		double* matrix_B = fill_matrix_rnd(matrix_size);
		double* res_on_cpu = new double[matrix_size * matrix_size];
		double* res_on_gpu = new double[matrix_size * matrix_size];

		float time_on_cpu = process_on_cpu(matrix_A, matrix_B, res_on_cpu, matrix_size);
		float time_on_gpu = process_on_gpu(matrix_A, matrix_B, res_on_gpu, matrix_size);
		double max_diff = get_max_diff(res_on_cpu, res_on_gpu, matrix_size);
	
		std::cout << "-------------------------------" << std::endl;
		std::cout << "matrix_size: \t" << matrix_size << std::endl;
		std::cout << "time on CPU: \t" << time_on_cpu << std::endl;
		std::cout << "time on GPU: \t" << time_on_gpu << std::endl;
		std::cout << "max diff: \t" << max_diff << std::endl;
		std::cout << std::endl;

		delete[] matrix_A;
		delete[] matrix_B;
		delete[] res_on_cpu;
		delete[] res_on_gpu;
	}
	return 0;
}