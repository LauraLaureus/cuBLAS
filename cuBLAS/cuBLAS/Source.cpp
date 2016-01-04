#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#include <mkl.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "eTimer.h"

#define N 2*1024

using namespace std;

int main(int argc, char* argv[]){

	double *host_A, *host_B, *host_C;
	double *dev_A, *dev_B, *dev_C;
	int sizematrix = N*N*sizeof(double);
	double alpha = 1.0, beta = 0.0;

	random_device gen;
	normal_distribution<double> dis(0.0, 1.0);

	host_A = (double*)mkl_malloc(sizematrix, 64);
	host_B = (double*)mkl_malloc(sizematrix, 64);
	host_C = (double*)mkl_malloc(sizematrix, 64);

	for (int y = 0; y < N; y++)
	{
		for (int x = 0; x < N; x++)
		{
			host_A[y*N + x] = dis(gen);
			host_B[y*N + x] = dis(gen);
		}
	}

	eTimer *Tcpu = new eTimer();
	eTimer *Tgpu = new eTimer();

	for (int  i = 0; i < 10; i++)
	{
		Tcpu->start();
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, host_A, N, host_B, N, beta, host_C, N);
		Tcpu->stop();
	}
	Tcpu->report("CPU");

	for (int x = 0; x < 5; x++)
	{
		printf("%g ", host_C[x]);
	}
	memset(host_C, 0, sizematrix);


	/*---------------------------------GPU----------------------------------------*/
	cudaError cudaStatus;
	cublasStatus_t cublasStatus;
	cublasHandle_t handle;
	
	cudaStatus = cudaGetDevice(0);
	cublasStatus = cublasCreate(&handle);

	cudaStatus = cudaMalloc((void**)&dev_A, sizematrix);
	cudaStatus = cudaMalloc((void**)&dev_B, sizematrix);
	cudaStatus = cudaMalloc((void**)&dev_C, sizematrix);


	cublasStatus = cublasSetMatrix(N, N, sizeof(double), host_A, N, dev_A, N);
	cublasStatus = cublasSetMatrix(N, N, sizeof(double), host_B, N, dev_B, N);

	Tgpu -> start();
	cublasStatus = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &alpha, dev_A, N, dev_B, N, &beta, dev_C, N);
	cudaStatus = cudaDeviceSynchronize();
	Tgpu->stop();
	Tgpu->report("GPU");

	cublasStatus = cublasGetMatrix(N, N, sizeof(double), dev_C, N, host_C, N);

	for (int x = 0; x < 5; x++) printf("%g ", host_C[x*N]);
	printf("\n");

	cudaStatus = cudaFree(dev_A);
	cudaStatus = cudaFree(dev_B);
	cudaStatus = cudaFree(dev_C);

	cudaStatus = cudaDeviceReset();

	mkl_free(host_A);
	mkl_free(host_B);
	mkl_free(host_C);

	delete Tcpu;
	delete Tgpu;

	getchar();


	return 0;
}
