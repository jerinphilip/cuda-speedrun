#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>

__device__ volatile int secondkernel, preprocessingdone, perthread;

__device__ void preprocess() {
	int a[10] = {0};
	for (int ii = 0; ii < 10000; ++ii)
		a[ii % 10]++;
}
__global__ void K1(volatile int *perthread) {
	preprocess();
	__syncthreads();

	preprocessingdone = 1;
	*perthread = 100;	// some number.

	if (secondkernel) {
		*perthread /= 2;
	}
	//if (threadIdx.x == 0) printf("perthread = %d\n", *perthread);
}
__global__ void K2(volatile int *perthread) {

	if (preprocessingdone)
		;	// do nothing.
	else {
		secondkernel = 1;
		*perthread = 100 / 2;
	}
}

__global__ void Kinit() {
	secondkernel = 0;
	preprocessingdone = 0;
}
int main() {
    srand(time(NULL));
    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    volatile int *perthread;
    cudaMalloc((int **)&perthread, sizeof(int));

    for (int ii = 0; ii < 10; ++ii) {

	Kinit<<<1, 1>>>();
	cudaDeviceSynchronize();

	K1<<<1, 64, 0, s1>>>(perthread);

	if (rand() % 2) {
		K2<<<1, 64, 0, s2>>>(perthread);
		printf("two kernels: ");
	} else
		printf("one kernel: ");

	cudaDeviceSynchronize();

	int hpt;
	cudaMemcpy(&hpt, (int *)perthread, sizeof(int), cudaMemcpyDeviceToHost);
	printf("per thread = %d\n", hpt);
    }

    return 0;
}
