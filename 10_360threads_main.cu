#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define BLOCK_X 10
#define BLOCK_Y 10
#define BLOCK_Z 1

#define THREAD_X 360
#define THREAD_Y 1 
#define THREAD_Z 1

#define N 3600 
#define PI 3.14159265358979323846
#define DEG_TO_RAD(deg)  ((deg) / 180.0 * (PI))

__global__ void normal_cosine_function10_360(double *B_d, double *radius_d)
{
	int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
	int thread_index = (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

	B_d[thread_index] = cos(radius_d[thread_index]);
}


int main()
{
	int i;
	double B[N];      // HOST
	double radius[N]; // HOST
	double *B_d;      // DEVICE
	double *radius_d; // DEVICE
	double deg = 0.0;
	FILE *outputfile;

	outputfile = fopen("./outputs/10_360_cos.txt", "w"); 
	if (outputfile == NULL) {
		printf("cannot open either directory or file! \n");
		exit(1);
	}

	for (int i = 0; i < N; i+=1) {
		radius[i] = DEG_TO_RAD(deg);
		deg += 360 /(double) N;
	}

        dim3 blocks(BLOCK_X,BLOCK_Y,BLOCK_Z);
        dim3 threads(THREAD_X,THREAD_Y,THREAD_Z);

	cudaMalloc( (void**) &B_d, N*sizeof(double));
	cudaMalloc( (void**) &radius_d, N*sizeof(double));
	
	cudaMemcpy(B_d, B, N*sizeof(double), cudaMemcpyHostToDevice); 
	cudaMemcpy(radius_d, radius, N*sizeof(double), cudaMemcpyHostToDevice); 
	
	normal_cosine_function10_360<<< blocks, threads >>>(B_d, radius_d);

        cudaMemcpy(B, B_d, N*sizeof(double), cudaMemcpyDeviceToHost);
	
	for(i = 0; i < N; i += 1){
		fprintf(outputfile,"%d %.16f\n",i, B[i]);
	}

	fclose(outputfile);

        cudaFree(B_d);
        cudaFree(radius_d);

    return 0;
}
