#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define BLOCK_X 5
#define BLOCK_Y 2
#define BLOCK_Z 2

#define THREAD_X 6
#define THREAD_Y 6 
#define THREAD_Z 5

#define N 3600 
#define PI 3.14159265358979323846
#define DEG_TO_RAD(deg)  ((deg) / 180.0 * (PI))

__global__ void cosine5_2_2__6_6_5(double *B_d, double *radius_d)
{
	int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
	int threads_count_before_current_block = (threads_per_block * gridDim.x * gridDim.y * blockIdx.z) + (threads_per_block * gridDim.x * blockIdx.y) + threads_per_block * blockIdx.x;
	int thread_index = threads_count_before_current_block + (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

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

	outputfile = fopen("./outputs/cosine5_2_2__6_6_5.txt", "w"); 
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
	
	cosine5_2_2__6_6_5<<< blocks, threads >>>(B_d, radius_d);

        cudaMemcpy(B, B_d, N*sizeof(double), cudaMemcpyDeviceToHost);
	
	for(i = 0; i < N; i += 1){
		fprintf(outputfile,"%d %.16f\n",i, B[i]);
	}
	for(i = 0; i < 5; i += 1){
		printf("%d %.16f\n",i, B[i]);
	}
	fclose(outputfile);

        cudaFree(B_d);
        cudaFree(radius_d);

    return 0;
}
