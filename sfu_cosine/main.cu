#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define N 3600 
#define PI 3.14159265358979323846
#define DEG_TO_RAD(deg)  ((deg) / 180.0 * (PI))

__global__ void normal_cosine_function(double *B_d, double *radius_d)
{
	for (int i = 0; i<=N; i+=1) {
		B_d[i] = __cosf(radius_d[i]);
	}
}


int main()
{
	int i;
	double B[N];      // HOST
	double radius[N];    // HOST
	double *B_d;      // DEVICE
	double *radius_d; // DEVICE
	double deg = 0.0;
	FILE *outputfile;

	outputfile = fopen("output_cosine.txt", "w"); 
	if (outputfile == NULL) {
		printf("cannot open file! \n");
		exit(1);
	}

	for (int i = 0; i <= N; i+=1) {
		radius[i] = DEG_TO_RAD(deg);
		deg += 0.1;
	}

        dim3 blocks(1,1,1);
        dim3 threads(1,1,1);

	cudaMalloc( (void**) &B_d, N*sizeof(double));
	cudaMalloc( (void**) &radius_d, N*sizeof(double));
	
	cudaMemcpy(B_d, B, N*sizeof(double), cudaMemcpyHostToDevice); 
	cudaMemcpy(radius_d, radius, N*sizeof(double), cudaMemcpyHostToDevice); 
	
	normal_cosine_function<<< blocks, threads >>>(B_d, radius_d);

        cudaMemcpy(B, B_d, N*sizeof(double), cudaMemcpyDeviceToHost);
	
	for(i=0;i<=N;i+=1){
		fprintf(outputfile,"%d %.16f\n",i, B[i]);
	}

	fclose(outputfile);

        cudaFree(B_d);
        cudaFree(radius_d);

    return 0;
}
