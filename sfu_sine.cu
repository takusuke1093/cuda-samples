#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define N 3600 
#define PI 3.14159265358979323846
#define DEG_TO_RAD(deg)  ((deg) / 180.0 * (PI))

__global__ void sfu_and_normal_sine_function(double *A_d, double *B_d)
{
	double deg = 0.0;	
	
	for (int i = 0; i<=N; i+=1) {
		double radius = DEG_TO_RAD(deg);
		A_d[i] = __sinf(radius);
		B_d[i] = sin(radius);
		deg+=0.1;
	}
}


int main()
{
	int i;
	double A[N];   // HOST
	double B[N];   // HOST
	double *A_d;   // DEVICE
	double *B_d;   // DEVICE
	FILE *outputfile;

	outputfile = fopen("output_sine.txt", "w"); 
	if (outputfile == NULL) {
		printf("cannot open file! \n");
		exit(1);
	}

        dim3 blocks(1,1,1);
        dim3 threads(1,1,1);

	cudaMalloc( (void**) &A_d, N*sizeof(double));
	cudaMalloc( (void**) &B_d, N*sizeof(double));
	
	cudaMemcpy(A_d, A, N*sizeof(double), cudaMemcpyHostToDevice); 
	cudaMemcpy(B_d, B, N*sizeof(double), cudaMemcpyHostToDevice); 
	
	sfu_and_normal_sine_function<<< blocks, threads >>>(A_d, B_d);

        cudaMemcpy(A, A_d, N*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(B, B_d, N*sizeof(double), cudaMemcpyDeviceToHost);
	
	for(i=0;i<=N;i+=1){
		fprintf(outputfile,"%d %.16f %.16f\n",i, A[i], B[i]);
	}

	fclose(outputfile);

        cudaFree(A_d);
        cudaFree(B_d);

    return 0;
}
