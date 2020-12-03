#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define N 3600 
#define PI 3.14159265358979323846
#define DEG_TO_RAD(deg)  ((deg) / 180.0 * (PI))

__global__ void matrix_vector_multi_gpu_1_1(double *A_d)
{
	double deg = 0.0;	
	
	for (int i = 0; i<=N; i+=1) {
		A_d[i] = cos(DEG_TO_RAD(deg));
		deg+=0.1;
	}
}


int main()
{
	int i;
	double A[N];   //HOST
	double *A_d;     //DEVICE
	FILE *outputfile;

	outputfile = fopen("output_normal_cosine.txt", "w"); 
	if (outputfile == NULL) {
		printf("cannot open file! \n");
		exit(1);
	}

        dim3 blocks(1,1,1);
        dim3 threads(1,1,1);

	cudaMalloc( (void**) &A_d, N*sizeof(double));
	
	cudaMemcpy(A_d, A, N*sizeof(double), cudaMemcpyHostToDevice); 
	
	matrix_vector_multi_gpu_1_1<<< blocks, threads >>>(A_d);

        cudaMemcpy(A, A_d, N*sizeof(double), cudaMemcpyDeviceToHost);
	
	for(i=0;i<=N;i+=1){
		fprintf(outputfile,"%f \n",A[i]);
	}

	fclose(outputfile);

        cudaFree(A_d);

    return 0;
}
