__kernel void gemm_nn_opencl(
	const int M,
	const int N,
	const int K,
	float ALPHA,
	__global float* A, const int lda,
	__global float* B, const int ldb,
	__global float* C, const int ldc)
{
	int k;
	int i = get_global_id(0);
	int j = get_global_id(1);
	float tmp;
	if((i < M) && (j < N)){
		tmp = 0.0;
		for(k=0;k<K;k++)
			tmp += A[i*M+k] * B[k*K+j];
		C[i*M+j] = tmp;
	}
}