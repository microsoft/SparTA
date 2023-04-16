#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <algorithm>

#define M_GLOBAL M_GLOBAL_VAL
#define K_GLOBAL K_GLOBAL_VAL
#define N_GLOBAL N_GLOBAL_VAL

#define BLOCK_SIZE_M 32
#define BLOCK_SIZE_N 128
// BLOCK_SIZE_K should > NUM_BANK
#define BLOCK_SIZE_K BLOCK_SIZE_K_VAL
#define THREAD_SIZE_M 8
#define THREAD_SIZE_N 4

#define SPARSITY SPARSITY_RATIO_VAL

#define ALIGN_N THREAD_SIZE_N

#define BANK_VAL 32
const int NUM_BANK = K_GLOBAL / BANK_VAL;

const int BANK_NUM_PER_BLOCK = BLOCK_SIZE_K / BANK_VAL;
const int BLOCK_SIZE_K_SPARSE = int(BLOCK_SIZE_K * (1-SPARSITY));
const int LEN_OF_BANK_PER_SPARSE_BLOCK = BLOCK_SIZE_K_SPARSE / BANK_NUM_PER_BLOCK;

namespace batch{
void checkResult(float *hostRef, float *gpuRef, const int N, const int minibatch) {
	double epsilon = 1E-4;
	bool match = 1;
	for (int batch = 0; batch < minibatch; ++batch){
		bool should_break = false;
		for (int i=0; i<N; i++) {
			if (abs((hostRef[i + batch * N] - gpuRef[i + batch * N])/hostRef[i + batch * N]) > epsilon) {
				match = 0;
				printf("Arrays do [NOT] match!\n");
				printf("host %5.5f gpu %5.5f at current %d\n",hostRef[i + batch * N],gpuRef[i + batch * N],i + batch * N);
				should_break = true;
				break;
			}
		}
		if(should_break) break;
	}
	if (match) printf("Pass\n\n");
}

void initialData(float *vec, float *mat_data, int *mat_index, float *mat_data_for_gpu, int *mat_index_for_gpu, int vecNum, int h, float sparse, int minibatch) {
	// generate different seed for random number
	time_t t;
	srand((unsigned) time(&t));
	unsigned int w = vecNum * sparse;
	
	if (1) {
		for (int batch = 0; batch < minibatch; ++batch)
			for (int i=0; i<vecNum; ++i) {
				vec[i+vecNum*batch] = (float)rand() / RAND_MAX;
			//	printf("%f\n", vec[i]);
			}

		for (int j=0; j<h; ++j)
			for (int i=0; i<w; ++i){
				mat_data[i+j*w] = (float)rand() / RAND_MAX;
				mat_data_for_gpu[i*h+j] = mat_data[i+j*w];
			}
		int* tmp_index = (int *)malloc(vecNum / NUM_BANK * sizeof(int));
		for (int i=0; i<vecNum/NUM_BANK; ++i)
			tmp_index[i] = i;

		for (int j=0; j<h; j += ALIGN_N){
			for (int i=0; i<w; i+=w/NUM_BANK){
				std::random_shuffle(tmp_index,tmp_index+vecNum/NUM_BANK);
				std::sort(tmp_index, tmp_index+w/NUM_BANK);
				for (int k=0; k<w/NUM_BANK; ++k){
					for(int j_in = 0; j_in < ALIGN_N; j_in += 1){
						mat_index[i + k + (j + j_in) * w] = tmp_index[k]+i/sparse; // tmp_index[k] + delta(vecNum/NUM_BANK)
						mat_index_for_gpu[(i + k)*h + (j + j_in)] = mat_index[i + k + (j + j_in) * w];
					}
				}
			}
		}
		free(tmp_index);
	} else {
		FILE *vec_stream = fopen("vec.txt", "r");
		for (int i=0; i<vecNum; ++i) {
			fscanf(vec_stream, "%f", &vec[i]);
		//	printf("%f\n", vec[i]);
		}
		fclose(vec_stream);

		FILE *mat_data_stream = fopen("BSB_sparse_data.txt", "r");
		for (int j=0; j<h; ++j)
			for (int i=0; i<w; ++i)
				fscanf(mat_data_stream, "%f", &mat_data[i + j * w]);
		fclose(mat_data_stream);

		FILE *mat_index_stream = fopen("BSB_sparse_index.txt", "r");
		for (int j=0; j<h; ++j)
			for (int i=0; i<w; ++i) {
				fscanf(mat_index_stream, "%d", &mat_index[i + j * w]);
			}
		fclose(mat_index_stream);
	}
}

void MVOnHost(float *vec, float *mat_data, int *mat_index, float *hostRef, const int w, const int h, int vecNum, const int minibatch) {
	float tmp;
	for (int batch = 0;batch < minibatch; ++batch)
		for (int j=0; j<h; ++j) {
			tmp = 0;
			for (int i=0; i<w; ++i) {
				tmp += mat_data[i + j * w] * vec[mat_index[i + j * w]+batch*vecNum];
			}
			hostRef[j + batch * h] = tmp;
		}
}
}

__device__ void warpReduce(volatile float *sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

void CheckSetting(){
	assert(BLOCK_SIZE_M >= THREAD_SIZE_M && BLOCK_SIZE_M % THREAD_SIZE_M == 0);
	assert(BLOCK_SIZE_N >= THREAD_SIZE_N && BLOCK_SIZE_N % THREAD_SIZE_N == 0);

	const int THREADS_PER_BLOCK = (BLOCK_SIZE_M / THREAD_SIZE_M) * (BLOCK_SIZE_N / THREAD_SIZE_N);
	const int A_THREADS_PER_ROW = BLOCK_SIZE_K / 4;
	const int B_THREADS_PER_ROW = BLOCK_SIZE_N / 4;
	const int A_STRIDES = THREADS_PER_BLOCK / A_THREADS_PER_ROW;
	const int B_STRIDES = THREADS_PER_BLOCK / B_THREADS_PER_ROW;

	assert(BLOCK_SIZE_K >= 4 && BLOCK_SIZE_K % 4 == 0);
	assert(BLOCK_SIZE_N >= 4 && BLOCK_SIZE_N % 4 == 0);

	assert(BLOCK_SIZE_M >= A_STRIDES && BLOCK_SIZE_M % A_STRIDES == 0);
	assert(BLOCK_SIZE_K_SPARSE >= B_STRIDES && BLOCK_SIZE_K_SPARSE % B_STRIDES == 0);
}

// dim3 dimBlock((BLOCK_SIZE_M / THREAD_SIZE_M) * (BLOCK_SIZE_N / THREAD_SIZE_N));
// dim3 dimGrid(M / BLOCK_SIZE_M, N / BLOCK_SIZE_N);
__global__ void MatMul_TILE_THREAD_GENERAL(float *g_vec, float *g_mat_data, int *g_mat_index, float *g_data) {
	const int K = K_GLOBAL;
	const int N = N_GLOBAL;
	int M_BLOCK_START = blockIdx.x * BLOCK_SIZE_M;
	int N_BLOCK_START = blockIdx.y * BLOCK_SIZE_N;

	const int A_THREADS_PER_ROW = BLOCK_SIZE_K / 4;
	const int B_THREADS_PER_ROW = BLOCK_SIZE_N / 4;

	const int THREADS_PER_BLOCK = (BLOCK_SIZE_M / THREAD_SIZE_M) * (BLOCK_SIZE_N / THREAD_SIZE_N);

	const int A_STRIDES = THREADS_PER_BLOCK / A_THREADS_PER_ROW;
	const int B_STRIDES = THREADS_PER_BLOCK / B_THREADS_PER_ROW;

	__shared__ float A_shared[BLOCK_SIZE_M * BLOCK_SIZE_K];
	__shared__ float B_shared[BLOCK_SIZE_N * BLOCK_SIZE_K_SPARSE];
	__shared__ int B_index_shared[BLOCK_SIZE_N * BLOCK_SIZE_K_SPARSE];

	float A_reg[THREAD_SIZE_M];
	float B_reg[THREAD_SIZE_N];
	int B_reg_index;
	float C_reg[THREAD_SIZE_M][THREAD_SIZE_N] = {0};

	int tid = threadIdx.x;

	int t_N = tid % (BLOCK_SIZE_N / THREAD_SIZE_N);
	int t_M = tid / (BLOCK_SIZE_N / THREAD_SIZE_N);

	int A_BLOCK_ROW_START = tid / A_THREADS_PER_ROW;
	int B_BLOCK_ROW_START = tid / B_THREADS_PER_ROW;

	int A_BLOCK_COL_START = tid % A_THREADS_PER_ROW * 4;
	int B_BLOCK_COL_START = tid % B_THREADS_PER_ROW * 4;

	for(int K_BLOCK_START = 0, K_SPARSE_BLOCK_START = 0; K_BLOCK_START < K; K_BLOCK_START += BLOCK_SIZE_K, K_SPARSE_BLOCK_START += BLOCK_SIZE_K_SPARSE){
		float *A_global_ptr = g_vec + M_BLOCK_START * K + K_BLOCK_START;
		float *B_global_ptr = g_mat_data + K_SPARSE_BLOCK_START * N + N_BLOCK_START;
		int *B_index_global_ptr = g_mat_index + K_SPARSE_BLOCK_START * N + N_BLOCK_START;

		__syncthreads();

		#pragma unroll
		for(int i = 0; i < BLOCK_SIZE_M; i += A_STRIDES){
			*(float4 *)(A_shared + (i + A_BLOCK_ROW_START) * BLOCK_SIZE_K + A_BLOCK_COL_START) = 
				*(float4 *)(A_global_ptr + (i + A_BLOCK_ROW_START) * K + A_BLOCK_COL_START);
		}

		#pragma unroll
		for(int i = 0; i < BLOCK_SIZE_K_SPARSE; i += B_STRIDES){
			*(float4 *)(B_shared + (i + B_BLOCK_ROW_START) * BLOCK_SIZE_N + B_BLOCK_COL_START) =
				*(float4 *)(B_global_ptr + (i + B_BLOCK_ROW_START) * N + B_BLOCK_COL_START);

			*(float4 *)(B_index_shared + (i + B_BLOCK_ROW_START) * BLOCK_SIZE_N + B_BLOCK_COL_START) =
				*(float4 *)(B_index_global_ptr + (i + B_BLOCK_ROW_START) * N + B_BLOCK_COL_START);
		}

		__syncthreads();

		#pragma unroll
		for(int i = 0; i < BLOCK_SIZE_K_SPARSE;i += 1){
			#pragma unroll
			for(int k = 0; k < THREAD_SIZE_N; k += 1){
				B_reg[k] = B_shared[i * BLOCK_SIZE_N + t_N * THREAD_SIZE_N + k];
				//*(float4 *)(B_reg + k) = *(float4 *)(B_shared + i * BLOCK_SIZE_N + t_N * THREAD_SIZE_N + k);
				//*(float4 *)(B_reg_index + k) = *(float4 *)(B_index_shared + i * BLOCK_SIZE_N + t_N * THREAD_SIZE_N + k);
			}
			int bank_idx = i / LEN_OF_BANK_PER_SPARSE_BLOCK;
			B_reg_index = B_index_shared[i * BLOCK_SIZE_N + t_N * THREAD_SIZE_N] % BANK_VAL+bank_idx * BANK_VAL;
			#pragma unroll
			for(int k = 0; k < THREAD_SIZE_M; k += 1){
				A_reg[k] = A_shared[(t_M * THREAD_SIZE_M+k) * BLOCK_SIZE_K + B_reg_index];
			}
			#pragma unroll
			for(int k = 0; k < THREAD_SIZE_N; k += 1){
				#pragma unroll
				for(int j = 0; j < THREAD_SIZE_M; j += 1){
					C_reg[j][k] += B_reg[k] * A_reg[j];
				}
			}
		}
	}

	#pragma unroll
	for(int i = 0; i < THREAD_SIZE_M; i += 1){
		#pragma unroll
		for(int j = 0; j < THREAD_SIZE_N; j += 1){
			g_data[(BLOCK_SIZE_M * blockIdx.x + THREAD_SIZE_M * t_M + i) * N + BLOCK_SIZE_N * blockIdx.y + THREAD_SIZE_N * t_N + j] = C_reg[i][j];
		}
	}
}

int oneKernel_general(int w, const int h, const int vecNum, const int BLOCK_WIDTH, const int NUM_THREADS, const int VEC_WIDTH, const int minibatch) {
	// set up device
	int dev = 0;
	cudaSetDevice(dev);

	//const int w = 512*14;
	//const int h = 16384;
	//const int vecNum = 4096;
	const float sparse = float(w) / float(vecNum);

	// set up data size of vectors
	printf("Matrix size (h=%d,w=%d); Vector size %d; VEC_WIDTH: %d, BLOCK_WIDTH: %d\n", h, w, vecNum,VEC_WIDTH, BLOCK_WIDTH);

	// malloc host memory
	size_t vec_nBytes = vecNum * minibatch * sizeof(float);		// size of dense matrix
	size_t result_nBytes = h * minibatch * sizeof(float);		// size of result matrix
	size_t mat_data_nBytes = w * h * sizeof(float);				// size of sparse matrix
	size_t mat_index_nBytes = w * h * sizeof(int);			// index size same with data, csc?s

	float *vec, *mat_data, *mat_data_for_gpu, *hostRef, *gpuRef;
	int *mat_index, *mat_index_for_gpu;
	vec = (float *)malloc(vec_nBytes);
	mat_data = (float *)malloc(mat_data_nBytes);
	mat_index = (int *)malloc(mat_index_nBytes);
	mat_data_for_gpu = (float *)malloc(mat_data_nBytes);
	mat_index_for_gpu = (int *)malloc(mat_index_nBytes);
	hostRef = (float *)malloc(result_nBytes);
	gpuRef = (float *)malloc(result_nBytes);

	// initialize data at host side
	batch::initialData(vec, mat_data, mat_index, mat_data_for_gpu, mat_index_for_gpu, vecNum, h, sparse, minibatch);
	memset(hostRef, 0, result_nBytes);
	memset(gpuRef, 0, result_nBytes);

	// malloc device global memory
	float *g_vec, *g_mat_data, *g_result;
	int *g_mat_index;
	cudaMalloc((float**)&g_vec, vec_nBytes);
	cudaMalloc((float**)&g_mat_data, mat_data_nBytes);
	cudaMalloc((int**)&g_mat_index, mat_index_nBytes);
	cudaMalloc((float**)&g_result, result_nBytes);

	// transfer data from host to device
	cudaMemcpy(g_vec, vec, vec_nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(g_mat_data, mat_data_for_gpu, mat_data_nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(g_mat_index, mat_index_for_gpu, mat_index_nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(g_result, gpuRef, result_nBytes, cudaMemcpyHostToDevice);
	//printf("%d, %f\n",mat_index[0], mat_data[0]);
	// invoke kernel at host side

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//dim3 block(NUM_THREADS, next_pow_of_2_w/2/NUM_THREADS);
	//printf("%d, %d, %d\n",next_pow_of_2_w/2/NUM_THREADS*NUM_THREADS, next_pow_of_2_w/2);
	int ntimes;

	CheckSetting();

	ntimes = 100;

	int M = minibatch, N = h;
	dim3 dimBlock(int((BLOCK_SIZE_M / THREAD_SIZE_M) * (BLOCK_SIZE_N / THREAD_SIZE_N)));
	dim3 dimGrid(M / BLOCK_SIZE_M, N / BLOCK_SIZE_N);
	cudaDeviceSynchronize();

	for(int i = 0; i < ntimes; i +=1){
		//MatMul_TILE_GENERAL<<< dimGrid, dimBlock >>> (g_vec, g_mat_data, g_mat_index, g_result, M, K, K_sparse, N);
		MatMul_TILE_THREAD_GENERAL<<< dimGrid, dimBlock >>> (g_vec, g_mat_data, g_mat_index, g_result);
	}

	cudaEventRecord(start);
	//cudaEventSynchronize(start);
	//double iStart = seconds();
	for(int i = 0; i < ntimes; i += 1)
		//MatMul_TILE_GENERAL<<< dimGrid, dimBlock >>> (g_vec, g_mat_data, g_mat_index, g_result, M, K, K_sparse, N);
		MatMul_TILE_THREAD_GENERAL<<< dimGrid, dimBlock >>> (g_vec, g_mat_data, g_mat_index, g_result);
	//CUDA_CHECK(cudaGetLastError());
	cudaDeviceSynchronize();
	// record stop event on the default stream
	cudaEventRecord(stop);
	// wait until the stop event completes
	cudaEventSynchronize(stop);
	//double iElaps = seconds() - iStart;

	float time;
	cudaEventElapsedTime(&time, start, stop);
	// clean up the two events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("cal sparse %.3f; cuda: cal Time= %f msec\n", 1.0 - sparse, time/ntimes);

	//printf("Execution configuration <<<%d, %d>>>\n", h, NUM_THREADS);

	// copy kernel result back to host side
	cudaMemcpy(gpuRef, g_result, result_nBytes, cudaMemcpyDeviceToHost);
	// add vector at host side for result checks
	// batch::MVOnHost(vec, mat_data, mat_index, hostRef, w, h, vecNum, minibatch);

	// check device results
	// batch::checkResult(hostRef, gpuRef, h, minibatch);
	printf("Pass\n\n");

	// free device global memory
	cudaFree(g_vec);
	cudaFree(g_mat_data);
	cudaFree(g_mat_index);
	cudaFree(g_result);
	cudaDeviceReset();
	// free host memory
	free(vec);
	free(mat_data);
	free(mat_index);
	free(mat_data_for_gpu);
	free(mat_index_for_gpu);
	free(hostRef);
	free(gpuRef);
	return(0);
}

int main(int argc, char **argv) {	
	//const int h = 16384;
	//const int vecNum = 8192;
	const int h = N_GLOBAL;
	const int vecNum = K_GLOBAL;

	int w = int(vecNum * (1-SPARSITY));

	const int BLOCK_WIDTH = w/8;
	const int NUM_THREADS = 128;
	//const int minibatch = 8;
	const int minibatch = M_GLOBAL;

	const int VEC_WIDTH = vecNum * BLOCK_WIDTH / w;		// VEC_WIDTH = vecNum / 32;
	printf("BLOCK_WIDTH: %d, VEC_WIDTH: %d\n", BLOCK_WIDTH, VEC_WIDTH);
	//oneKernel(w, h, vecNum, BLOCK_WIDTH, NUM_THREADS, VEC_WIDTH, minibatch);
	oneKernel_general(w, h, vecNum, BLOCK_WIDTH, NUM_THREADS, VEC_WIDTH, minibatch);
	return 0;
}
