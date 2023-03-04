#include <cuda_runtime.h>
#include <stdio.h>
#include <algorithm>

/*
static inline unsigned int
next_pow_of_2(unsigned int x) {
	x -= 1;
	x |= x>>1;
	x |= x>>2;
	x |= x>>4;
	x |= x>>8;
	x |= x>>16;
	return x+1;
}
*/

const int M_GLOBAL = 1;
const int K_GLOBAL = 20480;
const int N_GLOBAL = 5120;
const int BANK_VAL = 32;
const float SPARSITY_RATIO = 0.875;

const int NUM_BANK = K_GLOBAL / BANK_VAL;
const int BLOCK_minibatch = M_GLOBAL;

namespace batch{
void checkResult(float *hostRef, float *gpuRef, const int N, const int minibatch) {
	double epsilon = 1E-4;
	bool match = 1;
	for (int batch = 0; batch < minibatch; ++batch)
		for (int i=0; i<N; i++) {
			if (abs((hostRef[i + batch * N] - gpuRef[i + batch * N])/hostRef[i + batch * N]) > epsilon) {
				match = 0;
				printf("Arrays do [NOT] match!\n");
				printf("host %5.5f gpu %5.5f at current %d\n",hostRef[i + batch * N],gpuRef[i + batch * N],i + batch * N);
				break;
			}
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

		for (int j=0; j<h; ++j){
			for (int i=0; i<w; i+=w/NUM_BANK){
				std::random_shuffle(tmp_index,tmp_index+vecNum/NUM_BANK);
				std::sort(tmp_index, tmp_index+w/NUM_BANK);
				for (int k=0; k<w/NUM_BANK; ++k){
					mat_index[i + k + j * w] = tmp_index[k]+i/sparse; // tmp_index[k] + delta(vecNum/NUM_BANK)
					mat_index_for_gpu[(i + k)*h + j] = mat_index[i + k + j * w];
					//if (j==0) printf("ID:%d Index: %d\n", i+k, mat_index[i+k]);
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

// dim3 dimBlock(NUM_THREADS);
// dim3 dimGrid(h/NUM_THREADS, w/BLOCK_WIDTH, minibatch/BLOCK_minibatch);
__global__ void MatMulOnGPU_GENERAL(float *g_vec, float *g_mat_data, int *g_mat_index, float *g_odata, int w, int h, int BLOCK_WIDTH, int NUM_THREADS, int VEC_WIDTH, const int minibatch, const int vecNum) {
	int blockxInd;
	int vecInd;
	int blockElt;  // holds the current block width
	
	// Run this only 1 time
	if ((blockIdx.y + 1) * BLOCK_WIDTH <= w){
		blockElt = BLOCK_WIDTH;
	}
	else {
		blockElt = w % BLOCK_WIDTH;
	}
	blockxInd = blockIdx.y * BLOCK_WIDTH;
	vecInd = blockIdx.y * VEC_WIDTH;
	
	unsigned int threadyInd = blockIdx.x * NUM_THREADS + threadIdx.x;
	
	//if (blockIdx.x < 1 && threadIdx.x == 0) printf("griddim.y %d\n", blockIdx.y);
	//for (unsigned int h_id = blockIdx.x; h_id < h; h_id += gridDim.x) {
	// each thread loads one element from global to shared mem
	extern __shared__ float vec_data[];
	/*
	for (int batch = 0; batch < BLOCK_minibatch; ++batch)
		for (int i = threadIdx.x; i < VEC_WIDTH; i+=NUM_THREADS) {
			vec_data[i+batch*VEC_WIDTH] = g_vec[vecInd + i + batch*vecNum];
		}
	*/

	
	#pragma unroll
	for (int batch = 0; batch < BLOCK_minibatch; ++batch){
		#pragma unroll
		for (int i = 4 * threadIdx.x; i < VEC_WIDTH; i+=4*NUM_THREADS) {
			*(float4*)(vec_data+i+batch*VEC_WIDTH) = *(float4*)(g_vec+vecInd + i + (batch+blockIdx.z * BLOCK_minibatch)*vecNum);
		}
	}
		
	__syncthreads();

	/*
	float sdata_tmp;
	for (int batch = 0; batch < minibatch; ++batch) {
		sdata_tmp = 0;
		for(int index = 0; index < blockElt; ++index) {
			sdata_tmp += g_mat_data[threadyInd + (index + blockxInd) * h] * vec_data[g_mat_index[threadyInd + (index + blockxInd) * h] - vecInd+batch*VEC_WIDTH];
			//if (blockIdx.x < 1) printf("%d:%f %f %f\n", sdata[tid],g_mat_data[i] * g_vec[g_mat_index[i]], g_mat_data[i + tid_count] * g_vec[g_mat_index[i + tid_count]]);
		}
		atomicAdd(g_odata + batch * h + threadyInd, sdata_tmp);
	}*/

	float sdata[BLOCK_minibatch] = {0};

	float data_tmp = 0;
	int index_tmp = 0;

	#pragma unroll
	for(int index = 0; index < blockElt; ++index) {
		data_tmp = g_mat_data[threadyInd + (index + blockxInd) * h];
		index_tmp = g_mat_index[threadyInd + (index + blockxInd) * h] - vecInd;

		#pragma unroll
		for(int batch = 0; batch < BLOCK_minibatch; batch += 1){
			sdata[batch] += data_tmp * vec_data[index_tmp + batch * VEC_WIDTH];
		}
		//if (blockIdx.x < 1) printf("%d:%f %f %f\n", sdata[tid],g_mat_data[i] * g_vec[g_mat_index[i]], g_mat_data[i + tid_count] * g_vec[g_mat_index[i + tid_count]]);
	}

	#pragma unroll
	for(int batch = 0; batch < BLOCK_minibatch; batch += 1){
		atomicAdd(g_odata + h * (batch + blockIdx.z * BLOCK_minibatch) + threadyInd, sdata[batch]);
	}
}

// dim3 dimBlock(NUM_THREADS);
// dim3 dimGrid(h/NUM_THREADS, w/BLOCK_WIDTH);
__global__ void MatMulOnGPU_8(float *g_vec, float *g_mat_data, int *g_mat_index, float *g_odata, int w, int h, int BLOCK_WIDTH, int NUM_THREADS, int VEC_WIDTH, const int minibatch, const int vecNum) {
	int blockxInd;
	int vecInd;
	int blockElt;  // holds the current block width
	
	// Run this only 1 time
	if ((blockIdx.y + 1) * BLOCK_WIDTH <= w){
		blockElt = BLOCK_WIDTH;
	}
	else {
		blockElt = w % BLOCK_WIDTH;
	}
	blockxInd = blockIdx.y * BLOCK_WIDTH;
	vecInd = blockIdx.y * VEC_WIDTH;
	
	unsigned int threadyInd = blockIdx.x * NUM_THREADS + threadIdx.x;
	
	//if (blockIdx.x < 1 && threadIdx.x == 0) printf("griddim.y %d\n", blockIdx.y);
	//for (unsigned int h_id = blockIdx.x; h_id < h; h_id += gridDim.x) {
	// each thread loads one element from global to shared mem
	extern __shared__ float vec_data[];
	/*
	for (int batch = 0; batch < minibatch; ++batch)
		for (int i = threadIdx.x; i < VEC_WIDTH; i+=NUM_THREADS) {
			vec_data[i+batch*VEC_WIDTH] = g_vec[vecInd + i + batch*vecNum];
		}
	*/
	for (int batch = 0; batch < minibatch; ++batch){
		for (int i = 4 * threadIdx.x; i < VEC_WIDTH; i+=4*NUM_THREADS) {
			*(float4*)(vec_data+i+batch*VEC_WIDTH) = *(float4*)(g_vec+vecInd + i + batch*vecNum);
		}
	}
		
	__syncthreads();

	/*
	float sdata_tmp;
	for (int batch = 0; batch < minibatch; ++batch) {
		sdata_tmp = 0;
		for(int index = 0; index < blockElt; ++index) {
			sdata_tmp += g_mat_data[threadyInd + (index + blockxInd) * h] * vec_data[g_mat_index[threadyInd + (index + blockxInd) * h] - vecInd+batch*VEC_WIDTH];
			//if (blockIdx.x < 1) printf("%d:%f %f %f\n", sdata[tid],g_mat_data[i] * g_vec[g_mat_index[i]], g_mat_data[i + tid_count] * g_vec[g_mat_index[i + tid_count]]);
		}
		atomicAdd(g_odata + batch * h + threadyInd, sdata_tmp);
	}*/
	float sdata_1 = 0;
	float sdata_2 = 0;
	float sdata_3 = 0;
	float sdata_4 = 0;
	float sdata_5 = 0;
	float sdata_6 = 0;
	float sdata_7 = 0;
	float sdata_8 = 0;
	float data_tmp = 0;
	int index_tmp = 0;
	for(int index = 0; index < blockElt; ++index) {
		data_tmp = g_mat_data[threadyInd + (index + blockxInd) * h];
		index_tmp = g_mat_index[threadyInd + (index + blockxInd) * h] - vecInd;
		sdata_1 += data_tmp * vec_data[index_tmp];
		sdata_2 += data_tmp * vec_data[index_tmp + 1 * VEC_WIDTH];
		sdata_3 += data_tmp * vec_data[index_tmp + 2 * VEC_WIDTH];
		sdata_4 += data_tmp * vec_data[index_tmp + 3 * VEC_WIDTH];
		sdata_5 += data_tmp * vec_data[index_tmp + 4 * VEC_WIDTH];
		sdata_6 += data_tmp * vec_data[index_tmp + 5 * VEC_WIDTH];
		sdata_7 += data_tmp * vec_data[index_tmp + 6 * VEC_WIDTH];
		sdata_8 += data_tmp * vec_data[index_tmp + 7 * VEC_WIDTH];
		//if (blockIdx.x < 1) printf("%d:%f %f %f\n", sdata[tid],g_mat_data[i] * g_vec[g_mat_index[i]], g_mat_data[i + tid_count] * g_vec[g_mat_index[i + tid_count]]);
	}
	atomicAdd(g_odata + threadyInd, sdata_1);
	atomicAdd(g_odata + h * 1 + threadyInd, sdata_2);
	atomicAdd(g_odata + h * 2 + threadyInd, sdata_3);
	atomicAdd(g_odata + h * 3 + threadyInd, sdata_4);
	atomicAdd(g_odata + h * 4 + threadyInd, sdata_5);
	atomicAdd(g_odata + h * 5 + threadyInd, sdata_6);
	atomicAdd(g_odata + h * 6 + threadyInd, sdata_7);
	atomicAdd(g_odata + h * 7 + threadyInd, sdata_8);
}

__global__ void MatMulOnGPU_4(float *g_vec, float *g_mat_data, int *g_mat_index, float *g_odata, int w, int h, int BLOCK_WIDTH, int NUM_THREADS, int VEC_WIDTH, const int minibatch, const int vecNum) {
	__shared__ int blockxInd;
	__shared__ int vecInd;
	__shared__ int blockElt;  // holds the current block width
	
	// Run this only 1 time
	if (threadIdx.x == 0) {
		if ((blockIdx.y + 1) * BLOCK_WIDTH <= w){
			blockElt = BLOCK_WIDTH;
		}
		else {
			blockElt = w % BLOCK_WIDTH;
		}
		blockxInd = blockIdx.y * BLOCK_WIDTH;
		vecInd = blockIdx.y * VEC_WIDTH;
	}
	__syncthreads();
	
	unsigned int threadyInd = blockIdx.x * NUM_THREADS + threadIdx.x;
	
	//if (blockIdx.x < 1 && threadIdx.x == 0) printf("griddim.y %d\n", blockIdx.y);
	//for (unsigned int h_id = blockIdx.x; h_id < h; h_id += gridDim.x) {
	// each thread loads one element from global to shared mem
	extern __shared__ float vec_data[];
	for (int batch = 0; batch < minibatch; ++batch)
		for (int i = threadIdx.x; i < VEC_WIDTH; i+=NUM_THREADS) {
			vec_data[i+batch*VEC_WIDTH] = g_vec[vecInd + i + batch*vecNum];
		}
	__syncthreads();

	/*
	float sdata_tmp;
	for (int batch = 0; batch < minibatch; ++batch) {
		sdata_tmp = 0;
		for(int index = 0; index < blockElt; ++index) {
			sdata_tmp += g_mat_data[threadyInd + (index + blockxInd) * h] * vec_data[g_mat_index[threadyInd + (index + blockxInd) * h] - vecInd+batch*VEC_WIDTH];
			//if (blockIdx.x < 1) printf("%d:%f %f %f\n", sdata[tid],g_mat_data[i] * g_vec[g_mat_index[i]], g_mat_data[i + tid_count] * g_vec[g_mat_index[i + tid_count]]);
		}
		atomicAdd(g_odata + batch * h + threadyInd, sdata_tmp);
	}*/
	float sdata_1 = 0;
	float sdata_2 = 0;
	float sdata_3 = 0;
	float sdata_4 = 0;
	float data_tmp = 0;
	int index_tmp = 0;
	for(int index = 0; index < blockElt; ++index) {
		data_tmp = g_mat_data[threadyInd + (index + blockxInd) * h];
		index_tmp = g_mat_index[threadyInd + (index + blockxInd) * h] - vecInd;
		sdata_1 += data_tmp * vec_data[index_tmp];
		sdata_2 += data_tmp * vec_data[index_tmp + 1 * VEC_WIDTH];
		sdata_3 += data_tmp * vec_data[index_tmp + 2 * VEC_WIDTH];
		sdata_4 += data_tmp * vec_data[index_tmp + 3 * VEC_WIDTH];
		//if (blockIdx.x < 1) printf("%d:%f %f %f\n", sdata[tid],g_mat_data[i] * g_vec[g_mat_index[i]], g_mat_data[i + tid_count] * g_vec[g_mat_index[i + tid_count]]);
	}
	atomicAdd(g_odata + threadyInd, sdata_1);
	atomicAdd(g_odata + h * 1 + threadyInd, sdata_2);
	atomicAdd(g_odata + h * 2 + threadyInd, sdata_3);
	atomicAdd(g_odata + h * 3 + threadyInd, sdata_4);
}

__global__ void MatMulOnGPU_2(float *g_vec, float *g_mat_data, int *g_mat_index, float *g_odata, int w, int h, int BLOCK_WIDTH, int NUM_THREADS, int VEC_WIDTH, const int minibatch, const int vecNum) {
	__shared__ int blockxInd;
	__shared__ int vecInd;
	__shared__ int blockElt;  // holds the current block width
	
	// Run this only 1 time
	if (threadIdx.x == 0) {
		if ((blockIdx.y + 1) * BLOCK_WIDTH <= w){
			blockElt = BLOCK_WIDTH;		// calculate the length of block
		}
		else {
			blockElt = w % BLOCK_WIDTH;
		}
		blockxInd = blockIdx.y * BLOCK_WIDTH;
		vecInd = blockIdx.y * VEC_WIDTH;
	}
	__syncthreads();
	
	unsigned int threadyInd = blockIdx.x * NUM_THREADS + threadIdx.x;
	
	//if (blockIdx.x < 1 && threadIdx.x == 0) printf("griddim.y %d\n", blockIdx.y);
	//for (unsigned int h_id = blockIdx.x; h_id < h; h_id += gridDim.x) {
	// each thread loads one element from global to shared mem
	extern __shared__ float vec_data[];
	for (int batch = 0; batch < minibatch; ++batch)
		for (int i = threadIdx.x; i < VEC_WIDTH; i+=NUM_THREADS) {
			vec_data[i+batch*VEC_WIDTH] = g_vec[vecInd + i + batch*vecNum];
		}
	__syncthreads();

	/*
	float sdata_tmp;
	for (int batch = 0; batch < minibatch; ++batch) {
		sdata_tmp = 0;
		for(int index = 0; index < blockElt; ++index) {
			sdata_tmp += g_mat_data[threadyInd + (index + blockxInd) * h] * vec_data[g_mat_index[threadyInd + (index + blockxInd) * h] - vecInd+batch*VEC_WIDTH];
			//if (blockIdx.x < 1) printf("%d:%f %f %f\n", sdata[tid],g_mat_data[i] * g_vec[g_mat_index[i]], g_mat_data[i + tid_count] * g_vec[g_mat_index[i + tid_count]]);
		}
		atomicAdd(g_odata + batch * h + threadyInd, sdata_tmp);
	}*/
	float sdata_1 = 0;
	float sdata_2 = 0;
	float data_tmp = 0;
	int index_tmp = 0;
	for(int index = 0; index < blockElt; ++index) {
		data_tmp = g_mat_data[threadyInd + (index + blockxInd) * h];
		index_tmp = g_mat_index[threadyInd + (index + blockxInd) * h] - vecInd;
		sdata_1 += data_tmp * vec_data[index_tmp];
		sdata_2 += data_tmp * vec_data[index_tmp + 1 * VEC_WIDTH];
		//if (blockIdx.x < 1) printf("%d:%f %f %f\n", sdata[tid],g_mat_data[i] * g_vec[g_mat_index[i]], g_mat_data[i + tid_count] * g_vec[g_mat_index[i + tid_count]]);
	}
	atomicAdd(g_odata + threadyInd, sdata_1);
	atomicAdd(g_odata + h * 1 + threadyInd, sdata_2);
}

__global__ void MatMulOnGPU_1(float *g_vec, float *g_mat_data, int *g_mat_index, float *g_odata, int w, int h, int BLOCK_WIDTH, int NUM_THREADS, int VEC_WIDTH, const int minibatch, const int vecNum) {
	__shared__ int blockxInd;
	__shared__ int vecInd;
	__shared__ int blockElt;  // holds the current block width
	
	// Run this only 1 time
	if (threadIdx.x == 0) {
		if ((blockIdx.y + 1) * BLOCK_WIDTH <= w){
			blockElt = BLOCK_WIDTH;
		}
		else {
			blockElt = w % BLOCK_WIDTH;
		}
		blockxInd = blockIdx.y * BLOCK_WIDTH;
		vecInd = blockIdx.y * VEC_WIDTH;
	}
	__syncthreads();
	
	unsigned int threadyInd = blockIdx.x * NUM_THREADS + threadIdx.x;
	
	//if (blockIdx.x < 1 && threadIdx.x == 0) printf("griddim.y %d\n", blockIdx.y);
	//for (unsigned int h_id = blockIdx.x; h_id < h; h_id += gridDim.x) {
	// each thread loads one element from global to shared mem
	extern __shared__ float vec_data[];
	for (int batch = 0; batch < minibatch; ++batch)
		for (int i = threadIdx.x; i < VEC_WIDTH; i+=NUM_THREADS) {
			vec_data[i+batch*VEC_WIDTH] = g_vec[vecInd + i + batch*vecNum];
		}
	__syncthreads();

	/*
	float sdata_tmp;
	for (int batch = 0; batch < minibatch; ++batch) {
		sdata_tmp = 0;
		for(int index = 0; index < blockElt; ++index) {
			sdata_tmp += g_mat_data[threadyInd + (index + blockxInd) * h] * vec_data[g_mat_index[threadyInd + (index + blockxInd) * h] - vecInd+batch*VEC_WIDTH];
			//if (blockIdx.x < 1) printf("%d:%f %f %f\n", sdata[tid],g_mat_data[i] * g_vec[g_mat_index[i]], g_mat_data[i + tid_count] * g_vec[g_mat_index[i + tid_count]]);
		}
		atomicAdd(g_odata + batch * h + threadyInd, sdata_tmp);
	}*/
	float sdata_1 = 0;
	float data_tmp = 0;
	int index_tmp = 0;
	for(int index = 0; index < blockElt; ++index) {
		data_tmp = g_mat_data[threadyInd + (index + blockxInd) * h];
		index_tmp = g_mat_index[threadyInd + (index + blockxInd) * h] - vecInd;
		sdata_1 += data_tmp * vec_data[index_tmp];
		//if (blockIdx.x < 1) printf("%d:%f %f %f\n", sdata[tid],g_mat_data[i] * g_vec[g_mat_index[i]], g_mat_data[i + tid_count] * g_vec[g_mat_index[i + tid_count]]);
	}
	atomicAdd(g_odata + threadyInd, sdata_1);
}

int oneKernel(int w, const int h, const int vecNum, const int BLOCK_WIDTH, const int NUM_THREADS, const int VEC_WIDTH, const int minibatch) {
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

	//cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
	//printf("start kernel\n\n\n\n");
	dim3 dimBlock(NUM_THREADS);
	dim3 dimGrid(h/NUM_THREADS, w/BLOCK_WIDTH);
	cudaDeviceSynchronize();
	if (minibatch == 1) {
	cudaEventRecord(start);
	//cudaEventSynchronize(start);
	//double iStart = seconds();
	MatMulOnGPU_1<<< dimGrid, dimBlock, 3 * sizeof(int) + minibatch*VEC_WIDTH*sizeof(float)>>>(g_vec, g_mat_data, g_mat_index, g_result, w, h, BLOCK_WIDTH, NUM_THREADS, VEC_WIDTH, minibatch, vecNum);
	//CUDA_CHECK(cudaGetLastError());
	cudaDeviceSynchronize();
	// record stop event on the default stream
	cudaEventRecord(stop);
	// wait until the stop event completes
	cudaEventSynchronize(stop);
	//double iElaps = seconds() - iStart;
	}
		if (minibatch == 2) {
	cudaEventRecord(start);
	//cudaEventSynchronize(start);
	//double iStart = seconds();
	MatMulOnGPU_2<<< dimGrid, dimBlock, 3 * sizeof(int) + minibatch*VEC_WIDTH*sizeof(float)>>>(g_vec, g_mat_data, g_mat_index, g_result, w, h, BLOCK_WIDTH, NUM_THREADS, VEC_WIDTH, minibatch, vecNum);
	//CUDA_CHECK(cudaGetLastError());
	cudaDeviceSynchronize();
	// record stop event on the default stream
	cudaEventRecord(stop);
	// wait until the stop event completes
	cudaEventSynchronize(stop);
	//double iElaps = seconds() - iStart;
	}
		if (minibatch == 4) {
	cudaEventRecord(start);
	//cudaEventSynchronize(start);
	//double iStart = seconds();
	MatMulOnGPU_4<<< dimGrid, dimBlock, 3 * sizeof(int) + minibatch*VEC_WIDTH*sizeof(float)>>>(g_vec, g_mat_data, g_mat_index, g_result, w, h, BLOCK_WIDTH, NUM_THREADS, VEC_WIDTH, minibatch, vecNum);
	//CUDA_CHECK(cudaGetLastError());
	cudaDeviceSynchronize();
	// record stop event on the default stream
	cudaEventRecord(stop);
	// wait until the stop event completes
	cudaEventSynchronize(stop);
	//double iElaps = seconds() - iStart;
	}
		if (minibatch == 8) {
	cudaEventRecord(start);
	//cudaEventSynchronize(start);
	//double iStart = seconds();
	MatMulOnGPU_8<<< dimGrid, dimBlock, minibatch*VEC_WIDTH*sizeof(float)>>>(g_vec, g_mat_data, g_mat_index, g_result, w, h, BLOCK_WIDTH, NUM_THREADS, VEC_WIDTH, minibatch, vecNum);
	//CUDA_CHECK(cudaGetLastError());
	cudaDeviceSynchronize();
	// record stop event on the default stream
	cudaEventRecord(stop);
	// wait until the stop event completes
	cudaEventSynchronize(stop);
	//double iElaps = seconds() - iStart;
	}
	float time;
	cudaEventElapsedTime(&time, start, stop);
	// clean up the two events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("cal sparse %.3f; cuda: cal time %f msec\n", 1.0 - sparse, time);

	//printf("Execution configuration <<<%d, %d>>>\n", h, NUM_THREADS);

	// copy kernel result back to host side
	cudaMemcpy(gpuRef, g_result, result_nBytes, cudaMemcpyDeviceToHost);
	// add vector at host side for result checks
	batch::MVOnHost(vec, mat_data, mat_index, hostRef, w, h, vecNum, minibatch);

	// check device results
	batch::checkResult(hostRef, gpuRef, h, minibatch);

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

	//cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
	//printf("start kernel\n\n\n\n");
	dim3 dimBlock(NUM_THREADS);
	dim3 dimGrid(h/NUM_THREADS, w/BLOCK_WIDTH, minibatch/BLOCK_minibatch);
	cudaDeviceSynchronize();

	int ntimes = 50;
	
	for(int i = 0; i < ntimes; i += 1){
		MatMulOnGPU_GENERAL<<< dimGrid, dimBlock, BLOCK_minibatch*VEC_WIDTH*sizeof(float)>>>(g_vec, g_mat_data, g_mat_index, g_result, w, h, BLOCK_WIDTH, NUM_THREADS, VEC_WIDTH, minibatch, vecNum);
	}
	

	cudaEventRecord(start);
	//cudaEventSynchronize(start);
	//double iStart = seconds();
	for(int i = 0; i < ntimes; i += 1){
		//memset(g_result, 0, result_nBytes);
		cudaMemset(g_result, 0, result_nBytes);
		MatMulOnGPU_GENERAL<<< dimGrid, dimBlock, BLOCK_minibatch*VEC_WIDTH*sizeof(float)>>>(g_vec, g_mat_data, g_mat_index, g_result, w, h, BLOCK_WIDTH, NUM_THREADS, VEC_WIDTH, minibatch, vecNum);
	}
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
	batch::MVOnHost(vec, mat_data, mat_index, hostRef, w, h, vecNum, minibatch);

	// check device results
	batch::checkResult(hostRef, gpuRef, h, minibatch);

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

int main() {
	float sparsity_ratio = float(SPARSITY_RATIO);
	int w = int((1-sparsity_ratio) * K_GLOBAL);
	const int h = N_GLOBAL;
	const int vecNum = K_GLOBAL;
	//const int h = 6400;
	//const int vecNum = 3072;

	const int BLOCK_WIDTH = w/NUM_BANK;
	const int NUM_THREADS = 128;
	const int minibatch = M_GLOBAL;
	//const int minibatch = 768;

	const int VEC_WIDTH = vecNum * BLOCK_WIDTH / w;		// VEC_WIDTH = vecNum / 32;
	printf("BLOCK_WIDTH: %d, VEC_WIDTH: %d\n", BLOCK_WIDTH, VEC_WIDTH);
	//oneKernel(w, h, vecNum, BLOCK_WIDTH, NUM_THREADS, VEC_WIDTH, minibatch);
	oneKernel_general(w, h, vecNum, BLOCK_WIDTH, NUM_THREADS, VEC_WIDTH, minibatch);
	return 0;
}
