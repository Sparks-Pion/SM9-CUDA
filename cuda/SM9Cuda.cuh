#pragma once

#include "../sm9/SM9.cuh"
#include <stdio.h>

#define CUDA_TEST_CNT 1
#define BLOCK_SIZE 1
#define CUDA_HEAP_SIZE 1 * 1ull * 8388608ull // 显存堆大小(Byte)

// 实现Para结构体的cudaMemcpyKind拷贝
int ParaMemcpy2Host(struct SM9_Para *dst, struct SM9_Para *src);
int ParaMemcpy2Device(struct SM9_Para *dst, struct SM9_Para *src);

__global__ void SM9_Init_CUDA(int N, struct SM9_Para *para);
__global__ void SM9_GenerateSignKey_CUDA(int N, struct SM9_Para *para);

/*SM9CudaSign.cu*/
__global__ void SM9_Sign_Init_CUDA(int N, struct SM9_Para *para);
// Step1:g = e(P1, Ppub-s)
__global__ void SM9_Sign_Step1_CUDA(int N, struct SM9_Para *para);
// Step2:calculate w=g(r)
__global__ void SM9_Sign_Step2_CUDA(int N, struct SM9_Para *para);
// Step3:calculate h=H2(M||w,N)
__global__ void SM9_Sign_Step3_CUDA(int N, struct SM9_Para *para);
// Step4:l=(r-h)mod N
__global__ void SM9_Sign_Step4_CUDA(int N, struct SM9_Para *para);
// Step5:S=[l]dSA=(xS,yS)
__global__ void SM9_Sign_Step5_CUDA(int N, struct SM9_Para *para);
__global__ void SM9_Sign_Finish_CUDA(int N, struct SM9_Para *para);

/*SM9CudaVerify.cu*/
__global__ void SM9_Verify_Init_CUDA(int N, struct SM9_Para *para);
// Step 1:test if h in the rangge [1,N-1]
__global__ void SM9_Verify_Step1_CUDA(int N, struct SM9_Para *para);
// Step 2:test if S is on G1
__global__ void SM9_Verify_Step2_CUDA(int N, struct SM9_Para *para);
// Step3:g = e(P1, Ppub-s)
__global__ void SM9_Verify_Step3_CUDA(int N, struct SM9_Para *para);
// Step4:calculate t=g^h
__global__ void SM9_Verify_Step4_CUDA(int N, struct SM9_Para *para);
// Step5:calculate h1=H1(IDA||hid,N)
__global__ void SM9_Verify_Step5_CUDA(int N, struct SM9_Para *para);
// Step6:P=[h1]P2+Ppubs
__global__ void SM9_Verify_Step6_CUDA(int N, struct SM9_Para *para);
// Step7:u=e(S1,P)
__global__ void SM9_Verify_Step7_CUDA(int N, struct SM9_Para *para);
// Step8:w=u*t
__global__ void SM9_Verify_Step8_CUDA(int N, struct SM9_Para *para);
// Step9:h2=H2(M||w,N)
__global__ void SM9_Verify_Step9_CUDA(int N, struct SM9_Para *para);
__global__ void SM9_Verify_Finish_CUDA(int N, struct SM9_Para *para);
__global__ void SM9_Verify_CUDA(int N, struct SM9_Para *para);

float SM9TEST_INIT_CUDA(int N, int numBlock, int blockSize, struct SM9_Para *gpu_para);
float SM9TEST_SIGN_CUDA(int N, int numBlock, int blockSize, struct SM9_Para *gpu_para);
float SM9TEST_VERIFY_CUDA(int N, int numBlock, int blockSize, struct SM9_Para *gpu_para);
int SM9TEST_CUDA(int N, int blockSize, struct SM9_Para *para);

__device__ __host__ void Print_String(unsigned char *tmp, size_t size);
__device__ __host__ void Print_Miracl(miracl *mip);
__device__ __host__ void Print_Big(big tmp);
__device__ __host__ void Print_Zzn2(zzn2 *tmp);
__device__ __host__ void Print_Zzn4(zzn4 *tmp);
__device__ __host__ void Print_Zzn12(zzn12 *tmp);
__device__ __host__ void Print_Ecn2(ecn2 *tmp);
__device__ __host__ void Print_Epoint(epoint *tmp);
__device__ __host__ void Print_Para(struct SM9_Para *para);