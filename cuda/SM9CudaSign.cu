#include "SM9Cuda.cuh"

__global__ void SM9_Sign_Init_CUDA(int N, struct SM9_Para *para)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // 在numBlock!=(N+blockSize-1)/blockSize时使用
    for (int i = index; i < N; i += stride)
    {
        struct SM9_Para *cur_para = para + i;
        cur_para->SIGN.Z = NULL;
        cur_para->SIGN.h1 = mirvar(cur_para->mip, 0);
        cur_para->SIGN.r = mirvar(cur_para->mip, 0);
        cur_para->SIGN.h = mirvar(cur_para->mip, 0);
        cur_para->SIGN.l = mirvar(cur_para->mip, 0);
        cur_para->SIGN.tmp = mirvar(cur_para->mip, 0);
        cur_para->SIGN.zero = mirvar(cur_para->mip, 0);
        cur_para->SIGN.xS = mirvar(cur_para->mip, 0);
        cur_para->SIGN.yS = mirvar(cur_para->mip, 0);
        cur_para->SIGN.xdSA = mirvar(cur_para->mip, 0);
        cur_para->SIGN.ydSA = mirvar(cur_para->mip, 0);
        cur_para->SIGN.s = epoint_init(cur_para->mip);
        cur_para->SIGN.dSA = epoint_init(cur_para->mip);
        cur_para->SIGN.Ppubs.x.a = mirvar(cur_para->mip, 0);
        cur_para->SIGN.Ppubs.x.b = mirvar(cur_para->mip, 0);
        cur_para->SIGN.Ppubs.y.a = mirvar(cur_para->mip, 0);
        cur_para->SIGN.Ppubs.y.b = mirvar(cur_para->mip, 0);
        cur_para->SIGN.Ppubs.z.a = mirvar(cur_para->mip, 0);
        cur_para->SIGN.Ppubs.z.b = mirvar(cur_para->mip, 0);
        cur_para->SIGN.Ppubs.marker = MR_EPOINT_INFINITY;
        zzn12_init(cur_para, &(cur_para->SIGN.g));
        zzn12_init(cur_para, &(cur_para->SIGN.w));
        bytes_to_big(cur_para->mip, BNLEN, (char *)cur_para->rand, cur_para->SIGN.r);
        bytes_to_big(cur_para->mip, BNLEN, (char *)cur_para->dSA, cur_para->SIGN.xdSA);
        bytes_to_big(cur_para->mip, BNLEN, (char *)(cur_para->dSA + BNLEN), cur_para->SIGN.ydSA);
        epoint_set(cur_para->mip, cur_para->SIGN.xdSA, cur_para->SIGN.ydSA, 0, cur_para->SIGN.dSA);
        bytes128_to_ecn2(cur_para, cur_para->Ppub, &(cur_para->SIGN.Ppubs));
    }
    return;
}
__global__ void SM9_Sign_Step1_CUDA(int N, struct SM9_Para *para)
{
//    Print_Para(para);
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // 在numBlock!=(N+blockSize-1)/blockSize时使用
    for (int i = index; i < N; i += stride)
    {
        struct SM9_Para *cur_para = para + i;
        if (!ecap(cur_para, cur_para->SIGN.Ppubs, cur_para->P1, cur_para->para_t, cur_para->X, &(cur_para->SIGN.g)))
        {
            printf("SM9_MY_ECAP_12A_ERR!\n");
            return;
        }
    }
//    Print_Para(para);
    return;
}
__global__ void SM9_Sign_Step2_CUDA(int N, struct SM9_Para *para)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // 在numBlock!=(N+blockSize-1)/blockSize时使用
    for (int i = index; i < N; i += stride)
    {
        struct SM9_Para *cur_para = para + i;
        cur_para->SIGN.w = zzn12_pow(cur_para, cur_para->SIGN.g, cur_para->SIGN.r);
    }
    return;
}
__global__ void SM9_Sign_Step3_CUDA(int N, struct SM9_Para *para)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // 在numBlock!=(N+blockSize-1)/blockSize时使用
    for (int i = index; i < N; i += stride)
    {
        struct SM9_Para *cur_para = para + i;
        cur_para->SIGN.Zlen = cur_para->mlen + 32 * 12;
        cur_para->SIGN.Z = (unsigned char *)malloc(sizeof(char) * (cur_para->SIGN.Zlen + 1));
        if (cur_para->SIGN.Z == NULL)
        {
            printf("SM9_ASK_MEMORY_ERR!\n");
            return;
        }
        LinkCharZzn12(cur_para, cur_para->message, cur_para->mlen, cur_para->SIGN.w, cur_para->SIGN.Z, cur_para->SIGN.Zlen); // M||w
        cur_para->SIGN.buf = SM9_H2(cur_para, cur_para->SIGN.Z, cur_para->SIGN.Zlen, cur_para->N, cur_para->SIGN.h);
        if (cur_para->SIGN.buf != 0)
        {
            printf("SM9_H2 Wrong!\n");
            return;
        }
    }
    return;
}
__global__ void SM9_Sign_Step4_CUDA(int N, struct SM9_Para *para)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // 在numBlock!=(N+blockSize-1)/blockSize时使用
    for (int i = index; i < N; i += stride)
    {
        struct SM9_Para *cur_para = para + i;
        subtract(cur_para->mip, cur_para->SIGN.r, cur_para->SIGN.h, cur_para->SIGN.l); //(r-h)
        divide(cur_para->mip, cur_para->SIGN.l, cur_para->N, cur_para->SIGN.tmp);      //(r-h)%N
        while (mr_compare(cur_para->SIGN.l, cur_para->SIGN.zero) < 0)
            add(cur_para->mip, cur_para->SIGN.l, cur_para->N, cur_para->SIGN.l);
        if (mr_compare(cur_para->SIGN.l, cur_para->SIGN.zero) == 0)
        {
            printf("SM9_L_error!\n");
            return;
        }
    }
    return;
}
__global__ void SM9_Sign_Step5_CUDA(int N, struct SM9_Para *para)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // 在numBlock!=(N+blockSize-1)/blockSize时使用
    for (int i = index; i < N; i += stride)
    {
        struct SM9_Para *cur_para = para + i;
        ecurve_mult(cur_para->mip, cur_para->SIGN.l, cur_para->SIGN.dSA, cur_para->SIGN.s); // 多倍点乘
        epoint_get(cur_para->mip, cur_para->SIGN.s, cur_para->SIGN.xS, cur_para->SIGN.yS);
    }
    return;
}
__global__ void SM9_Sign_Finish_CUDA(int N, struct SM9_Para *para)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // 在numBlock!=(N+blockSize-1)/blockSize时使用
    for (int i = index; i < N; i += stride)
    {
        struct SM9_Para *cur_para = para + i;
        big_to_bytes(cur_para->mip, 32, cur_para->SIGN.h, (char *)cur_para->h, 1);
        big_to_bytes(cur_para->mip, 32, cur_para->SIGN.xS, (char *)cur_para->S, 1);
        big_to_bytes(cur_para->mip, 32, cur_para->SIGN.yS, (char *)(cur_para->S + 32), 1);
        free(cur_para->SIGN.Z);
    }
    
    return;
}
