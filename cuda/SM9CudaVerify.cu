#include "SM9Cuda.cuh"

__global__ void SM9_Verify_Init_CUDA(int N, struct SM9_Para *para)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // 在numBlock!=(N+blockSize-1)/blockSize时使用
    for (int i = index; i < N; i += stride)
    {
        struct SM9_Para *cur_para = para + i;
        cur_para->VERIFY.Z1 = cur_para->VERIFY.Z2 = NULL;
        cur_para->VERIFY.h = mirvar(cur_para->mip, 0);
        cur_para->VERIFY.h1 = mirvar(cur_para->mip, 0);
        cur_para->VERIFY.h2 = mirvar(cur_para->mip, 0);
        cur_para->VERIFY.xS = mirvar(cur_para->mip, 0);
        cur_para->VERIFY.yS = mirvar(cur_para->mip, 0);
        cur_para->VERIFY.P.x.a = mirvar(cur_para->mip, 0);
        cur_para->VERIFY.P.x.b = mirvar(cur_para->mip, 0);
        cur_para->VERIFY.P.y.a = mirvar(cur_para->mip, 0);
        cur_para->VERIFY.P.y.b = mirvar(cur_para->mip, 0);
        cur_para->VERIFY.P.z.a = mirvar(cur_para->mip, 0);
        cur_para->VERIFY.P.z.b = mirvar(cur_para->mip, 0);
        cur_para->VERIFY.P.marker = MR_EPOINT_INFINITY;
        cur_para->VERIFY.Ppubs.x.a = mirvar(cur_para->mip, 0);
        cur_para->VERIFY.Ppubs.x.b = mirvar(cur_para->mip, 0);
        cur_para->VERIFY.Ppubs.y.a = mirvar(cur_para->mip, 0);
        cur_para->VERIFY.Ppubs.y.b = mirvar(cur_para->mip, 0);
        cur_para->VERIFY.Ppubs.z.a = mirvar(cur_para->mip, 0);
        cur_para->VERIFY.Ppubs.z.b = mirvar(cur_para->mip, 0);
        cur_para->VERIFY.Ppubs.marker = MR_EPOINT_INFINITY;
        cur_para->VERIFY.S1 = epoint_init(cur_para->mip);
        zzn12_init(cur_para, &(cur_para->VERIFY.g)), zzn12_init(cur_para, &(cur_para->VERIFY.t));
        zzn12_init(cur_para, &(cur_para->VERIFY.u));
        zzn12_init(cur_para, &(cur_para->VERIFY.w));
        bytes_to_big(cur_para->mip, BNLEN, (char *)cur_para->h, cur_para->VERIFY.h);
        bytes_to_big(cur_para->mip, BNLEN, (char *)cur_para->S, cur_para->VERIFY.xS);
        bytes_to_big(cur_para->mip, BNLEN, (char *)(cur_para->S + BNLEN), cur_para->VERIFY.yS);
        bytes128_to_ecn2(cur_para, cur_para->Ppub, &(cur_para->VERIFY.Ppubs));
    }
    return;
}
__global__ void SM9_Verify_Step1_CUDA(int N, struct SM9_Para *para)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // 在numBlock!=(N+blockSize-1)/blockSize时使用
    for (int i = index; i < N; i += stride)
    {
        struct SM9_Para *cur_para = para + i;
        // if (Test_Range(para, h)) // 验证整数h是都在区间内
        //     return SM9_H_OUTRANGE;
    }
    return;
}
__global__ void SM9_Verify_Step2_CUDA(int N, struct SM9_Para *para)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // 在numBlock!=(N+blockSize-1)/blockSize时使用
    for (int i = index; i < N; i += stride)
    {
        struct SM9_Para *cur_para = para + i;
        epoint_set(cur_para->mip, cur_para->VERIFY.xS, cur_para->VERIFY.yS, 0, cur_para->VERIFY.S1); // 验证点是否在曲线上
        // if (Test_Point(para, S1))
        //     return SM9_S_NOT_VALID_G1;
    }
    return;
}
__global__ void SM9_Verify_Step3_CUDA(int N, struct SM9_Para *para)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // 在numBlock!=(N+blockSize-1)/blockSize时使用
    for (int i = index; i < N; i += stride)
    {
        struct SM9_Para *cur_para = para + i;
        if (!ecap(cur_para, cur_para->VERIFY.Ppubs, cur_para->P1, cur_para->para_t, cur_para->X, &(cur_para->VERIFY.g)))
        {
            printf("SM9_MY_ECAP_12A_ERR!\n");
            return;
        }
        // test if a ZZn12 element is of order q
        // if (!member(para, g, para->para_t, para->X))
        //     return SM9_MEMBER_ERR;
    }
    return;
}
__global__ void SM9_Verify_Step4_CUDA(int N, struct SM9_Para *para)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // 在numBlock!=(N+blockSize-1)/blockSize时使用
    for (int i = index; i < N; i += stride)
    {
        struct SM9_Para *cur_para = para + i;
        cur_para->VERIFY.t = zzn12_pow(cur_para, cur_para->VERIFY.g, cur_para->VERIFY.h);
    }
    return;
}
__global__ void SM9_Verify_Step5_CUDA(int N, struct SM9_Para *para)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // 在numBlock!=(N+blockSize-1)/blockSize时使用
    for (int i = index; i < N; i += stride)
    {
        struct SM9_Para *cur_para = para + i;
        cur_para->VERIFY.Zlen1 = cuda_strlen((const char *)cur_para->IDA) + 1;
        cur_para->VERIFY.Z1 = (unsigned char *)malloc(sizeof(char) * (cur_para->VERIFY.Zlen1 + 1));
        if (cur_para->VERIFY.Z1 == NULL)
        {
            printf("SM9_ASK_MEMORY_ERR!\n");
            return;
        }
        memcpy(cur_para->VERIFY.Z1, cur_para->IDA, cuda_strlen((const char *)cur_para->IDA));
        memcpy(cur_para->VERIFY.Z1 + cuda_strlen((const char *)cur_para->IDA), cur_para->hid, 1);
        cur_para->VERIFY.buf = SM9_H1(cur_para, cur_para->VERIFY.Z1, cur_para->VERIFY.Zlen1, cur_para->N, cur_para->VERIFY.h1);
        if (cur_para->VERIFY.buf != 0)
        {
            printf("SM9_H1 Wrong!\n");
            return;
        }
    }
    return;
}
__global__ void SM9_Verify_Step6_CUDA(int N, struct SM9_Para *para)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // 在numBlock!=(N+blockSize-1)/blockSize时使用
    for (int i = index; i < N; i += stride)
    {
        struct SM9_Para *cur_para = para + i;
        ecn2_copy(&(cur_para->P2), &(cur_para->VERIFY.P));
        ecn2_mul(cur_para->mip, cur_para->VERIFY.h1, &(cur_para->VERIFY.P));
        ecn2_add(cur_para->mip, &(cur_para->VERIFY.Ppubs), &(cur_para->VERIFY.P));
    }
    return;
}
__global__ void SM9_Verify_Step7_CUDA(int N, struct SM9_Para *para)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // 在numBlock!=(N+blockSize-1)/blockSize时使用
    for (int i = index; i < N; i += stride)
    {
        struct SM9_Para *cur_para = para + i;
        if (!ecap(cur_para, cur_para->VERIFY.P, cur_para->VERIFY.S1, cur_para->para_t, cur_para->X, &(cur_para->VERIFY.u)))
        {
            printf("SM9_MY_ECAP_12A_ERR!\n");
            return;
        }
        // test if a ZZn12 element is of order q
        // if (!member(para, u, para->para_t, para->X))
        //     return SM9_MEMBER_ERR;
    }
    return;
}
__global__ void SM9_Verify_Step8_CUDA(int N, struct SM9_Para *para)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // 在numBlock!=(N+blockSize-1)/blockSize时使用
    for (int i = index; i < N; i += stride)
    {
        struct SM9_Para *cur_para = para + i;
        zzn12_mul(cur_para, cur_para->VERIFY.u, cur_para->VERIFY.t, &(cur_para->VERIFY.w));
    }
    return;
}
__global__ void SM9_Verify_Step9_CUDA(int N, struct SM9_Para *para)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // 在numBlock!=(N+blockSize-1)/blockSize时使用
    for (int i = index; i < N; i += stride)
    {
        struct SM9_Para *cur_para = para + i;
        cur_para->VERIFY.Zlen2 = cur_para->mlen + 32 * 12;
        cur_para->VERIFY.Z2 = (unsigned char *)malloc(sizeof(char) * (cur_para->VERIFY.Zlen2 + 1));
        if (cur_para->VERIFY.Z2 == NULL)
        {
            printf("SM9_ASK_MEMORY_ERR!\n");
            return;
        }
        LinkCharZzn12(cur_para, cur_para->message, cur_para->mlen, cur_para->VERIFY.w, cur_para->VERIFY.Z2, cur_para->VERIFY.Zlen2);
        cur_para->VERIFY.buf = SM9_H2(cur_para, cur_para->VERIFY.Z2, cur_para->VERIFY.Zlen2, cur_para->N, cur_para->VERIFY.h2);
        if (cur_para->VERIFY.buf != 0)
        {
            printf("SM9_H2 Wrong!\n");
            return;
        }
    }
    return;
}
__global__ void SM9_Verify_Finish_CUDA(int N, struct SM9_Para *para)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // 在numBlock!=(N+blockSize-1)/blockSize时使用
    for (int i = index; i < N; i += stride)
    {
        struct SM9_Para *cur_para = para + i;
        free(cur_para->VERIFY.Z1);
        free(cur_para->VERIFY.Z2);
        //    printf("\n签名验证结果：\n");
        if (mr_compare(cur_para->VERIFY.h2, cur_para->VERIFY.h) != 0)
        {
            printf("SM9_DATA_MEMCMP_ERR!\n");
            return;
        }
        // else
        //     printf("h 等于 h2，验证成功！\n\n");
    }
//    Print_Para(para);
    return;
}

__global__ void SM9_Verify_CUDA(int N, struct SM9_Para *para)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // 在numBlock!=(N+blockSize-1)/blockSize时使用

    // printf("\n SM9签名验证 开始 - %d \n", index);

    for (int i = index; i < N; i += stride)
    {
        struct SM9_Para *cur_para = para + i;
        cur_para->tmp = SM9_Verify(cur_para, cur_para->h, cur_para->S, cur_para->hid, cur_para->IDA, cur_para->message, cur_para->mlen, cur_para->Ppub);
        if (cur_para->tmp != 0)
            return;
    }

    // printf("\n SM9签名验证 结束 - %d \n", index);
    return;
}
