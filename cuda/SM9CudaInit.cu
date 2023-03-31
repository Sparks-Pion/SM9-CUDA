#include "SM9Cuda.cuh"

__global__ void SM9_Init_CUDA(int N, struct SM9_Para *para)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // 在numBlock!=(N+blockSize-1)/blockSize时使用

    for (int i = index; i < N; i += stride)
    {
        struct SM9_Para *cur_para = para + i;
        cur_para->mlen = cuda_strlen((const char *)cur_para->message);
        cur_para->tmp = SM9_Init(cur_para);
        if (cur_para->tmp != 0)
        {
            printf("SM9_Init Wrong!\n");
            return;
        }
        cur_para->ks = mirvar(cur_para->mip, 0);
        bytes_to_big(cur_para->mip, 32, (const char *)cur_para->dA, cur_para->ks);
    }

    return;
}
__global__ void SM9_GenerateSignKey_CUDA(int N, struct SM9_Para *para)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // 在numBlock!=(N+blockSize-1)/blockSize时使用

    // printf("\n SM9 密钥生成 开始 - %d \n", index);

    for (int i = index; i < N; i += stride)
    {
        struct SM9_Para *cur_para = para + i;
        cur_para->tmp = SM9_GenerateSignKey(cur_para, cur_para->hid, cur_para->IDA, cuda_strlen((const char *)cur_para->IDA), cur_para->ks, cur_para->Ppub, cur_para->dSA);
        if (cur_para->tmp != 0)
        {
            printf("SM9_GenerateSignKey Wrong!\n");
            return;
        }
        if (cuda_memcmp(cur_para->Ppub, cur_para->std_Ppub, 128) != 0)
        {
            printf("SM9_GEPUB_ERR!\n");
            return;
        }
        if (cuda_memcmp(cur_para->dSA, cur_para->std_dSA, 64) != 0)
        {
            printf("SM9_GEPRI_ERR!\n");
            return;
        }
    }

    // printf("\n SM9 密钥生成 结束 - %d \n", index);

    // printf("Miracl_GPU:%p\n",para->mip);
    // printf("Miracl_active_GPU:%d\n", para->mip->active);
    // printf("Miracl_IOBASE_GPU:%d\n", para->mip->IOBASE);
    // printf("Miracl_user_GPU:%p\n", para->mip->user);
    // printf("Miracl_nib_CPU:%d\n", para->mip->nib);

    // for (int i = 0; i < sizeof(struct SM9_Para); i++)
    // {
    //     printf("%02X ", ((unsigned char *)para)[i]);
    // }
    // printf("\n");

    // printf("P1X_len_GPU:%d\n", para->P1->X->len);

    // Print_Para(para);
    return;
}