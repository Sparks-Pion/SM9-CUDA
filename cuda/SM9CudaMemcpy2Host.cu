#include "SM9Cuda.cuh"

__global__ void GPU2CPUMemcpy(void *gpu, void *src, size_t size)
{
    memcpy(gpu, src, size);
}

int NewMallocAtHost(void **ptr, size_t size)
{
    // cudaMalloc((void **)ptr, size);
    *ptr = malloc(size);
    return 0;
}

int BigMemcpy2Host(big dst, big src) // src from gpu
{
    // 似乎host无法对device上cudaMalloc的地址进行cudaMemcpyDeviceToHost
    // 调用global函数把device上cudaMalloc的地址中的内容复制到host上cudaMalloc的地址里面
    void *gpu;
    cudaMalloc((void **)&gpu, sizeof(bigtype));
    GPU2CPUMemcpy<<<1, 1>>>(gpu, src, sizeof(bigtype));
    cudaDeviceSynchronize();
    src = (big)gpu;

    void *buffer;
    buffer = malloc(sizeof(bigtype));
    cudaMemcpy(buffer, src, sizeof(bigtype), cudaMemcpyDeviceToHost);
    src = (big)buffer;                 // 重制src，因为host上没办法直接访问cuda上的内容
    memcpy(dst, src, sizeof(bigtype)); // cudaMemcpy(dst, src, sizeof(bigtype), cudaMemcpyDeviceToHost);
    // printf("len_GPU:%d\n", src->len);
    if (src->len != 0)
    {
        NewMallocAtHost((void **)&(dst->w), (src->len & 0xfffff) * sizeof(mr_small));
        // src->w依旧是在device中cudaMalloc的，需要GPU2CPUMemcpy//cudaMemcpy(dst->w, src->w, (src->len) * sizeof(mr_small), cudaMemcpyDeviceToHost);
        void *gpu;
        cudaMalloc((void **)&gpu, (src->len & 0xfffff) * sizeof(mr_small));
        GPU2CPUMemcpy<<<1, 1>>>(gpu, src->w, (src->len & 0xfffff) * sizeof(mr_small));
        cudaDeviceSynchronize();
        cudaMemcpy(dst->w, (mr_small *)gpu, (src->len & 0xfffff) * sizeof(mr_small), cudaMemcpyDeviceToHost);
    }
    free(buffer);
    return 0;
}

int EpointMemcpy2Host(epoint *dst, epoint *src) // src from gpu
{
    // 似乎host无法对device上cudaMalloc的地址进行cudaMemcpyDeviceToHost
    // 调用global函数把device上cudaMalloc的地址中的内容复制到host上cudaMalloc的地址里面
    void *gpu;
    cudaMalloc((void **)&gpu, sizeof(epoint));
    GPU2CPUMemcpy<<<1, 1>>>(gpu, src, sizeof(epoint));
    cudaDeviceSynchronize();
    src = (epoint *)gpu;

    void *buffer;
    buffer = malloc(sizeof(epoint));
    cudaMemcpy(buffer, src, sizeof(epoint), cudaMemcpyDeviceToHost);
    src = (epoint *)buffer;           // 重制src，因为host上没办法直接访问cuda上的内容
    memcpy(dst, src, sizeof(epoint)); // cudaMemcpy(dst, src, sizeof(epoint), cudaMemcpyDeviceToHost);
    if (src->X != NULL)
    {
        NewMallocAtHost((void **)&(dst->X), sizeof(bigtype));
        BigMemcpy2Host(dst->X, src->X);
    }
    if (src->Y != NULL)
    {
        NewMallocAtHost((void **)&(dst->Y), sizeof(bigtype));
        BigMemcpy2Host(dst->Y, src->Y);
    }
    if (src->Z != NULL)
    {
        NewMallocAtHost((void **)&(dst->Z), sizeof(bigtype));
        BigMemcpy2Host(dst->Z, src->Z);
    }
    free(buffer);
    return 0;
}

int Zzn2Memcpy2Host(zzn2 *dst, zzn2 *src) // src from cpu
{
    memcpy(dst, src, sizeof(zzn2));

    if (src->a != NULL)
    {
        NewMallocAtHost((void **)&(dst->a), sizeof(bigtype));
        BigMemcpy2Host(dst->a, src->a);
    }
    if (src->b != NULL)
    {
        NewMallocAtHost((void **)&(dst->b), sizeof(bigtype));
        BigMemcpy2Host(dst->b, src->b);
    }
    return 0;
}

int Zzn4Memcpy2Host(zzn4 *dst, zzn4 *src) // src from cpu
{
    memcpy(dst, src, sizeof(zzn4));

    Zzn2Memcpy2Host(&(dst->a), &(src->a));
    Zzn2Memcpy2Host(&(dst->b), &(src->b));
    return 0;
}

int Zzn12Memcpy2Host(zzn12 *dst, zzn12 *src) // src from cpu
{
    memcpy(dst, src, sizeof(zzn12));

    Zzn4Memcpy2Host(&(dst->a), &(src->a));
    Zzn4Memcpy2Host(&(dst->b), &(src->b));
    Zzn4Memcpy2Host(&(dst->c), &(src->c));
    return 0;
}

int Ecn2Memcpy2Host(ecn2 *dst, ecn2 *src) // src from cpu
{
    memcpy(dst, src, sizeof(ecn2));

    Zzn2Memcpy2Host(&(dst->x), &(src->x));
    Zzn2Memcpy2Host(&(dst->y), &(src->y));
    Zzn2Memcpy2Host(&(dst->z), &(src->z));
    return 0;
}

int MiraclMemcpy2Host(miracl *dst, miracl *src) // src from gpu
{
    // printf("Miracl_CPU:%p\n", src);
    // 似乎host无法对device上cudaMalloc的地址进行cudaMemcpyDeviceToHost
    // 调用global函数把device上cudaMalloc的地址中的内容复制到host上cudaMalloc的地址里面
    void *gpu;
    cudaMalloc((void **)&gpu, sizeof(miracl));
    GPU2CPUMemcpy<<<1, 1>>>(gpu, src, sizeof(miracl));
    cudaDeviceSynchronize();
    src = (miracl *)gpu;

    void *buffer;
    buffer = malloc(sizeof(miracl));
    cudaMemcpy(buffer, src, sizeof(miracl), cudaMemcpyDeviceToHost);
    // printf("Miracl_active_CPU:%d\n", ((miracl *)buffer)->active);
    // printf("Miracl_IOBASE_CPU:%d\n", ((miracl *)buffer)->IOBASE);
    // printf("Miracl_user_CPU:%p\n", ((miracl *)buffer)->user);
    // printf("Miracl_nib_CPU:%d\n", ((miracl *)buffer)->nib);
    src = (miracl *)buffer;
    memcpy(dst, src, sizeof(miracl)); // cudaMemcpy(dst, src, sizeof(miracl), cudaMemcpyDeviceToHost);
    // ????? BOOL(*user) (void); /* pointer to user supplied function */
    if (src->modulus != NULL)
    {
        NewMallocAtHost((void **)&(dst->modulus), sizeof(bigtype));
        BigMemcpy2Host(dst->modulus, src->modulus);
    }
    if (src->pR != NULL)
    {
        NewMallocAtHost((void **)&(dst->pR), sizeof(bigtype));
        BigMemcpy2Host(dst->pR, src->pR);
    }
//    if (src->prime != NULL)
//    {
//        NewMallocAtHost((void **)&(dst->prime), sizeof(mr_utype));
//        cudaMemcpy(dst->prime, src->prime, sizeof(mr_utype), cudaMemcpyDeviceToHost);
//    }
//    if (src->cr != NULL)
//    {
//        NewMallocAtHost((void **)&(dst->cr), sizeof(mr_utype));
//        cudaMemcpy(dst->cr, src->cr, sizeof(mr_utype), cudaMemcpyDeviceToHost);
//    }
//    if (src->inverse != NULL)
//    {
//        NewMallocAtHost((void **)&(dst->inverse), sizeof(mr_utype));
//        cudaMemcpy(dst->inverse, src->inverse, sizeof(mr_utype), cudaMemcpyDeviceToHost);
//    }
//    if (src->roots != NULL)
//    {
//        NewMallocAtHost((void **)&(dst->roots), sizeof(mr_utype *));
//        cudaMemcpy(dst->roots, src->roots, sizeof(mr_utype *), cudaMemcpyDeviceToHost);
//        mr_utype *tmp;
//        cudaMemcpy(&tmp, src->roots, sizeof(mr_utype *), cudaMemcpyDeviceToHost);
//        if (tmp != NULL)
//        {
//            NewMallocAtHost((void **)&(*(dst->roots)), sizeof(mr_utype));
//            cudaMemcpy(*(dst->roots), tmp, sizeof(mr_utype), cudaMemcpyDeviceToHost);
//        }
//        free(tmp);
//    }
//    { // small_chinese chin
//        if (src->chin.C != NULL)
//        {
//            NewMallocAtHost((void **)&(dst->chin.C), sizeof(mr_utype));
//            cudaMemcpy(dst->chin.C, src->chin.C, sizeof(mr_utype), cudaMemcpyDeviceToHost);
//        }
//        if (src->chin.V != NULL)
//        {
//            NewMallocAtHost((void **)&(dst->chin.V), sizeof(mr_utype));
//            cudaMemcpy(dst->chin.V, src->chin.V, sizeof(mr_utype), cudaMemcpyDeviceToHost);
//        }
//        if (src->chin.M != NULL)
//        {
//            NewMallocAtHost((void **)&(dst->chin.M), sizeof(mr_utype));
//            cudaMemcpy(dst->chin.M, src->chin.M, sizeof(mr_utype), cudaMemcpyDeviceToHost);
//        }
//    }
//    if (src->s1 != NULL)
//    {
//        NewMallocAtHost((void **)&(dst->s1), sizeof(mr_utype *));
//        cudaMemcpy(dst->s1, src->s1, sizeof(mr_utype *), cudaMemcpyDeviceToHost);
//        mr_utype *tmp;
//        cudaMemcpy(&tmp, src->s1, sizeof(mr_utype *), cudaMemcpyDeviceToHost);
//        if (tmp != NULL)
//        {
//            NewMallocAtHost((void **)&(*(dst->s1)), sizeof(mr_utype));
//            cudaMemcpy(*(dst->s1), tmp, sizeof(mr_utype), cudaMemcpyDeviceToHost);
//        }
//        free(tmp);
//    }
//    if (src->s2 != NULL)
//    {
//        NewMallocAtHost((void **)&(dst->s2), sizeof(mr_utype *));
//        cudaMemcpy(dst->s2, src->s2, sizeof(mr_utype *), cudaMemcpyDeviceToHost);
//        mr_utype *tmp;
//        cudaMemcpy(&tmp, src->s2, sizeof(mr_utype *), cudaMemcpyDeviceToHost);
//        if (tmp != NULL)
//        {
//            NewMallocAtHost((void **)&(*(dst->s2)), sizeof(mr_utype));
//            cudaMemcpy(*(dst->s2), tmp, sizeof(mr_utype), cudaMemcpyDeviceToHost);
//        }
//        free(tmp);
//    }
//    if (src->t != NULL)
//    {
//        NewMallocAtHost((void **)&(dst->t), sizeof(mr_utype *));
//        cudaMemcpy(dst->t, src->t, sizeof(mr_utype *), cudaMemcpyDeviceToHost);
//        mr_utype *tmp;
//        cudaMemcpy(&tmp, src->t, sizeof(mr_utype *), cudaMemcpyDeviceToHost);
//        if (tmp != NULL)
//        {
//            NewMallocAtHost((void **)&(*(dst->t)), sizeof(mr_utype));
//            cudaMemcpy(*(dst->t), tmp, sizeof(mr_utype), cudaMemcpyDeviceToHost);
//        }
//        free(tmp);
//    }
//    if (src->wa != NULL)
//    {
//        NewMallocAtHost((void **)&(dst->wa), sizeof(mr_utype));
//        cudaMemcpy(dst->wa, src->wa, sizeof(mr_utype), cudaMemcpyDeviceToHost);
//    }
//    if (src->wb != NULL)
//    {
//        NewMallocAtHost((void **)&(dst->wb), sizeof(mr_utype));
//        cudaMemcpy(dst->wb, src->wb, sizeof(mr_utype), cudaMemcpyDeviceToHost);
//    }
//    if (src->wc != NULL)
//    {
//        NewMallocAtHost((void **)&(dst->wc), sizeof(mr_utype));
//        cudaMemcpy(dst->wc, src->wc, sizeof(mr_utype), cudaMemcpyDeviceToHost);
//    }
    if (src->w0 != NULL)
    {
        NewMallocAtHost((void **)&(dst->w0), sizeof(bigtype));
        BigMemcpy2Host(dst->w0, src->w0);
    }
    if (src->w1 != NULL)
    {
        NewMallocAtHost((void **)&(dst->w1), sizeof(bigtype));
        BigMemcpy2Host(dst->w1, src->w1);
    }
    if (src->w2 != NULL)
    {
        NewMallocAtHost((void **)&(dst->w2), sizeof(bigtype));
        BigMemcpy2Host(dst->w2, src->w2);
    }
    if (src->w3 != NULL)
    {
        NewMallocAtHost((void **)&(dst->w3), sizeof(bigtype));
        BigMemcpy2Host(dst->w3, src->w3);
    }
    if (src->w4 != NULL)
    {
        NewMallocAtHost((void **)&(dst->w4), sizeof(bigtype));
        BigMemcpy2Host(dst->w4, src->w4);
    }
    if (src->w5 != NULL)
    {
        NewMallocAtHost((void **)&(dst->w5), sizeof(bigtype));
        BigMemcpy2Host(dst->w5, src->w5);
    }
    if (src->w6 != NULL)
    {
        NewMallocAtHost((void **)&(dst->w6), sizeof(bigtype));
        BigMemcpy2Host(dst->w6, src->w6);
    }
    if (src->w7 != NULL)
    {
        NewMallocAtHost((void **)&(dst->w7), sizeof(bigtype));
        BigMemcpy2Host(dst->w7, src->w7);
    }
    if (src->w8 != NULL)
    {
        NewMallocAtHost((void **)&(dst->w8), sizeof(bigtype));
        BigMemcpy2Host(dst->w8, src->w8);
    }
    if (src->w9 != NULL)
    {
        NewMallocAtHost((void **)&(dst->w9), sizeof(bigtype));
        BigMemcpy2Host(dst->w9, src->w9);
    }
    if (src->w10 != NULL)
    {
        NewMallocAtHost((void **)&(dst->w10), sizeof(bigtype));
        BigMemcpy2Host(dst->w10, src->w10);
    }
    if (src->w11 != NULL)
    {
        NewMallocAtHost((void **)&(dst->w11), sizeof(bigtype));
        BigMemcpy2Host(dst->w11, src->w11);
    }
    if (src->w12 != NULL)
    {
        NewMallocAtHost((void **)&(dst->w12), sizeof(bigtype));
        BigMemcpy2Host(dst->w12, src->w12);
    }
    if (src->w13 != NULL)
    {
        NewMallocAtHost((void **)&(dst->w13), sizeof(bigtype));
        BigMemcpy2Host(dst->w13, src->w13);
    }
    if (src->w14 != NULL)
    {
        NewMallocAtHost((void **)&(dst->w14), sizeof(bigtype));
        BigMemcpy2Host(dst->w14, src->w14);
    }
    if (src->w15 != NULL)
    {
        NewMallocAtHost((void **)&(dst->w15), sizeof(bigtype));
        BigMemcpy2Host(dst->w15, src->w15);
    }
    if (src->sru != NULL)
    {
        NewMallocAtHost((void **)&(dst->sru), sizeof(bigtype));
        BigMemcpy2Host(dst->sru, src->sru);
    }
    if (src->one != NULL)
    {
        NewMallocAtHost((void **)&(dst->one), sizeof(bigtype));
        BigMemcpy2Host(dst->one, src->one);
    }
    if (src->A != NULL)
    {
        NewMallocAtHost((void **)&(dst->A), sizeof(bigtype));
        BigMemcpy2Host(dst->A, src->A);
    }
    if (src->B != NULL)
    {
        NewMallocAtHost((void **)&(dst->B), sizeof(bigtype));
        BigMemcpy2Host(dst->B, src->B);
    }
    if (src->PRIMES != NULL)
    {
        NewMallocAtHost((void **)&(dst->PRIMES), sizeof(src->PRIMES));
        cudaMemcpy(dst->PRIMES, src->PRIMES, sizeof(src->PRIMES), cudaMemcpyDeviceToHost); // sizeof(src->PRIMES)?
    }
//    if (src->IOBUFF != NULL)
//    {
//        // NewMallocAtHost((void **)&(dst->IOBUFF), sizeof(src->IOBUFF));
//        // cudaMemcpy(dst->IOBUFF, src->IOBUFF, sizeof(src->IOBUFF), cudaMemcpyDeviceToHost); // sizeof(src->IOBUFF)?
//
//        NewMallocAtHost((void **)&(dst->IOBUFF), 5 * sizeof(char)); // magic_number
//        void *gpu;
//        cudaMalloc((void **)&gpu, 5 * sizeof(char));
//        GPU2CPUMemcpy<<<1, 1>>>(gpu, src->IOBUFF, 5 * sizeof(char));
//        cudaDeviceSynchronize();
//        cudaMemcpy(dst->IOBUFF, gpu, 5 * sizeof(char), cudaMemcpyDeviceToHost);
//    }
    if (src->pi != NULL)
    {
        NewMallocAtHost((void **)&(dst->pi), sizeof(bigtype));
        BigMemcpy2Host(dst->pi, src->pi);
    }
//    if (src->workspace != NULL)
//    {
//        // NewMallocAtHost((void **)&(dst->workspace), sizeof(src->workspace));
//        // cudaMemcpy(dst->workspace, src->workspace, sizeof(src->workspace), cudaMemcpyDeviceToHost); // sizeof(src->workspace)?
//
//        NewMallocAtHost((void **)&(dst->workspace), 5 * sizeof(char)); // magic_number
//        void *gpu;
//        cudaMalloc((void **)&gpu, 5 * sizeof(char));
//        GPU2CPUMemcpy<<<1, 1>>>(gpu, src->workspace, 5 * sizeof(char));
//        cudaDeviceSynchronize();
//        cudaMemcpy(dst->workspace, gpu, 5 * sizeof(char), cudaMemcpyDeviceToHost);
//    }
    free(buffer);
    return 0;
}

int ParaMemcpy2Host(struct SM9_Para *dst, struct SM9_Para *src)
{
    // 似乎host无法对device上cudaMalloc的地址进行cudaMemcpyDeviceToHost
    // 调用global函数把device上cudaMalloc的地址中的内容复制到host上cudaMalloc的地址里面
    void *gpu;
    cudaMalloc((void **)&gpu, sizeof(struct SM9_Para));
    GPU2CPUMemcpy<<<1, 1>>>(gpu, src, sizeof(struct SM9_Para));
    cudaDeviceSynchronize();
    src = (struct SM9_Para *)gpu;

    // 重制src，因为host上没办法直接访问cuda上的内容
    void *buffer;
    buffer = malloc(sizeof(struct SM9_Para));
    cudaMemcpy(buffer, src, sizeof(struct SM9_Para), cudaMemcpyDeviceToHost);
    src = (struct SM9_Para *)buffer;
    memcpy(dst, src, sizeof(struct SM9_Para)); // cudaMemcpy(dst, src, sizeof(struct SM9_Para), cudaMemcpyDeviceToHost);
    if (src->P1 != NULL)
    {
        NewMallocAtHost((void **)&(dst->P1), sizeof(epoint));
        EpointMemcpy2Host(dst->P1, src->P1);
    }
    { // ecn2 P2
        Ecn2Memcpy2Host(&(dst->P2), &(src->P2));
    }
    if (src->N != NULL)
    {
        NewMallocAtHost((void **)&(dst->N), sizeof(bigtype));
        BigMemcpy2Host(dst->N, src->N);
    }
    if (src->para_a != NULL)
    {
        NewMallocAtHost((void **)&(dst->para_a), sizeof(bigtype));
        BigMemcpy2Host(dst->para_a, src->para_a);
    }
    if (src->para_b != NULL)
    {
        NewMallocAtHost((void **)&(dst->para_b), sizeof(bigtype));
        BigMemcpy2Host(dst->para_b, src->para_b);
    }
    if (src->para_t != NULL)
    {
        NewMallocAtHost((void **)&(dst->para_t), sizeof(bigtype));
        BigMemcpy2Host(dst->para_t, src->para_t);
    }
    if (src->para_q != NULL)
    {
        NewMallocAtHost((void **)&(dst->para_q), sizeof(bigtype));
        BigMemcpy2Host(dst->para_q, src->para_q);
    }
    { // zzn2 X
        Zzn2Memcpy2Host(&(dst->X), &(src->X));
    }
    if (src->mip != NULL)
    {
        NewMallocAtHost((void **)&(dst->mip), sizeof(miracl));
        MiraclMemcpy2Host(dst->mip, src->mip);
    }
    { // zzn4 Z0, Z1, Z2, Z3, T0, T1
        Zzn4Memcpy2Host(&(dst->Z0), &(src->Z0));
        Zzn4Memcpy2Host(&(dst->Z1), &(src->Z1));
        Zzn4Memcpy2Host(&(dst->Z2), &(src->Z2));
        Zzn4Memcpy2Host(&(dst->Z3), &(src->Z3));
        Zzn4Memcpy2Host(&(dst->T0), &(src->T0));
        Zzn4Memcpy2Host(&(dst->T1), &(src->T1));
    }
    if (src->ks != NULL)
    {
        NewMallocAtHost((void **)&(dst->ks), sizeof(bigtype));
        BigMemcpy2Host(dst->ks, src->ks);
    }
    { // struct SM9_Sign_Para SIGN
        if (src->SIGN.h1 != NULL)
        {
            NewMallocAtHost((void **)&(dst->SIGN.h1), sizeof(bigtype));
            BigMemcpy2Host(dst->SIGN.h1, src->SIGN.h1);
        }
        if (src->SIGN.r != NULL)
        {
            NewMallocAtHost((void **)&(dst->SIGN.r), sizeof(bigtype));
            BigMemcpy2Host(dst->SIGN.r, src->SIGN.r);
        }
        if (src->SIGN.h != NULL)
        {
            NewMallocAtHost((void **)&(dst->SIGN.h), sizeof(bigtype));
            BigMemcpy2Host(dst->SIGN.h, src->SIGN.h);
        }
        if (src->SIGN.l != NULL)
        {
            NewMallocAtHost((void **)&(dst->SIGN.l), sizeof(bigtype));
            BigMemcpy2Host(dst->SIGN.l, src->SIGN.l);
        }
        if (src->SIGN.xdSA != NULL)
        {
            NewMallocAtHost((void **)&(dst->SIGN.xdSA), sizeof(bigtype));
            BigMemcpy2Host(dst->SIGN.xdSA, src->SIGN.xdSA);
        }
        if (src->SIGN.ydSA != NULL)
        {
            NewMallocAtHost((void **)&(dst->SIGN.ydSA), sizeof(bigtype));
            BigMemcpy2Host(dst->SIGN.ydSA, src->SIGN.ydSA);
        }
        if (src->SIGN.xS != NULL)
        {
            NewMallocAtHost((void **)&(dst->SIGN.xS), sizeof(bigtype));
            BigMemcpy2Host(dst->SIGN.xS, src->SIGN.xS);
        }
        if (src->SIGN.yS != NULL)
        {
            NewMallocAtHost((void **)&(dst->SIGN.yS), sizeof(bigtype));
            BigMemcpy2Host(dst->SIGN.yS, src->SIGN.yS);
        }
        if (src->SIGN.tmp != NULL)
        {
            NewMallocAtHost((void **)&(dst->SIGN.tmp), sizeof(bigtype));
            BigMemcpy2Host(dst->SIGN.tmp, src->SIGN.tmp);
        }
        if (src->SIGN.zero != NULL)
        {
            NewMallocAtHost((void **)&(dst->SIGN.zero), sizeof(bigtype));
            BigMemcpy2Host(dst->SIGN.zero, src->SIGN.zero);
        }
        { // zzn12 g, w;
            Zzn12Memcpy2Host(&(dst->SIGN.g), &(src->SIGN.g));
            Zzn12Memcpy2Host(&(dst->SIGN.w), &(src->SIGN.w));
        }
        if (src->SIGN.s != NULL)
        {
            NewMallocAtHost((void **)&(dst->SIGN.s), sizeof(epoint));
            EpointMemcpy2Host(dst->SIGN.s, src->SIGN.s);
        }
        if (src->SIGN.dSA != NULL)
        {
            NewMallocAtHost((void **)&(dst->SIGN.dSA), sizeof(epoint));
            EpointMemcpy2Host(dst->SIGN.dSA, src->SIGN.dSA);
        }
        Ecn2Memcpy2Host(&(dst->SIGN.Ppubs), &(src->SIGN.Ppubs));
        if (src->SIGN.Z != NULL)
        {
            NewMallocAtHost((void **)&(dst->SIGN.Z), src->SIGN.Zlen * sizeof(unsigned char));
            void *gpu;
            cudaMalloc((void **)&gpu, src->SIGN.Zlen * sizeof(unsigned char));
            GPU2CPUMemcpy<<<1, 1>>>(gpu, src->SIGN.Z, src->SIGN.Zlen * sizeof(unsigned char));
            cudaDeviceSynchronize();
            cudaMemcpy(dst->SIGN.Z, gpu, src->SIGN.Zlen * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        }
    }
    { // SM9_Verify_Para VERIFY
        if (src->VERIFY.h != NULL)
        {
            NewMallocAtHost((void **)&(dst->VERIFY.h), sizeof(bigtype));
            BigMemcpy2Host(dst->VERIFY.h, src->VERIFY.h);
        }
        if (src->VERIFY.xS != NULL)
        {
            NewMallocAtHost((void **)&(dst->VERIFY.xS), sizeof(bigtype));
            BigMemcpy2Host(dst->VERIFY.xS, src->VERIFY.xS);
        }
        if (src->VERIFY.yS != NULL)
        {
            NewMallocAtHost((void **)&(dst->VERIFY.yS), sizeof(bigtype));
            BigMemcpy2Host(dst->VERIFY.yS, src->VERIFY.yS);
        }
        if (src->VERIFY.h1 != NULL)
        {
            NewMallocAtHost((void **)&(dst->VERIFY.h1), sizeof(bigtype));
            BigMemcpy2Host(dst->VERIFY.h1, src->VERIFY.h1);
        }
        if (src->VERIFY.h2 != NULL)
        {
            NewMallocAtHost((void **)&(dst->VERIFY.h2), sizeof(bigtype));
            BigMemcpy2Host(dst->VERIFY.h2, src->VERIFY.h2);
        }
        if (src->VERIFY.S1 != NULL)
        {
            NewMallocAtHost((void **)&(dst->VERIFY.S1), sizeof(epoint));
            EpointMemcpy2Host(dst->VERIFY.S1, src->VERIFY.S1);
        }
        { // zzn12 g, t, u, w;
            Zzn12Memcpy2Host(&(dst->VERIFY.g), &(src->VERIFY.g));
            Zzn12Memcpy2Host(&(dst->VERIFY.t), &(src->VERIFY.t));
            Zzn12Memcpy2Host(&(dst->VERIFY.u), &(src->VERIFY.u));
            Zzn12Memcpy2Host(&(dst->VERIFY.w), &(src->VERIFY.w));
        }
        Ecn2Memcpy2Host(&(dst->VERIFY.P), &(src->VERIFY.P));
        Ecn2Memcpy2Host(&(dst->VERIFY.Ppubs), &(src->VERIFY.Ppubs));
        if (src->VERIFY.Z1 != NULL)
        {
            NewMallocAtHost((void **)&(dst->VERIFY.Z1), src->VERIFY.Zlen1 * sizeof(unsigned char));
            void *gpu;
            cudaMalloc((void **)&gpu, src->VERIFY.Zlen1 * sizeof(unsigned char));
            GPU2CPUMemcpy<<<1, 1>>>(gpu, src->VERIFY.Z1, src->VERIFY.Zlen1 * sizeof(unsigned char));
            cudaDeviceSynchronize();
            cudaMemcpy(dst->VERIFY.Z1, gpu, src->VERIFY.Zlen1 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        }
        if (src->VERIFY.Z2 != NULL)
        {
            NewMallocAtHost((void **)&(dst->VERIFY.Z2), src->VERIFY.Zlen2 * sizeof(unsigned char));
            void *gpu;
            cudaMalloc((void **)&gpu, src->VERIFY.Zlen2 * sizeof(unsigned char));
            GPU2CPUMemcpy<<<1, 1>>>(gpu, src->VERIFY.Z2, src->VERIFY.Zlen2 * sizeof(unsigned char));
            cudaDeviceSynchronize();
            cudaMemcpy(dst->VERIFY.Z2, gpu, src->VERIFY.Zlen2 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        }
    }

    free(buffer);
    return 0;
}
