#include "SM9Cuda.cuh"

int NewMallocAtDevice(void **ptr, size_t size)
{
    cudaMalloc((void **)ptr, size);
    return 0;
}

int BigMemcpy2Device(big dst, big src) // dst from gpu
{
    big old_dst = dst;
    void *buffer;
    buffer = malloc(sizeof(bigtype));
    dst = (big)buffer;
    memcpy(dst, src, sizeof(bigtype));
    if (src->len != 0)
    {
        NewMallocAtDevice((void **)&(dst->w), (src->len & 0xfffff) * sizeof(mr_small));
        cudaMemcpy(dst->w, src->w, (src->len & 0xfffff) * sizeof(mr_small), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(old_dst, dst, sizeof(bigtype), cudaMemcpyHostToDevice);
    free(buffer);
    return 0;
}

int EpointMemcpy2Device(epoint *dst, epoint *src) // dst from gpu
{
    epoint *old_dst = dst;
    void *buffer;
    buffer = malloc(sizeof(epoint));
    dst = (epoint *)buffer;
    memcpy(dst, src, sizeof(epoint));
    if (src->X != NULL)
    {
        NewMallocAtDevice((void **)&(dst->X), sizeof(bigtype));
        BigMemcpy2Device(dst->X, src->X);
    }
    if (src->Y != NULL)
    {
        NewMallocAtDevice((void **)&(dst->Y), sizeof(bigtype));
        BigMemcpy2Device(dst->Y, src->Y);
    }
    if (src->Z != NULL)
    {
        NewMallocAtDevice((void **)&(dst->Z), sizeof(bigtype));
        BigMemcpy2Device(dst->Z, src->Z);
    }
    cudaMemcpy(old_dst, dst, sizeof(epoint), cudaMemcpyHostToDevice);
    free(buffer);
    return 0;
}

int Zzn2Memcpy2Device(zzn2 *dst, zzn2 *src) // dst from cpu
{
    memcpy(dst, src, sizeof(zzn2));

    if (src->a != NULL)
    {
        NewMallocAtDevice((void **)&(dst->a), sizeof(bigtype));
        BigMemcpy2Device(dst->a, src->a);
    }
    if (src->b != NULL)
    {
        NewMallocAtDevice((void **)&(dst->b), sizeof(bigtype));
        BigMemcpy2Device(dst->b, src->b);
    }
    return 0;
}

int Zzn4Memcpy2Device(zzn4 *dst, zzn4 *src) // dst from cpu
{
    memcpy(dst, src, sizeof(zzn4));

    Zzn2Memcpy2Device(&(dst->a), &(src->a));
    Zzn2Memcpy2Device(&(dst->b), &(src->b));
    return 0;
}

int Zzn12Memcpy2Device(zzn12 *dst, zzn12 *src) // dst from cpu
{
    memcpy(dst, src, sizeof(zzn12));

    Zzn4Memcpy2Device(&(dst->a), &(src->a));
    Zzn4Memcpy2Device(&(dst->b), &(src->b));
    Zzn4Memcpy2Device(&(dst->c), &(src->c));
    return 0;
}

int Ecn2Memcpy2Device(ecn2 *dst, ecn2 *src) // dst from cpu
{
    memcpy(dst, src, sizeof(ecn2));

    Zzn2Memcpy2Device(&(dst->x), &(src->x));
    Zzn2Memcpy2Device(&(dst->y), &(src->y));
    Zzn2Memcpy2Device(&(dst->z), &(src->z));
    return 0;
}

int MiraclMemcpy2Device(miracl *dst, miracl *src) // dst from gpu
{
    miracl *old_dst = dst;
    void *buffer;
    buffer = malloc(sizeof(miracl));
    dst = (miracl *)buffer;
    memcpy(dst, src, sizeof(miracl));
    // ????? BOOL(*user) (void); /* pointer to user supplied function */
    if (src->modulus != NULL)
    {
        NewMallocAtDevice((void **)&(dst->modulus), sizeof(bigtype));
        BigMemcpy2Device(dst->modulus, src->modulus);
    }
    if (src->pR != NULL)
    {
        NewMallocAtDevice((void **)&(dst->pR), sizeof(bigtype));
        BigMemcpy2Device(dst->pR, src->pR);
    }
//    if (src->prime != NULL)
//    {
//        NewMallocAtDevice((void **)&(dst->prime), sizeof(mr_utype));
//        cudaMemcpy(dst->prime, src->prime, sizeof(mr_utype), cudaMemcpyHostToDevice);
//    }
//    if (src->cr != NULL)
//    {
//        NewMallocAtDevice((void **)&(dst->cr), sizeof(mr_utype));
//        cudaMemcpy(dst->cr, src->cr, sizeof(mr_utype), cudaMemcpyHostToDevice);
//    }
//    if (src->inverse != NULL)
//    {
//        NewMallocAtDevice((void **)&(dst->inverse), sizeof(mr_utype));
//        cudaMemcpy(dst->inverse, src->inverse, sizeof(mr_utype), cudaMemcpyHostToDevice);
//    }
//    if (src->roots != NULL)
//    {
//        if (*(src->roots) == NULL)
//        {
//            NewMallocAtDevice((void **)&(dst->roots), sizeof(mr_utype *));
//            cudaMemcpy(dst->roots, src->roots, sizeof(mr_utype *), cudaMemcpyHostToDevice);
//        }
//        else
//        {
//            mr_utype *tmp;
//            NewMallocAtDevice((void **)&(tmp), sizeof(mr_utype));
//            cudaMemcpy(tmp, *(src->roots), sizeof(mr_utype), cudaMemcpyHostToDevice);
//            NewMallocAtDevice((void **)&(dst->roots), sizeof(mr_utype *));
//            cudaMemcpy(dst->roots, &tmp, sizeof(mr_utype *), cudaMemcpyHostToDevice);
//        }
//    }
//    { // small_chinese chin
//        if (src->chin.C != NULL)
//        {
//            NewMallocAtDevice((void **)&(dst->chin.C), sizeof(mr_utype));
//            cudaMemcpy(dst->chin.C, src->chin.C, sizeof(mr_utype), cudaMemcpyHostToDevice);
//        }
//        if (src->chin.V != NULL)
//        {
//            NewMallocAtDevice((void **)&(dst->chin.V), sizeof(mr_utype));
//            cudaMemcpy(dst->chin.V, src->chin.V, sizeof(mr_utype), cudaMemcpyHostToDevice);
//        }
//        if (src->chin.M != NULL)
//        {
//            NewMallocAtDevice((void **)&(dst->chin.M), sizeof(mr_utype));
//            cudaMemcpy(dst->chin.M, src->chin.M, sizeof(mr_utype), cudaMemcpyHostToDevice);
//        }
//    }
//    if (src->s1 != NULL)
//    {
//        if (*(src->s1) == NULL)
//        {
//            NewMallocAtDevice((void **)&(dst->s1), sizeof(mr_utype *));
//            cudaMemcpy(dst->s1, src->s1, sizeof(mr_utype *), cudaMemcpyHostToDevice);
//        }
//        else
//        {
//            mr_utype *tmp;
//            NewMallocAtDevice((void **)&(tmp), sizeof(mr_utype));
//            cudaMemcpy(tmp, *(src->s1), sizeof(mr_utype), cudaMemcpyHostToDevice);
//            NewMallocAtDevice((void **)&(dst->s1), sizeof(mr_utype *));
//            cudaMemcpy(dst->s1, &tmp, sizeof(mr_utype *), cudaMemcpyHostToDevice);
//        }
//    }
//    if (src->s2 != NULL)
//    {
//        if (*(src->s2) == NULL)
//        {
//            NewMallocAtDevice((void **)&(dst->s2), sizeof(mr_utype *));
//            cudaMemcpy(dst->s2, src->s2, sizeof(mr_utype *), cudaMemcpyHostToDevice);
//        }
//        else
//        {
//            mr_utype *tmp;
//            NewMallocAtDevice((void **)&(tmp), sizeof(mr_utype));
//            cudaMemcpy(tmp, *(src->s2), sizeof(mr_utype), cudaMemcpyHostToDevice);
//            NewMallocAtDevice((void **)&(dst->s2), sizeof(mr_utype *));
//            cudaMemcpy(dst->s2, &tmp, sizeof(mr_utype *), cudaMemcpyHostToDevice);
//        }
//    }
//    if (src->t != NULL)
//    {
//        if (*(src->t) == NULL)
//        {
//            NewMallocAtDevice((void **)&(dst->t), sizeof(mr_utype *));
//            cudaMemcpy(dst->t, src->t, sizeof(mr_utype *), cudaMemcpyHostToDevice);
//        }
//        else
//        {
//            mr_utype *tmp;
//            NewMallocAtDevice((void **)&(tmp), sizeof(mr_utype));
//            cudaMemcpy(tmp, *(src->t), sizeof(mr_utype), cudaMemcpyHostToDevice);
//            NewMallocAtDevice((void **)&(dst->t), sizeof(mr_utype *));
//            cudaMemcpy(dst->t, &tmp, sizeof(mr_utype *), cudaMemcpyHostToDevice);
//        }
//    }
//    if (src->wa != NULL)
//    {
//        NewMallocAtDevice((void **)&(dst->wa), sizeof(mr_utype));
//        cudaMemcpy(dst->wa, src->wa, sizeof(mr_utype), cudaMemcpyHostToDevice);
//    }
//    if (src->wb != NULL)
//    {
//        NewMallocAtDevice((void **)&(dst->wb), sizeof(mr_utype));
//        cudaMemcpy(dst->wb, src->wb, sizeof(mr_utype), cudaMemcpyHostToDevice);
//    }
//    if (src->wc != NULL)
//    {
//        NewMallocAtDevice((void **)&(dst->wc), sizeof(mr_utype));
//        cudaMemcpy(dst->wc, src->wc, sizeof(mr_utype), cudaMemcpyHostToDevice);
//    }
    if (src->w0 != NULL)
    {
        NewMallocAtDevice((void **)&(dst->w0), sizeof(bigtype));
        BigMemcpy2Device(dst->w0, src->w0);
    }
    if (src->w1 != NULL)
    {
        NewMallocAtDevice((void **)&(dst->w1), sizeof(bigtype));
        BigMemcpy2Device(dst->w1, src->w1);
    }
    if (src->w2 != NULL)
    {
        NewMallocAtDevice((void **)&(dst->w2), sizeof(bigtype));
        BigMemcpy2Device(dst->w2, src->w2);
    }
    if (src->w3 != NULL)
    {
        NewMallocAtDevice((void **)&(dst->w3), sizeof(bigtype));
        BigMemcpy2Device(dst->w3, src->w3);
    }
    if (src->w4 != NULL)
    {
        NewMallocAtDevice((void **)&(dst->w4), sizeof(bigtype));
        BigMemcpy2Device(dst->w4, src->w4);
    }
    if (src->w5 != NULL)
    {
        NewMallocAtDevice((void **)&(dst->w5), sizeof(bigtype));
        BigMemcpy2Device(dst->w5, src->w5);
    }
    if (src->w6 != NULL)
    {
        NewMallocAtDevice((void **)&(dst->w6), sizeof(bigtype));
        BigMemcpy2Device(dst->w6, src->w6);
    }
    if (src->w7 != NULL)
    {
        NewMallocAtDevice((void **)&(dst->w7), sizeof(bigtype));
        BigMemcpy2Device(dst->w7, src->w7);
    }
    if (src->w8 != NULL)
    {
        NewMallocAtDevice((void **)&(dst->w8), sizeof(bigtype));
        BigMemcpy2Device(dst->w8, src->w8);
    }
    if (src->w9 != NULL)
    {
        NewMallocAtDevice((void **)&(dst->w9), sizeof(bigtype));
        BigMemcpy2Device(dst->w9, src->w9);
    }
    if (src->w10 != NULL)
    {
        NewMallocAtDevice((void **)&(dst->w10), sizeof(bigtype));
        BigMemcpy2Device(dst->w10, src->w10);
    }
    if (src->w11 != NULL)
    {
        NewMallocAtDevice((void **)&(dst->w11), sizeof(bigtype));
        BigMemcpy2Device(dst->w11, src->w11);
    }
    if (src->w12 != NULL)
    {
        NewMallocAtDevice((void **)&(dst->w12), sizeof(bigtype));
        BigMemcpy2Device(dst->w12, src->w12);
    }
    if (src->w13 != NULL)
    {
        NewMallocAtDevice((void **)&(dst->w13), sizeof(bigtype));
        BigMemcpy2Device(dst->w13, src->w13);
    }
    if (src->w14 != NULL)
    {
        NewMallocAtDevice((void **)&(dst->w14), sizeof(bigtype));
        BigMemcpy2Device(dst->w14, src->w14);
    }
    if (src->w15 != NULL)
    {
        NewMallocAtDevice((void **)&(dst->w15), sizeof(bigtype));
        BigMemcpy2Device(dst->w15, src->w15);
    }
    if (src->sru != NULL)
    {
        NewMallocAtDevice((void **)&(dst->sru), sizeof(bigtype));
        BigMemcpy2Device(dst->sru, src->sru);
    }
    if (src->one != NULL)
    {
        NewMallocAtDevice((void **)&(dst->one), sizeof(bigtype));
        BigMemcpy2Device(dst->one, src->one);
    }
    if (src->A != NULL)
    {
        NewMallocAtDevice((void **)&(dst->A), sizeof(bigtype));
        BigMemcpy2Device(dst->A, src->A);
    }
    if (src->B != NULL)
    {
        NewMallocAtDevice((void **)&(dst->B), sizeof(bigtype));
        BigMemcpy2Device(dst->B, src->B);
    }
    if (src->PRIMES != NULL)
    {
        NewMallocAtDevice((void **)&(dst->PRIMES), sizeof(src->PRIMES));
        cudaMemcpy(dst->PRIMES, src->PRIMES, sizeof(src->PRIMES), cudaMemcpyHostToDevice); // sizeof(src->PRIMES)?
    }
//    if (src->IOBUFF != NULL)
//    {
//        NewMallocAtDevice((void **)&(dst->IOBUFF), sizeof(src->IOBUFF));
//        cudaMemcpy(dst->IOBUFF, src->IOBUFF, sizeof(src->IOBUFF), cudaMemcpyHostToDevice); // sizeof(src->IOBUFF)?
//    }
    if (src->pi != NULL)
    {
        NewMallocAtDevice((void **)&(dst->pi), sizeof(bigtype));
        BigMemcpy2Device(dst->pi, src->pi);
    }
//    if (src->workspace != NULL)
//    {
//        NewMallocAtDevice((void **)&(dst->workspace), sizeof(src->workspace));
//        cudaMemcpy(dst->workspace, src->workspace, sizeof(src->workspace), cudaMemcpyHostToDevice); // sizeof(src->workspace)?
//    }
    cudaMemcpy(old_dst, dst, sizeof(miracl), cudaMemcpyHostToDevice);
    free(buffer);
    return 0;
}

// 目标的dst在显存上时，需要先造一个在内存上的buffer，把结构体构造好后cudaMemcpy进dst
int ParaMemcpy2Device(struct SM9_Para *dst, struct SM9_Para *src)
{
    struct SM9_Para *old_dst = dst;
    void *buffer;
    buffer = malloc(sizeof(SM9_Para));
    dst = (struct SM9_Para *)buffer;
    memcpy(dst, src, sizeof(struct SM9_Para));
    if (src->P1 != NULL)
    {
        NewMallocAtDevice((void **)&(dst->P1), sizeof(epoint));
        EpointMemcpy2Device(dst->P1, src->P1);
    }
    { // ecn2 P2
        Ecn2Memcpy2Device(&(dst->P2), &(src->P2));
    }
    if (src->N != NULL)
    {
        NewMallocAtDevice((void **)&(dst->N), sizeof(bigtype));
        BigMemcpy2Device(dst->N, src->N);
    }
    if (src->para_a != NULL)
    {
        NewMallocAtDevice((void **)&(dst->para_a), sizeof(bigtype));
        BigMemcpy2Device(dst->para_a, src->para_a);
    }
    if (src->para_b != NULL)
    {
        NewMallocAtDevice((void **)&(dst->para_b), sizeof(bigtype));
        BigMemcpy2Device(dst->para_b, src->para_b);
    }
    if (src->para_t != NULL)
    {
        NewMallocAtDevice((void **)&(dst->para_t), sizeof(bigtype));
        BigMemcpy2Device(dst->para_t, src->para_t);
    }
    if (src->para_q != NULL)
    {
        NewMallocAtDevice((void **)&(dst->para_q), sizeof(bigtype));
        BigMemcpy2Device(dst->para_q, src->para_q);
    }
    { // zzn2 X
        Zzn2Memcpy2Device(&(dst->X), &(src->X));
    }
    if (src->mip != NULL)
    {
        NewMallocAtDevice((void **)&(dst->mip), sizeof(miracl));
        MiraclMemcpy2Device(dst->mip, src->mip);
    }
    { // zzn4 Z0, Z1, Z2, Z3, T0, T1
        Zzn4Memcpy2Device(&(dst->Z0), &(src->Z0));
        Zzn4Memcpy2Device(&(dst->Z1), &(src->Z1));
        Zzn4Memcpy2Device(&(dst->Z2), &(src->Z2));
        Zzn4Memcpy2Device(&(dst->Z3), &(src->Z3));
        Zzn4Memcpy2Device(&(dst->T0), &(src->T0));
        Zzn4Memcpy2Device(&(dst->T1), &(src->T1));
    }
    if (src->ks != NULL)
    {
        NewMallocAtDevice((void **)&(dst->ks), sizeof(bigtype));
        BigMemcpy2Device(dst->ks, src->ks);
    }
    { // struct SM9_Sign_Para SIGN
        if (src->SIGN.h1 != NULL)
        {
            NewMallocAtDevice((void **)&(dst->SIGN.h1), sizeof(bigtype));
            BigMemcpy2Device(dst->SIGN.h1, src->SIGN.h1);
        }
        if (src->SIGN.r != NULL)
        {
            NewMallocAtDevice((void **)&(dst->SIGN.r), sizeof(bigtype));
            BigMemcpy2Device(dst->SIGN.r, src->SIGN.r);
        }
        if (src->SIGN.h != NULL)
        {
            NewMallocAtDevice((void **)&(dst->SIGN.h), sizeof(bigtype));
            BigMemcpy2Device(dst->SIGN.h, src->SIGN.h);
        }
        if (src->SIGN.l != NULL)
        {
            NewMallocAtDevice((void **)&(dst->SIGN.l), sizeof(bigtype));
            BigMemcpy2Device(dst->SIGN.l, src->SIGN.l);
        }
        if (src->SIGN.xdSA != NULL)
        {
            NewMallocAtDevice((void **)&(dst->SIGN.xdSA), sizeof(bigtype));
            BigMemcpy2Device(dst->SIGN.xdSA, src->SIGN.xdSA);
        }
        if (src->SIGN.ydSA != NULL)
        {
            NewMallocAtDevice((void **)&(dst->SIGN.ydSA), sizeof(bigtype));
            BigMemcpy2Device(dst->SIGN.ydSA, src->SIGN.ydSA);
        }
        if (src->SIGN.xS != NULL)
        {
            NewMallocAtDevice((void **)&(dst->SIGN.xS), sizeof(bigtype));
            BigMemcpy2Device(dst->SIGN.xS, src->SIGN.xS);
        }
        if (src->SIGN.yS != NULL)
        {
            NewMallocAtDevice((void **)&(dst->SIGN.yS), sizeof(bigtype));
            BigMemcpy2Device(dst->SIGN.yS, src->SIGN.yS);
        }
        if (src->SIGN.tmp != NULL)
        {
            NewMallocAtDevice((void **)&(dst->SIGN.tmp), sizeof(bigtype));
            BigMemcpy2Device(dst->SIGN.tmp, src->SIGN.tmp);
        }
        if (src->SIGN.zero != NULL)
        {
            NewMallocAtDevice((void **)&(dst->SIGN.zero), sizeof(bigtype));
            BigMemcpy2Device(dst->SIGN.zero, src->SIGN.zero);
        }
        { // zzn12 g, w;
            Zzn12Memcpy2Device(&(dst->SIGN.g), &(src->SIGN.g));
            Zzn12Memcpy2Device(&(dst->SIGN.w), &(src->SIGN.w));
        }
        if (src->SIGN.s != NULL)
        {
            NewMallocAtDevice((void **)&(dst->SIGN.s), sizeof(epoint));
            EpointMemcpy2Device(dst->SIGN.s, src->SIGN.s);
        }
        if (src->SIGN.dSA != NULL)
        {
            NewMallocAtDevice((void **)&(dst->SIGN.dSA), sizeof(epoint));
            EpointMemcpy2Device(dst->SIGN.dSA, src->SIGN.dSA);
        }
        Ecn2Memcpy2Device(&(dst->SIGN.Ppubs), &(src->SIGN.Ppubs));
        if (src->SIGN.Z != NULL)
        {
            NewMallocAtDevice((void **)&(dst->SIGN.Z), src->SIGN.Zlen * sizeof(unsigned char));
            cudaMemcpy(dst->SIGN.Z, src->SIGN.Z, src->SIGN.Zlen * sizeof(unsigned char), cudaMemcpyHostToDevice);
        }
    }
    { // SM9_Verify_Para VERIFY
        if (src->VERIFY.h != NULL)
        {
            NewMallocAtDevice((void **)&(dst->VERIFY.h), sizeof(bigtype));
            BigMemcpy2Device(dst->VERIFY.h, src->VERIFY.h);
        }
        if (src->VERIFY.xS != NULL)
        {
            NewMallocAtDevice((void **)&(dst->VERIFY.xS), sizeof(bigtype));
            BigMemcpy2Device(dst->VERIFY.xS, src->VERIFY.xS);
        }
        if (src->VERIFY.yS != NULL)
        {
            NewMallocAtDevice((void **)&(dst->VERIFY.yS), sizeof(bigtype));
            BigMemcpy2Device(dst->VERIFY.yS, src->VERIFY.yS);
        }
        if (src->VERIFY.h1 != NULL)
        {
            NewMallocAtDevice((void **)&(dst->VERIFY.h1), sizeof(bigtype));
            BigMemcpy2Device(dst->VERIFY.h1, src->VERIFY.h1);
        }
        if (src->VERIFY.h2 != NULL)
        {
            NewMallocAtDevice((void **)&(dst->VERIFY.h2), sizeof(bigtype));
            BigMemcpy2Device(dst->VERIFY.h2, src->VERIFY.h2);
        }
        if (src->VERIFY.S1 != NULL)
        {
            NewMallocAtDevice((void **)&(dst->VERIFY.S1), sizeof(epoint));
            EpointMemcpy2Device(dst->VERIFY.S1, src->VERIFY.S1);
        }
        { // zzn12 g, t, u, w;
            Zzn12Memcpy2Device(&(dst->VERIFY.g), &(src->VERIFY.g));
            Zzn12Memcpy2Device(&(dst->VERIFY.t), &(src->VERIFY.t));
            Zzn12Memcpy2Device(&(dst->VERIFY.u), &(src->VERIFY.u));
            Zzn12Memcpy2Device(&(dst->VERIFY.w), &(src->VERIFY.w));
        }
        Ecn2Memcpy2Device(&(dst->VERIFY.P), &(src->VERIFY.P));
        Ecn2Memcpy2Device(&(dst->VERIFY.Ppubs), &(src->VERIFY.Ppubs));
        if (src->VERIFY.Z1 != NULL)
        {
            NewMallocAtDevice((void **)&(dst->VERIFY.Z1), src->VERIFY.Zlen1 * sizeof(unsigned char));
            cudaMemcpy(dst->VERIFY.Z1, src->VERIFY.Z1, src->VERIFY.Zlen1 * sizeof(unsigned char), cudaMemcpyHostToDevice);
        }
        if (src->VERIFY.Z2 != NULL)
        {
            NewMallocAtDevice((void **)&(dst->VERIFY.Z2), src->VERIFY.Zlen2 * sizeof(unsigned char));
            cudaMemcpy(dst->VERIFY.Z2, src->VERIFY.Z2, src->VERIFY.Zlen2 * sizeof(unsigned char), cudaMemcpyHostToDevice);
        }
    }

    cudaMemcpy(old_dst, dst, sizeof(SM9_Para), cudaMemcpyHostToDevice);
    free(buffer);
    return 0;
}
