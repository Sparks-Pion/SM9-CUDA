#include "SM9Cuda.cuh"

__device__ __host__ void Print_String(unsigned char *tmp, size_t size)
{
    for (int i = 0; i < size; i++)
        printf("%02X", tmp[i]);
    printf("\n");
}

__device__ __host__ void Print_Miracl(miracl *mip)
{
    printf("miracl:\n");
    printf("base:%llu\n", mip->base);
    printf("apbase:%llu\n", mip->apbase);
    printf("pack:%d\n", mip->pack);
    printf("lg2b:%d\n", mip->lg2b);
    printf("base2:%llu\n", mip->base2);
//    printf("user:%p\n", mip->user);
    printf("nib:%d\n", mip->nib);
    printf("depth:%d\n", mip->depth);
    printf("trace:\n");
    for (int i = 0; i < MR_MAXDEPTH; i++)
        printf("%d ", (mip->trace)[i]);
    printf("\n");
    printf("check:%d\n", mip->check);
    printf("fout:%d\n", mip->fout);
    printf("fin:%d\n", mip->fin);
    printf("active:%d\n", mip->active);
    printf("ira:\n");
    for (int i = 0; i < NK; i++)
        printf("%u ", (mip->ira)[i]);
    printf("\n");
    printf("rndptr:%d\n", mip->rndptr);
    printf("borrow:%u\n", mip->borrow);
    printf("ndash:%llu\n", mip->ndash);
    printf("modulus:\n");
    if (mip->modulus != NULL)
        Print_Big(mip->modulus);
    printf("pR:\n");
    if (mip->pR != NULL)
        Print_Big(mip->pR);
    printf("ACTIVE:%d\n", mip->ACTIVE);
    printf("MONTY:%d\n", mip->MONTY);
    printf("SS:%d\n", mip->SS);
    printf("KOBLITZ:%d\n", mip->KOBLITZ);
    printf("coord:%d\n", mip->coord);
    printf("Asize:%d\n", mip->Asize);
    printf("Bsize:%d\n", mip->Bsize);
    printf("M:%d\n", mip->M);
    printf("AA:%d\n", mip->AA);
    printf("BB:%d\n", mip->BB);
    printf("CC:%d\n", mip->CC);
//    printf("logN:%d\n", mip->logN);
//    printf("nprimes:%d\n", mip->nprimes);
//    printf("degree:%d\n", mip->degree);
//    printf("prime:\n");
//    if (mip->prime != NULL)
//        printf("%ld\n", *(mip->prime));
//    printf("cr:\n");
//    if (mip->cr != NULL)
//        printf("%ld\n", *(mip->cr));
//    printf("inverse:\n");
//    if (mip->inverse != NULL)
//        printf("%ld\n", *(mip->inverse));
    // roots
//    printf("chin:\n");
//    if (mip->chin.C != NULL)
//        printf("chinC:%ld\n", *(mip->chin.C));
//    if (mip->chin.V != NULL)
//        printf("chinV:%ld\n", *(mip->chin.V));
//    if (mip->chin.M != NULL)
//        printf("chinM:%ld\n", *(mip->chin.M));
//    printf("chinNP:%d\n", mip->chin.NP);
//    printf("const1:%ld\n", mip->const1);
//    printf("const2:%ld\n", mip->const2);
//    printf("const3:%ld\n", mip->const3);
//    printf("msw:%llu\n", mip->msw);
//    printf("lsw:%llu\n", mip->lsw);
    // s1
    // s2
    // t
//    if (mip->wa != NULL)
//        printf("wa:%ld\n", *(mip->wa));
//    if (mip->wb != NULL)
//        printf("wb:%ld\n", *(mip->wb));
//    if (mip->wc != NULL)
//        printf("wc:%ld\n", *(mip->wc));
    printf("same:%d\n", mip->same);
    printf("first_one:%d\n", mip->first_one);
    printf("debug:%d\n", mip->debug);
    printf("w0:\n");
    if (mip->w0 != NULL)
        Print_Big(mip->w0);
    printf("w1:\n");
    if (mip->w1 != NULL)
        Print_Big(mip->w1);
    printf("w2:\n");
    if (mip->w2 != NULL)
        Print_Big(mip->w2);
    printf("w3:\n");
    if (mip->w3 != NULL)
        Print_Big(mip->w3);
    printf("w4:\n");
    if (mip->w4 != NULL)
        Print_Big(mip->w4);
    printf("w5:\n");
    if (mip->w5 != NULL)
        Print_Big(mip->w5);
    printf("w6:\n");
    if (mip->w6 != NULL)
        Print_Big(mip->w6);
    printf("w7:\n");
    if (mip->w7 != NULL)
        Print_Big(mip->w7);
    printf("w8:\n");
    if (mip->w8 != NULL)
        Print_Big(mip->w8);
    printf("w9:\n");
    if (mip->w9 != NULL)
        Print_Big(mip->w9);
    printf("w10:\n");
    if (mip->w10 != NULL)
        Print_Big(mip->w10);
    printf("w11:\n");
    if (mip->w11 != NULL)
        Print_Big(mip->w11);
    printf("w12:\n");
    if (mip->w12 != NULL)
        Print_Big(mip->w12);
    printf("w13:\n");
    if (mip->w13 != NULL)
        Print_Big(mip->w13);
    printf("w14:\n");
    if (mip->w14 != NULL)
        Print_Big(mip->w14);
    printf("w15:\n");
    if (mip->w15 != NULL)
        Print_Big(mip->w15);
    printf("sru:\n");
    if (mip->sru != NULL)
        Print_Big(mip->sru);
    printf("one:\n");
    if (mip->one != NULL)
        Print_Big(mip->one);
    printf("A:\n");
    if (mip->A != NULL)
        Print_Big(mip->A);
    printf("B:\n");
    if (mip->B != NULL)
        Print_Big(mip->B);
//    printf("IOBSIZ:%d\n", mip->IOBSIZ);
    printf("ERCON:%d\n", mip->ERCON);
    printf("ERNUM:%d\n", mip->ERNUM);
    printf("NTRY:%d\n", mip->NTRY);
//    printf("INPLEN:%d\n", mip->INPLEN);
//    printf("IOBASE:%d\n", mip->IOBASE);
    printf("EXACT:%d\n", mip->EXACT);
    printf("RPOINT:%d\n", mip->RPOINT);
    printf("TRACER:%d\n", mip->TRACER);
    printf("PRIMES:\n");
    if (mip->PRIMES != NULL)
        printf("%d\n", *(mip->PRIMES));
    printf("IOBUFF:\n");
//    if (mip->IOBUFF != NULL)
//        printf("%s\n", mip->IOBUFF);
    printf("workprec:%d\n", mip->workprec);
    printf("stprec:%d\n", mip->stprec);
    printf("RS:%d\n", mip->RS);
    printf("RD:%d\n", mip->RD);
    printf("D:%.4lf\n", mip->D);
    printf("db:%.4lf\n", mip->db);
    printf("n:%.4lf\n", mip->n);
    printf("p:%.4lf\n", mip->p);
    printf("a:%d\n", mip->a);
    printf("b:%d\n", mip->b);
    printf("c:%d\n", mip->c);
    printf("d:%d\n", mip->d);
    printf("r:%d\n", mip->r);
    printf("q:%d\n", mip->q);
    printf("oldn:%d\n", mip->oldn);
    printf("ndig:%d\n", mip->ndig);
    printf("u:%llu\n", mip->u);
    printf("v:%llu\n", mip->v);
    printf("ku:%llu\n", mip->ku);
    printf("kv:%llu\n", mip->kv);
    printf("last:%d\n", mip->last);
    printf("carryon:%d\n", mip->carryon);
    printf("pi:\n");
    if (mip->pi != NULL)
        Print_Big(mip->pi);
    printf("workspace:\n");
//    if (mip->workspace != NULL)
//        printf("%s\n", mip->workspace);
    printf("TWIST:%d\n", mip->TWIST);
    printf("qnr:%d\n", mip->qnr);
    printf("cnr:%d\n", mip->cnr);
    printf("pmod8:%d\n", mip->pmod8);
    printf("pmod9:%d\n", mip->pmod9);
    printf("NO_CARRY:%d\n", mip->NO_CARRY);
}
__device__ __host__ void Print_Big(big tmp)
{
    printf("big:\n");
    for (int i = 0; i < tmp->len; i++)
        printf("%llu ", tmp->w[i]);
    printf("\n");
}
__device__ __host__ void Print_Zzn2(zzn2 *tmp)
{
    printf("Zzn2:\n");
    if (tmp->a != NULL)
        Print_Big(tmp->a);
    if (tmp->b != NULL)
        Print_Big(tmp->b);
}
__device__ __host__ void Print_Zzn4(zzn4 *tmp)
{
    printf("Zzn4:\n");
    Print_Zzn2(&(tmp->a));
    Print_Zzn2(&(tmp->b));
    printf("unitary:%d\n", tmp->unitary);
}
__device__ __host__ void Print_Zzn12(zzn12 *tmp)
{
    printf("Zzn12:\n");
    Print_Zzn4(&(tmp->a));
    Print_Zzn4(&(tmp->b));
    Print_Zzn4(&(tmp->c));
    printf("unitary:%d\n", tmp->unitary);
    printf("miller:%d\n", tmp->miller);
}
__device__ __host__ void Print_Ecn2(ecn2 *tmp)
{
    printf("Ecn2:\n");
    printf("marker:%d\n", tmp->marker);
    Print_Zzn2(&(tmp->x));
    Print_Zzn2(&(tmp->x));
    Print_Zzn2(&(tmp->x));
}
__device__ __host__ void Print_Epoint(epoint *tmp)
{
    printf("Epoint:\n");
    printf("marker:%d\n", tmp->marker);
    if (tmp->X != NULL)
        Print_Big(tmp->X);
    if (tmp->Y != NULL)
        Print_Big(tmp->Y);
    if (tmp->Z != NULL)
        Print_Big(tmp->Z);
}
__device__ __host__ void Print_Para(struct SM9_Para *para)
{
    printf("【para】:\n");
    printf("SM9_q:\n");
    Print_String((unsigned char *)para->SM9_q, 32);
    printf("SM9_N:\n");
    Print_String((unsigned char *)para->SM9_N, 32);
    printf("SM9_P1x:\n");
    Print_String((unsigned char *)para->SM9_P1x, 32);
    printf("SM9_P1y:\n");
    Print_String((unsigned char *)para->SM9_P1y, 32);
    printf("SM9_P2:\n");
    Print_String((unsigned char *)para->SM9_P2, 128);
    printf("SM9_t:\n");
    Print_String((unsigned char *)para->SM9_t, 32);
    printf("SM9_a:\n");
    Print_String((unsigned char *)para->SM9_a, 32);
    printf("SM9_b:\n");
    Print_String((unsigned char *)para->SM9_b, 32);
    printf("P1:\n");
    if (para->P1 != NULL)
        Print_Epoint(para->P1);
    printf("P2:\n");
    Print_Ecn2(&(para->P2));
    printf("N:\n");
    if (para->N != NULL)
        Print_Big(para->N);
    printf("para_a:\n");
    if (para->para_a != NULL)
        Print_Big(para->para_a);
    printf("para_b:\n");
    if (para->para_b != NULL)
        Print_Big(para->para_b);
    printf("para_t:\n");
    if (para->para_t != NULL)
        Print_Big(para->para_t);
    printf("para_q:\n");
    if (para->para_q != NULL)
        Print_Big(para->para_q);
    printf("X:\n");
    Print_Zzn2(&(para->X));
    printf("mip:\n");
    if (para->mip != NULL)
        Print_Miracl(para->mip);
    printf("Z0:\n");
    Print_Zzn4(&(para->Z0));
    printf("Z1:\n");
    Print_Zzn4(&(para->Z1));
    printf("Z2:\n");
    Print_Zzn4(&(para->Z2));
    printf("Z3:\n");
    Print_Zzn4(&(para->Z3));
    printf("T0:\n");
    Print_Zzn4(&(para->T0));
    printf("T1:\n");
    Print_Zzn4(&(para->T1));
    printf("dA:\n");
    Print_String((unsigned char *)para->dA, 32);
    printf("rand:\n");
    Print_String((unsigned char *)para->rand, 32);
    printf("h:\n");
    Print_String((unsigned char *)para->h, 32);
    printf("S:\n");
    Print_String((unsigned char *)para->S, 64);
    printf("Ppub:\n");
    Print_String((unsigned char *)para->Ppub, 128);
    printf("dSA:\n");
    Print_String((unsigned char *)para->dSA, 64);
    printf("std_h:\n");
    Print_String((unsigned char *)para->std_h, 32);
    printf("std_S:\n");
    Print_String((unsigned char *)para->std_S, 64);
    printf("std_Ppub:\n");
    Print_String((unsigned char *)para->std_Ppub, 128);
    printf("std_dSA:\n");
    Print_String((unsigned char *)para->std_dSA, 64);
    printf("hid:\n");
    Print_String((unsigned char *)para->hid, 2);
    printf("IDA:\n");
    Print_String((unsigned char *)para->IDA, 10);
    printf("message:\n");
    Print_String((unsigned char *)para->message, 30);
    printf("mlen:%d\n", para->mlen);
    printf("tmp:%d\n", para->tmp);
    printf("ks:\n");
    if (para->ks != NULL)
        Print_Big(para->ks);
    printf("【SM9_Sign_Para】:\n");
    printf("h1:\n");
    if (para->SIGN.h1 != NULL)
        Print_Big(para->SIGN.h1);
    printf("r:\n");
    if (para->SIGN.r != NULL)
        Print_Big(para->SIGN.r);
    printf("h:\n");
    if (para->SIGN.h != NULL)
        Print_Big(para->SIGN.h);
    printf("l:\n");
    if (para->SIGN.l != NULL)
        Print_Big(para->SIGN.l);
    printf("xdSA:\n");
    if (para->SIGN.xdSA != NULL)
        Print_Big(para->SIGN.xdSA);
    printf("ydSA:\n");
    if (para->SIGN.ydSA != NULL)
        Print_Big(para->SIGN.ydSA);
    printf("xS:\n");
    if (para->SIGN.xS != NULL)
        Print_Big(para->SIGN.xS);
    printf("yS:\n");
    if (para->SIGN.yS != NULL)
        Print_Big(para->SIGN.yS);
    printf("tmp:\n");
    if (para->SIGN.tmp != NULL)
        Print_Big(para->SIGN.tmp);
    printf("zero:\n");
    if (para->SIGN.zero != NULL)
        Print_Big(para->SIGN.zero);
    printf("g:\n");
    Print_Zzn12(&(para->SIGN.g));
    printf("w:\n");
    Print_Zzn12(&(para->SIGN.w));
    printf("s:\n");
    if (para->SIGN.s != NULL)
        Print_Epoint(para->SIGN.s);
    printf("dSA:\n");
    if (para->SIGN.dSA != NULL)
        Print_Epoint(para->SIGN.dSA);
    printf("Ppubs:\n");
    Print_Ecn2(&(para->SIGN.Ppubs));
    printf("Zlen:%d\n", para->SIGN.Zlen);
    printf("buf:%d\n", para->SIGN.buf);
    printf("Z:\n");
    if (para->SIGN.Z != NULL)
        printf("%s\n", para->SIGN.Z);
    printf("【SM9_Verify_Para】:\n");
    printf("h:\n");
    if (para->VERIFY.h != NULL)
        Print_Big(para->VERIFY.h);
    printf("xS:\n");
    if (para->VERIFY.xS != NULL)
        Print_Big(para->VERIFY.xS);
    printf("yS:\n");
    if (para->VERIFY.yS != NULL)
        Print_Big(para->VERIFY.yS);
    printf("h1:\n");
    if (para->VERIFY.h1 != NULL)
        Print_Big(para->VERIFY.h1);
    printf("h2:\n");
    if (para->VERIFY.h2 != NULL)
        Print_Big(para->VERIFY.h2);
    printf("S1:\n");
    if (para->VERIFY.S1 != NULL)
        Print_Epoint(para->VERIFY.S1);
    printf("g:\n");
    Print_Zzn12(&(para->VERIFY.g));
    printf("t:\n");
    Print_Zzn12(&(para->VERIFY.t));
    printf("u:\n");
    Print_Zzn12(&(para->VERIFY.u));
    printf("w:\n");
    Print_Zzn12(&(para->VERIFY.w));
    printf("P:\n");
    Print_Ecn2(&(para->VERIFY.P));
    printf("Ppubs:\n");
    Print_Ecn2(&(para->VERIFY.Ppubs));
    printf("Zlen1:%d\n", para->VERIFY.Zlen1);
    printf("Zlen2:%d\n", para->VERIFY.Zlen2);
    printf("buf:%d\n", para->VERIFY.buf);
    printf("Z1:\n");
    if (para->VERIFY.Z1 != NULL)
        printf("%s\n", para->VERIFY.Z1);
    printf("Z2:\n");
    if (para->VERIFY.Z2 != NULL)
        printf("%s\n", para->VERIFY.Z2);
}
