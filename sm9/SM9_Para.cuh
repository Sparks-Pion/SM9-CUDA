#pragma once

#include "../miracl/miracl.cuh"

typedef struct
{
    zzn4 a, b, c;
    BOOL unitary; // "unitary property means that fast squaring can be used, and inversions are just conjugates
    BOOL miller;  // "miller" property means that arithmetic on this instance can ignore multiplications
                  // or divisions by constants - as instance will eventually be raised to (p-1).
} zzn12;

struct SM9_Sign_Para
{
    big h1, r, h, l, xdSA, ydSA;
    big xS, yS, tmp, zero;
    zzn12 g, w;
    epoint *s, *dSA;
    ecn2 Ppubs;
    int Zlen, buf;
    unsigned char *Z = NULL;
};

struct SM9_Verify_Para
{
    big h, xS, yS, h1, h2;
    epoint *S1;
    zzn12 g, t, u, w;
    ecn2 P, Ppubs;
    int Zlen1, Zlen2, buf;
    unsigned char *Z1 = NULL, *Z2 = NULL;
};

struct SM9_Para
{
    // para from SM9_sv.h:
    unsigned char SM9_q[32];
    unsigned char SM9_N[32];
    unsigned char SM9_P1x[32];
    unsigned char SM9_P1y[32];
    unsigned char SM9_P2[128];
    unsigned char SM9_t[32];
    unsigned char SM9_a[32];
    unsigned char SM9_b[32];
    epoint *P1;
    ecn2 P2;
    big N; // order of group, N(t)
    big para_a, para_b, para_t, para_q;
    // para from zzn12_operation.h:
    zzn2 X; // Frobniues constant
    miracl *mip;
    zzn4 Z0, Z1, Z2, Z3, T0, T1; // Karatsuba for zzn12_calc

    // para from SM9_SelfCheck
    unsigned char dA[32];
    unsigned char rand[32];
    unsigned char h[32], S[64]; // Signature
    unsigned char Ppub[128], dSA[64];
    unsigned char std_h[32];
    unsigned char std_S[64];
    unsigned char std_Ppub[128];
    unsigned char std_dSA[64];
    unsigned char hid[2];
    unsigned char IDA[10];
    unsigned char message[30]; // the message to be signed
    int mlen;                  // the length of message
    int tmp;
    big ks;

    struct SM9_Sign_Para SIGN;
    struct SM9_Verify_Para VERIFY;

    zzn2 w1, w2, w3, w4, w5, w6, w7, w8, lam, extra;
    zzn2 fx, fy, fz, fw, fr;
    big b1;
    ecn2 P;
    zzn12 w;

    zzn12 res_g;

    zzn12 member_w;
    big member_six;

    zzn4 tmp1, tmp2;
//    zzn12 inverse_res;
    zzn2 X2, X3;
    big zero, tmp000, tmp11;
};