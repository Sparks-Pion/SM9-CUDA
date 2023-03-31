#pragma once

#include "SM9_Para.cuh"

/************************************************************************
File name:    zzn12_operation.h
Version:
Date:         Dec 15,2016
Description:  this code is achieved according to zzn12a.h and zzn12a.cpp in MIRCAL C++ source file writen by M. Scott.
so,see zzn12a.h and zzn12a.cpp for details.
this code define one struct zzn12,and based on it give many fuctions.
Function List:
1.zzn12_init           //Initiate struct zzn12
2.zzn12_copy           //copy one zzn12 to another
3.zzn12_mul            //z=x*y,achieve multiplication with two zzn12
4.zzn12_conj           //achieve conjugate complex
5.zzn12_inverse        //element inversion
6.zzn12_powq           //
7.zzn12_div            //division operation
8.zzn12_pow            //regular zzn12 powering

Notes:
**************************************************************************/
// typedef struct
// {
//     zzn4 a, b, c;
//     BOOL unitary; // "unitary property means that fast squaring can be used, and inversions are just conjugates
//     BOOL miller;  // "miller" property means that arithmetic on this instance can ignore multiplications
//                   // or divisions by constants - as instance will eventually be raised to (p-1).
// } zzn12;
extern __FUNCTION_HEADER__ void zzn12_init(struct SM9_Para *para, zzn12 *x);
extern __FUNCTION_HEADER__ void zzn12_copy(struct SM9_Para *para, zzn12 *x, zzn12 *y);
extern __FUNCTION_HEADER__ void zzn12_mul(struct SM9_Para *para, zzn12 x, zzn12 y, zzn12 *z);
extern __FUNCTION_HEADER__ void zzn12_conj(struct SM9_Para *para, zzn12 *x, zzn12 *y);
extern __FUNCTION_HEADER__ zzn12 zzn12_inverse(struct SM9_Para *para, zzn12 w);
extern __FUNCTION_HEADER__ void zzn12_powq(struct SM9_Para *para, zzn2 F, zzn12 *y);
extern __FUNCTION_HEADER__ void zzn12_div(struct SM9_Para *para, zzn12 x, zzn12 y, zzn12 *z);
extern __FUNCTION_HEADER__ zzn12 zzn12_pow(struct SM9_Para *para, zzn12 x, big k);

/************************************************************************
File name:    R-ate.h
Version:
Date:         Dec 15,2016
Description:  this code is achieved according to ake12bnx.cpp in MIRCAL C++ source file.
see ake12bnx.cpp for details.
this code gives calculation of R-ate pairing
Function List:
1.zzn2_pow                //regular zzn2 powering
2.set_frobenius_constant  //calculate frobenius_constant X
3.q_power_frobenius
4.line
5.g
6.fast_pairing
7.ecap

Notes:
**************************************************************************/
extern __FUNCTION_HEADER__ zzn2 zzn2_pow(struct SM9_Para *para, zzn2 x, big k);
extern __FUNCTION_HEADER__ void q_power_frobenius(struct SM9_Para *para, ecn2 A, zzn2 F);
extern __FUNCTION_HEADER__ zzn12 line(struct SM9_Para *para, ecn2 A, ecn2 *C, ecn2 *B, zzn2 slope, zzn2 extra, BOOL Doubling, big Qx, big Qy);
extern __FUNCTION_HEADER__ zzn12 g(struct SM9_Para *para, ecn2 *A, ecn2 *B, big Qx, big Qy);
extern __FUNCTION_HEADER__ BOOL fast_pairing(struct SM9_Para *para, ecn2 P, big Qx, big Qy, big x, zzn2 X, zzn12 *r);
extern __FUNCTION_HEADER__ BOOL ecap(struct SM9_Para *para, ecn2 P, epoint *Q, big x, zzn2 X, zzn12 *r);
extern __FUNCTION_HEADER__ BOOL member(struct SM9_Para *para, zzn12 r, big x, zzn2 F);

/************************************************************************
    FileName:
        SM3.h
    Version:
        SM3_V1.1
    Date:
        Sep 18,2016
    Description:
        This headfile provide macro defination, parameter definition and function declaration needed in SM3 algorithm implement
    Function List:
        1.SM3_256 //calls SM3_init, SM3_process and SM3_done to calculate hash value
        2.SM3_init //init the SM3 state
        3.SM3_process //compress the the first len/64 blocks of the message
        4.SM3_done //compress the rest message and output the hash value
        5.SM3_compress //called by SM3_process and SM3_done, compress a single block of message
        6.BiToW //called by SM3_compress,to calculate W from Bi
        7.WToW1 //called by SM3_compress, calculate W' from W
        8.CF //called by SM3_compress, to calculate CF function.
        9.BigEndian //called by SM3_compress and SM3_done.GM/T 0004-2012 requires to use big-endian. //if CPU uses little-endian, BigEndian function is a necessary call to change the little-endian format into big-endian format.
        10.SM3_SelfTest //test whether the SM3 calculation is correct by comparing the hash result with the standard data
    History:
        1. Date: Sep 18,2016
    Author: Mao Yingying, Huo Lili
    Modification: 1)add notes to all the functions
                  2)add SM3_SelfTest function
************************************************************************/
#define SM3_len 256
#define SM3_T1 0x79CC4519
#define SM3_T2 0x7A879D8A
#define SM3_IVA 0x7380166f
#define SM3_IVB 0x4914b2b9
#define SM3_IVC 0x172442d7
#define SM3_IVD 0xda8a0600
#define SM3_IVE 0xa96f30bc
#define SM3_IVF 0x163138aa
#define SM3_IVG 0xe38dee4d
#define SM3_IVH 0xb0fb0e4e
/* Various logical functions */
#define SM3_p1(x) (x ^ SM3_rotl32(x, 15) ^ SM3_rotl32(x, 23))
#define SM3_p0(x) (x ^ SM3_rotl32(x, 9) ^ SM3_rotl32(x, 17))
#define SM3_ff0(a, b, c) (a ^ b ^ c)
#define SM3_ff1(a, b, c) ((a & b) | (a & c) | (b & c))
#define SM3_gg0(e, f, g) (e ^ f ^ g)
#define SM3_gg1(e, f, g) ((e & f) | ((~e) & g))
#define SM3_rotl32(x, n) ((((unsigned int)x) << n) | (((unsigned int)x) >> (32 - n)))
#define SM3_rotr32(x, n) ((((unsigned int)x) >> n) | (((unsigned int)x) << (32 - n)))
typedef struct
{
    unsigned int state[8];
    unsigned int length;
    unsigned int curlen;
    unsigned char buf[64];
} SM3_STATE;
extern __FUNCTION_HEADER__ void BiToWj(unsigned int Bi[], unsigned int Wj[]);
extern __FUNCTION_HEADER__ void WjToWj1(unsigned int Wj[], unsigned int Wj1[]);
extern __FUNCTION_HEADER__ void CF(unsigned int Wj[], unsigned int Wj1[], unsigned int V[]);
extern __FUNCTION_HEADER__ void BigEndian(unsigned char src[], unsigned int bytelen, unsigned char des[]);
extern __FUNCTION_HEADER__ void SM3_init(SM3_STATE *md);
extern __FUNCTION_HEADER__ void SM3_compress(SM3_STATE *md);
extern __FUNCTION_HEADER__ void SM3_process(SM3_STATE *md, unsigned char buf[], int len);
extern __FUNCTION_HEADER__ void SM3_done(SM3_STATE *md, unsigned char *hash);
extern __FUNCTION_HEADER__ void SM3_256(unsigned char buf[], int len, unsigned char hash[]);
extern __FUNCTION_HEADER__ void SM3_KDF(unsigned char *Z, unsigned short zlen, unsigned short klen, unsigned char *K);

///************************************************************************
//  File name:    SM9_sv.h
//  Version:      SM9_sv_V1.0
//  Date:         Dec 15,2016
//  Description:  implementation of SM9 signature algorithm and verification algorithm
//                all operations based on BN curve line function
//  Function List:
//        1.bytes128_to_ecn2     //convert 128 bytes into ecn2
//        2.zzn12_ElementPrint   //print all element of struct zzn12
//        3.ecn2_Bytes128_Print  //print 128 bytes of ecn2
//        4.LinkCharZzn12        //link two different types(unsigned char and zzn12)to one(unsigned char)
//        5.Test_Point           //test if the given point is on SM9 curve
//        6.Test_Range           //test if the big x belong to the range[1,N-1]
//        7.SM9_Init             //initiate SM9 curve
//        8.SM9_H1               //function H1 in SM9 standard 5.4.2.2
//        9.SM9_H2               //function H2 in SM9 standard 5.4.2.3
//        10.SM9_GenerateSignKey //generate signed private and public key
//        11.SM9_Sign            //SM9 signature algorithm
//        12.SM9_Verify          //SM9 verification
//        13.SM9_SelfCheck()     //SM9 slef-check

//
// Notes:
// This SM9 implementation source code can be used for academic, non-profit making or non-commercial use only.
// This SM9 implementation is created on MIRACL. SM9 implementation source code provider does not provide MIRACL library, MIRACL license or any permission to use MIRACL library. Any commercial use of MIRACL requires a license which may be obtained from Shamus Software Ltd.

//**************************************************************************/
#define BNLEN 32 // BN curve with 256bit is used in SM9 algorithm
#define SM9_ASK_MEMORY_ERR 0x00000001      // 申请内存失败
#define SM9_H_OUTRANGE 0x00000002          // 签名H不属于[1,N-1]
#define SM9_DATA_MEMCMP_ERR 0x00000003     // 数据对比不一致
#define SM9_MEMBER_ERR 0x00000004          // 群的阶错误
#define SM9_MY_ECAP_12A_ERR 0x00000005     // R-ate对计算出现错误
#define SM9_S_NOT_VALID_G1 0x00000006      // S不属于群G1
#define SM9_G1BASEPOINT_SET_ERR 0x00000007 // G1基点设置错误
#define SM9_G2BASEPOINT_SET_ERR 0x00000008 // G2基点设置错误
#define SM9_L_error 0x00000009             // 参数L错误
#define SM9_GEPUB_ERR 0x0000000A           // 生成公钥错误
#define SM9_GEPRI_ERR 0x0000000B           // 生成私钥错误
#define SM9_SIGN_ERR 0x0000000C            // 签名错误

extern __FUNCTION_HEADER__ BOOL bytes128_to_ecn2(struct SM9_Para *para, unsigned char Ppubs[], ecn2 *res);
extern __FUNCTION_HEADER__ void zzn12_ElementPrint(struct SM9_Para *para, zzn12 x);
extern __FUNCTION_HEADER__ void ecn2_Bytes128_Print(struct SM9_Para *para, ecn2 x);
extern __FUNCTION_HEADER__ void LinkCharZzn12(struct SM9_Para *para, unsigned char *message, int len, zzn12 w, unsigned char *Z, int Zlen);
extern __FUNCTION_HEADER__ int Test_Point(struct SM9_Para *para, epoint *point);
extern __FUNCTION_HEADER__ int Test_Range(struct SM9_Para *para, big x);
extern __FUNCTION_HEADER__ int SM9_Init(struct SM9_Para *para);
extern __FUNCTION_HEADER__ int SM9_H1(struct SM9_Para *para, unsigned char Z[], int Zlen, big n, big h1);
extern __FUNCTION_HEADER__ int SM9_H2(struct SM9_Para *para, unsigned char Z[], int Zlen, big n, big h2);
extern __FUNCTION_HEADER__ int SM9_GenerateSignKey(struct SM9_Para *para, unsigned char hid[], unsigned char *ID, int IDlen, big ks, unsigned char Ppubs[], unsigned char dsa[]);
extern __FUNCTION_HEADER__ int SM9_Sign(struct SM9_Para *para, unsigned char hid[], unsigned char *IDA, unsigned char *message, int len, unsigned char rand[],
                                        unsigned char dsa[], unsigned char Ppub[], unsigned char H[], unsigned char S[]);
extern __FUNCTION_HEADER__ int SM9_Verify(struct SM9_Para *para, unsigned char H[], unsigned char S[], unsigned char hid[], unsigned char *IDA, unsigned char *message, int len,
                                          unsigned char Ppub[]);
extern __FUNCTION_HEADER__ int SM9_SelfCheck(struct SM9_Para *para);
extern __FUNCTION_HEADER__ void set_frobenius_constant(struct SM9_Para *para, zzn2 *X);