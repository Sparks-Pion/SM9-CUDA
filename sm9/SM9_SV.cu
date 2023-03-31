///************************************************************************
//  File name:    SM9_sv.c
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

#include "SM9.cuh"
#define MAX_LEN 1024

/****************************************************************
Function:       bytes128_to_ecn2
Description:    convert 128 bytes into ecn2
Calls:          MIRACL functions
Called By:      SM9_Init
Input:          Ppubs[]
Output:         ecn2 *res
Return:         FALSE: execution error
TRUE: execute correctly
Others:
****************************************************************/
__FUNCTION_HEADER__ BOOL bytes128_to_ecn2(struct SM9_Para *para, unsigned char Ppubs[], ecn2 *res)
{
    zzn2 x, y;
    big a, b;
    ecn2 r;
    r.x.a = mirvar(para->mip, 0);
    r.x.b = mirvar(para->mip, 0);
    r.y.a = mirvar(para->mip, 0);
    r.y.b = mirvar(para->mip, 0);
    r.z.a = mirvar(para->mip, 0);
    r.z.b = mirvar(para->mip, 0);
    r.marker = MR_EPOINT_INFINITY;

    x.a = mirvar(para->mip, 0);
    x.b = mirvar(para->mip, 0);
    y.a = mirvar(para->mip, 0);
    y.b = mirvar(para->mip, 0);
    a = mirvar(para->mip, 0);
    b = mirvar(para->mip, 0);

    bytes_to_big(para->mip, BNLEN, (const char *)Ppubs, b);
    bytes_to_big(para->mip, BNLEN, (const char *)Ppubs + BNLEN, a);
    zzn2_from_bigs(para->mip, a, b, &x);
    bytes_to_big(para->mip, BNLEN, (const char *)Ppubs + BNLEN * 2, b);
    bytes_to_big(para->mip, BNLEN, (const char *)Ppubs + BNLEN * 3, a);
    zzn2_from_bigs(para->mip, a, b, &y);

    return ecn2_set(para->mip, &x, &y, res);
}

/****************************************************************
Function:       zzn12_ElementPrint
Description:    print all element of struct zzn12
Calls:          MIRACL functions
Called By:      SM9_Sign,SM9_Verify
Input:          zzn12 x
Output:         NULL
Return:         NULL
Others:
****************************************************************/
// never used!!!
// void zzn12_ElementPrint(zzn12 x)
// {
//     big tmp;
//     tmp = mirvar(para->mip, 0);

//     redc(para->mip, x.c.b.b, tmp);
//     cotnum(tmp, stdout);
//     redc(para->mip, x.c.b.a, tmp);
//     cotnum(tmp, stdout);
//     redc(para->mip, x.c.a.b, tmp);
//     cotnum(tmp, stdout);
//     redc(para->mip, x.c.a.a, tmp);
//     cotnum(tmp, stdout);
//     redc(para->mip, x.b.b.b, tmp);
//     cotnum(tmp, stdout);
//     redc(para->mip, x.b.b.a, tmp);
//     cotnum(tmp, stdout);
//     redc(para->mip, x.b.a.b, tmp);
//     cotnum(tmp, stdout);
//     redc(para->mip, x.b.a.a, tmp);
//     cotnum(tmp, stdout);
//     redc(para->mip, x.a.b.b, tmp);
//     cotnum(tmp, stdout);
//     redc(para->mip, x.a.b.a, tmp);
//     cotnum(tmp, stdout);
//     redc(para->mip, x.a.a.b, tmp);
//     cotnum(tmp, stdout);
//     redc(para->mip, x.a.a.a, tmp);
//     cotnum(tmp, stdout);
// }

/****************************************************************
Function:       ecn2_Bytes128_Print
Description:    print 128 bytes of ecn2
Calls:          MIRACL functions
Called By:      SM9_Sign,SM9_Verify
Input:          ecn2 x
Output:         NULL
Return:         NULL
Others:
****************************************************************/
// never used!!!
// void ecn2_Bytes128_Print(ecn2 x)
// {
//     big tmp;
//     tmp = mirvar(para->mip, 0);

//     redc(para->mip, x.x.b, tmp);
//     cotnum(tmp, stdout);
//     redc(para->mip, x.x.a, tmp);
//     cotnum(tmp, stdout);
//     redc(para->mip, x.y.b, tmp);
//     cotnum(tmp, stdout);
//     redc(para->mip, x.y.a, tmp);
//     cotnum(tmp, stdout);
// }

/****************************************************************
Function:       LinkCharZzn12
Description:    link two different types(unsigned char and zzn12)to one(unsigned char)
Calls:          MIRACL functions
Called By:      SM9_Sign,SM9_Verify
Input:          message:
len:    length of message
w:      zzn12 element
Output:         Z:      the characters array stored message and w
Zlen:   length of Z
Return:         NULL
Others:
****************************************************************/
__FUNCTION_HEADER__ void LinkCharZzn12(struct SM9_Para *para, unsigned char *message, int len, zzn12 w, unsigned char *Z, int Zlen)
{
    big tmp;

    tmp = mirvar(para->mip, 0);

    memcpy(Z, message, len);
    redc(para->mip, w.c.b.b, tmp);
    big_to_bytes(para->mip, BNLEN, tmp, (char *)(Z + len), 1);
    redc(para->mip, w.c.b.a, tmp);
    big_to_bytes(para->mip, BNLEN, tmp, (char *)(Z + len + BNLEN), 1);
    redc(para->mip, w.c.a.b, tmp);
    big_to_bytes(para->mip, BNLEN, tmp, (char *)(Z + len + BNLEN * 2), 1);
    redc(para->mip, w.c.a.a, tmp);
    big_to_bytes(para->mip, BNLEN, tmp, (char *)(Z + len + BNLEN * 3), 1);
    redc(para->mip, w.b.b.b, tmp);
    big_to_bytes(para->mip, BNLEN, tmp, (char *)(Z + len + BNLEN * 4), 1);
    redc(para->mip, w.b.b.a, tmp);
    big_to_bytes(para->mip, BNLEN, tmp, (char *)(Z + len + BNLEN * 5), 1);
    redc(para->mip, w.b.a.b, tmp);
    big_to_bytes(para->mip, BNLEN, tmp, (char *)(Z + len + BNLEN * 6), 1);
    redc(para->mip, w.b.a.a, tmp);
    big_to_bytes(para->mip, BNLEN, tmp, (char *)(Z + len + BNLEN * 7), 1);
    redc(para->mip, w.a.b.b, tmp);
    big_to_bytes(para->mip, BNLEN, tmp, (char *)(Z + len + BNLEN * 8), 1);
    redc(para->mip, w.a.b.a, tmp);
    big_to_bytes(para->mip, BNLEN, tmp, (char *)(Z + len + BNLEN * 9), 1);
    redc(para->mip, w.a.a.b, tmp);
    big_to_bytes(para->mip, BNLEN, tmp, (char *)(Z + len + BNLEN * 10), 1);
    redc(para->mip, w.a.a.a, tmp);
    big_to_bytes(para->mip, BNLEN, tmp, (char *)(Z + len + BNLEN * 11), 1);
}

/****************************************************************
Function:       Test_Point
Description:    test if the given point is on SM9 curve
Calls:
Called By:      SM9_Verify
Input:          point
Output:         null
Return:         0: success
1: not a valid point on curve

Others:
****************************************************************/
__FUNCTION_HEADER__ int Test_Point(struct SM9_Para *para, epoint *point)
{
    big x, y, x_3, tmp;
    epoint *buf;

    x = mirvar(para->mip, 0);
    y = mirvar(para->mip, 0);
    x_3 = mirvar(para->mip, 0);
    tmp = mirvar(para->mip, 0);
    buf = epoint_init(para->mip);

    // test if y^2=x^3+b
    epoint_get(para->mip, point, x, y);
    power(para->mip, x, 3, para->para_q, x_3); // x_3=x^3 mod p
    multiply(para->mip, x, para->para_a, x);
    divide(para->mip, x, para->para_q, tmp);
    add(para->mip, x_3, x, x); // x=x^3+ax+b
    add(para->mip, x, para->para_b, x);
    divide(para->mip, x, para->para_q, tmp); // x=x^3+ax+b mod p
    power(para->mip, y, 2, para->para_q, y); // y=y^2 mod p
    if (mr_compare(x, y) != 0)
        return 1;

    // test infinity
    ecurve_mult(para->mip, para->N, point, buf);
    if (point_at_infinity(buf) == FALSE)
        return 1;

    return 0;
}

/****************************************************************
Function:       Test_Range
Description:    test if the big x belong to the range[1,n-1]
Calls:
Called By:      SM9_Verify
Input:          big x    ///a miracl data type
Output:         null
Return:         0: success
1: x==n,fail
Others:
****************************************************************/
__FUNCTION_HEADER__ int Test_Range(struct SM9_Para *para, big x)
{
    big one, decr_n;

    one = mirvar(para->mip, 0);
    decr_n = mirvar(para->mip, 0);

    convert(para->mip, 1, one);
    decr(para->mip, para->N, 1, decr_n);

    if ((mr_compare(x, one) < 0) | (mr_compare(x, decr_n) > 0))
        return 1;
    return 0;
}

/****************************************************************
Function:       SM9_Init
Description:    Initiate SM9 curve
Calls:          MIRACL functions
Called By:      SM9_SelfCheck
Input:          null
Output:         null
Return:         0: success;
7: base point P1 error
8: base point P2 error
Others:
****************************************************************/
__FUNCTION_HEADER__ int SM9_Init(struct SM9_Para *para)
{
    big P1_x, P1_y;

    para->mip = mirsys(128, 16);
    ;
//    para->mip->IOBASE = 16;

    para->para_q = mirvar(para->mip, 0);
    para->N = mirvar(para->mip, 0);
    P1_x = mirvar(para->mip, 0);
    P1_y = mirvar(para->mip, 0);
    para->para_a = mirvar(para->mip, 0);
    para->para_b = mirvar(para->mip, 0);
    para->para_t = mirvar(para->mip, 0);
    para->X.a = mirvar(para->mip, 0);
    para->X.b = mirvar(para->mip, 0);
    para->P2.x.a = mirvar(para->mip, 0);
    para->P2.x.b = mirvar(para->mip, 0);
    para->P2.y.a = mirvar(para->mip, 0);
    para->P2.y.b = mirvar(para->mip, 0);
    para->P2.z.a = mirvar(para->mip, 0);
    para->P2.z.b = mirvar(para->mip, 0);
    para->P2.marker = MR_EPOINT_INFINITY;

    // zzn12_init here
    para->Z0.a.a = mirvar(para->mip, 0);
    para->Z0.a.b = mirvar(para->mip, 0);
    para->Z0.b.a = mirvar(para->mip, 0);
    para->Z0.b.b = mirvar(para->mip, 0);
    para->Z0.unitary = FALSE;
    para->Z1.a.a = mirvar(para->mip, 0);
    para->Z1.a.b = mirvar(para->mip, 0);
    para->Z1.b.a = mirvar(para->mip, 0);
    para->Z1.b.b = mirvar(para->mip, 0);
    para->Z1.unitary = FALSE;
    para->Z2.a.a = mirvar(para->mip, 0);
    para->Z2.a.b = mirvar(para->mip, 0);
    para->Z2.b.a = mirvar(para->mip, 0);
    para->Z2.b.b = mirvar(para->mip, 0);
    para->Z2.unitary = FALSE;
    para->Z3.a.a = mirvar(para->mip, 0);
    para->Z3.a.b = mirvar(para->mip, 0);
    para->Z3.b.a = mirvar(para->mip, 0);
    para->Z3.b.b = mirvar(para->mip, 0);
    para->Z3.unitary = FALSE;
    para->T0.a.a = mirvar(para->mip, 0);
    para->T0.a.b = mirvar(para->mip, 0);
    para->T0.b.a = mirvar(para->mip, 0);
    para->T0.b.b = mirvar(para->mip, 0);
    para->T0.unitary = FALSE;
    para->T1.a.a = mirvar(para->mip, 0);
    para->T1.a.b = mirvar(para->mip, 0);
    para->T1.b.a = mirvar(para->mip, 0);
    para->T1.b.b = mirvar(para->mip, 0);
    para->T1.unitary = FALSE;

    // SM9_Sign_init here
    ;
    // SM9_Verify_init here
    ;
    para->w1.a = mirvar(para->mip, 0);
    para->w1.b = mirvar(para->mip, 0);
    para->w2.a = mirvar(para->mip, 0);
    para->w2.b = mirvar(para->mip, 0);
    para->w3.a = mirvar(para->mip, 0);
    para->w3.b = mirvar(para->mip, 0);
    para->w4.a = mirvar(para->mip, 0);
    para->w4.b = mirvar(para->mip, 0);
    para->w5.a = mirvar(para->mip, 0);
    para->w5.b = mirvar(para->mip, 0);
    para->w6.a = mirvar(para->mip, 0);
    para->w6.b = mirvar(para->mip, 0);
    para->w7.a = mirvar(para->mip, 0);
    para->w7.b = mirvar(para->mip, 0);
    para->w8.a = mirvar(para->mip, 0);
    para->w8.b = mirvar(para->mip, 0);
    para->lam.a = mirvar(para->mip, 0);
    para->lam.b = mirvar(para->mip, 0);
    para->extra.a = mirvar(para->mip, 0);
    para->extra.b = mirvar(para->mip, 0);
    para->fx.a = mirvar(para->mip, 0);
    para->fx.b = mirvar(para->mip, 0);
    para->fy.a = mirvar(para->mip, 0);
    para->fy.b = mirvar(para->mip, 0);
    para->fz.a = mirvar(para->mip, 0);
    para->fz.b = mirvar(para->mip, 0);
    para->fw.a = mirvar(para->mip, 0);
    para->fw.b = mirvar(para->mip, 0);
    para->fr.a = mirvar(para->mip, 0);
    para->fr.b = mirvar(para->mip, 0);

    para->b1 = mirvar(para->mip, 0);

    para->P.x.a = mirvar(para->mip, 0);
    para->P.x.b = mirvar(para->mip, 0);
    para->P.y.a = mirvar(para->mip, 0);
    para->P.y.b = mirvar(para->mip, 0);
    para->P.z.a = mirvar(para->mip, 0);
    para->P.z.b = mirvar(para->mip, 0);
    para->P.marker = MR_EPOINT_INFINITY;

    zzn12_init(para, &(para->w));
    zzn12_init(para, &(para->res_g));
    para->member_six = mirvar(para->mip, 0);
    zzn12_init(para, &(para->member_w));

    para->tmp1.a.a = mirvar(para->mip, 0);para->tmp1.a.b = mirvar(para->mip, 0);
    para->tmp1.b.a = mirvar(para->mip, 0);para->tmp1.b.b = mirvar(para->mip, 0);
    para->tmp1.unitary = FALSE;
    para->tmp2.a.a = mirvar(para->mip, 0);para->tmp2.a.b = mirvar(para->mip, 0);
    para->tmp2.b.a = mirvar(para->mip, 0);para->tmp2.b.b = mirvar(para->mip, 0);
    para->tmp2.unitary = FALSE;
//    zzn12_init(para, &(para->inverse_res));
    para->X2.a = mirvar(para->mip, 0);para->X2.b = mirvar(para->mip, 0);
    para->X3.a = mirvar(para->mip, 0);para->X3.b = mirvar(para->mip, 0);
    para->zero = mirvar(para->mip, 0);
    para->tmp000 = mirvar(para->mip, 0);
    para->tmp11 = mirvar(para->mip, 0);

//    para->zero = mirvar(para->mip, 0); para->n = mirvar(para->mip, 0); para->negify_x = mirvar(para->mip, 0);
//    para->A.x.a = mirvar(para->mip, 0); para->A.x.b = mirvar(para->mip, 0); para->A.y.a = mirvar(para->mip, 0); para->A.y.b = mirvar(para->mip, 0);
//    para->A.z.a = mirvar(para->mip, 0); para->A.z.b = mirvar(para->mip, 0); para->A.marker = MR_EPOINT_INFINITY;
//    para->KA.x.a = mirvar(para->mip, 0); para->KA.x.b = mirvar(para->mip, 0); para->KA.y.a = mirvar(para->mip, 0); para->KA.y.b = mirvar(para->mip, 0);
//    para->KA.z.a = mirvar(para->mip, 0); para->KA.z.b = mirvar(para->mip, 0); para->KA.marker = MR_EPOINT_INFINITY;
//    zzn12_init(para, &(para->t0)); zzn12_init(para, &(para->x0)); zzn12_init(para, &(para->x1)); zzn12_init(para, &
//    (para->x2));
//    zzn12_init(para, &(para->x3)); zzn12_init(para, &(para->x4)); zzn12_init(para, &(para->x5)); zzn12_init(para, &
//            (para->res));


    para->P1 = epoint_init(para->mip);
    bytes_to_big(para->mip, BNLEN, (const char *)(para->SM9_q), para->para_q);
    bytes_to_big(para->mip, BNLEN, (const char *)(para->SM9_P1x), P1_x);
    bytes_to_big(para->mip, BNLEN, (const char *)(para->SM9_P1y), P1_y);
    bytes_to_big(para->mip, BNLEN, (const char *)(para->SM9_a), para->para_a);
    bytes_to_big(para->mip, BNLEN, (const char *)(para->SM9_b), para->para_b);
    bytes_to_big(para->mip, BNLEN, (const char *)(para->SM9_N), para->N);
    bytes_to_big(para->mip, BNLEN, (const char *)(para->SM9_t), para->para_t);

    para->mip->TWIST = MR_SEXTIC_M;
    ecurve_init(para->mip, para->para_a, para->para_b, para->para_q, MR_PROJECTIVE); // Initialises GF(q) elliptic curve
    // MR_PROJECTIVE specifying  projective coordinates

    if (!epoint_set(para->mip, P1_x, P1_y, 0, para->P1))
        return SM9_G1BASEPOINT_SET_ERR;

    if (!(bytes128_to_ecn2(para, para->SM9_P2, &(para->P2))))
        return SM9_G2BASEPOINT_SET_ERR;

    set_frobenius_constant(para, &(para->X));

    return 0;
}

/****************************************************************
Function:       SM9_H1
Description:    function H1 in SM9 standard 5.4.2.2
Calls:          MIRACL functions,SM3_KDF
Called By:      SM9_Verify
Input:          Z:
Zlen:the length of Z
n:Frobniues constant X
Output:         h1=H1(Z,Zlen)
Return:         0: success;
1: asking for memory error
Others:
****************************************************************/

__FUNCTION_HEADER__ int ceilfunc(double v)
{
    int vi;
    vi = (int)v;
    if ((v - vi) > 0)
        vi += 1;
    return vi;
}

__FUNCTION_HEADER__ int SM9_H1(struct SM9_Para *para, unsigned char Z[], int Zlen, big n, big h1)
{
    int hlen, i, ZHlen;
    big hh, i256, tmp, n1;
    unsigned char *ZH = NULL, *ha = NULL;

    hh = mirvar(para->mip, 0);
    i256 = mirvar(para->mip, 0);
    tmp = mirvar(para->mip, 0);
    n1 = mirvar(para->mip, 0);
    convert(para->mip, 1, i256);
    ZHlen = Zlen + 1;

    hlen = (int)ceilfunc((5.0 * logb2(para->mip, n)) / 32.0);
    decr(para->mip, n, 1, n1);
    ZH = (unsigned char *)malloc(sizeof(char) * (ZHlen + 1));
    if (ZH == NULL)
        return SM9_ASK_MEMORY_ERR;
    memcpy(ZH + 1, Z, Zlen);
    ZH[0] = 0x01;
    ha = (unsigned char *)malloc(sizeof(char) * (hlen + 1));
    if (ha == NULL)
        return SM9_ASK_MEMORY_ERR;
    SM3_KDF(ZH, ZHlen, hlen, ha);

    for (i = hlen - 1; i >= 0; i--) // key[从大到小]
    {
        premult(para->mip, i256, ha[i], tmp);
        add(para->mip, hh, tmp, hh);
        premult(para->mip, i256, 256, i256);
        divide(para->mip, i256, n1, tmp);
        divide(para->mip, hh, n1, tmp);
    }
    incr(para->mip, hh, 1, h1);
    free(ZH);
    free(ha);
    return 0;
}
/****************************************************************
Function:       SM9_H2
Description:    function H2 in SM9 standard 5.4.2.3
Calls:          MIRACL functions,SM3_KDF
Called By:      SM9_Sign,SM9_Verify
Input:          Z:
Zlen:the length of Z
n:Frobniues constant X
Output:         h2=H2(Z,Zlen)
Return:         0: success;
1: asking for memory error
Others:
****************************************************************/
__FUNCTION_HEADER__ int SM9_H2(struct SM9_Para *para, unsigned char Z[], int Zlen, big n, big h2)
{
    int hlen, ZHlen, i;
    big hh, i256, tmp, n1;
    unsigned char *ZH = NULL, *ha = NULL;

    hh = mirvar(para->mip, 0);
    i256 = mirvar(para->mip, 0);
    tmp = mirvar(para->mip, 0);
    n1 = mirvar(para->mip, 0);
    convert(para->mip, 1, i256);
    ZHlen = Zlen + 1;

    hlen = (int)ceilfunc((5.0 * logb2(para->mip, n)) / 32.0);
    decr(para->mip, n, 1, n1);
    ZH = (unsigned char *)malloc(sizeof(char) * (ZHlen + 1));
    if (ZH == NULL)
        return SM9_ASK_MEMORY_ERR;
    memcpy(ZH + 1, Z, Zlen);
    ZH[0] = 0x02;
    ha = (unsigned char *)malloc(sizeof(char) * (hlen + 1));
    if (ha == NULL)
        return SM9_ASK_MEMORY_ERR;
    SM3_KDF(ZH, ZHlen, hlen, ha);

    for (i = hlen - 1; i >= 0; i--) // key[从大到小]
    {
        premult(para->mip, i256, ha[i], tmp);
        add(para->mip, hh, tmp, hh);
        premult(para->mip, i256, 256, i256);
        divide(para->mip, i256, n1, tmp);
        divide(para->mip, hh, n1, tmp);
    }
    incr(para->mip, hh, 1, h2);
    free(ZH);
    free(ha);
    return 0;
}

/****************************************************************
Function:       SM9_GenerateSignKey
Description:    Generate Signed key
Calls:          MIRACL functions,SM9_H1,xgcd,ecn2_Bytes128_Print
Called By:      SM9_SelfCheck
Input:          hid:0x01
ID:identification
IDlen:the length of ID
ks:master private key used to generate signature public key and private key
Output:         Ppub:signature public key
dSA: signature private key
Return:         0: success;
1: asking for memory error
Others:
****************************************************************/
__FUNCTION_HEADER__ int SM9_GenerateSignKey(struct SM9_Para *para, unsigned char hid[], unsigned char *ID, int IDlen, big ks, unsigned char Ppubs[], unsigned char dsa[])
{
    big h1, t1, t2, rem, xdSA, ydSA, tmp;
    unsigned char *Z = NULL;
    int Zlen = IDlen + 1, buf;
    ecn2 Ppub;
    epoint *dSA;

    h1 = mirvar(para->mip, 0);
    t1 = mirvar(para->mip, 0);
    t2 = mirvar(para->mip, 0);
    rem = mirvar(para->mip, 0);
    tmp = mirvar(para->mip, 0);
    xdSA = mirvar(para->mip, 0);
    ydSA = mirvar(para->mip, 0);
    dSA = epoint_init(para->mip);
    Ppub.x.a = mirvar(para->mip, 0);
    Ppub.x.b = mirvar(para->mip, 0);
    Ppub.y.a = mirvar(para->mip, 0);
    Ppub.y.b = mirvar(para->mip, 0);
    Ppub.z.a = mirvar(para->mip, 0);
    Ppub.z.b = mirvar(para->mip, 0);
    Ppub.marker = MR_EPOINT_INFINITY;

    Z = (unsigned char *)malloc(sizeof(char) * (Zlen + 1));
    memcpy(Z, ID, IDlen);
    memcpy(Z + IDlen, hid, 1);

    buf = SM9_H1(para, Z, Zlen, para->N, h1);
    // 在Mac上运行，每次运行SM3，哈希值都不同！

    if (buf != 0)
        return buf;
    add(para->mip, h1, ks, t1); // t1=H1(IDA||hid,N)+ks

    // 少一步：若t1=0，则重新生成主私钥，这里因为主私钥是指定的，所以就少了该步
    xgcd(para->mip, t1, para->N, t1, t1, t1); // t1=t1(-1)
    multiply(para->mip, ks, t1, t2);
    divide(para->mip, t2, para->N, rem); // t2=ks*t1(-1)

    // dSA=[t2]P1
    ecurve_mult(para->mip, t2, para->P1, dSA);

    // Ppub=[ks]P2
    ecn2_copy(&(para->P2), &Ppub);
    ecn2_mul(para->mip, ks, &Ppub);
    //    printf("\n*********************主私钥ks为：\n");
    //    cotnum(ks, stdout);

    //    printf("\n**********************主公钥Ppubs=[ks]P2为：\n");
    //    ecn2_Bytes128_Print(Ppub);

    //    printf("\n*********************用户私钥dsA=[t2]P1=(xdA, ydA)为：\n");
    //    epoint_get(para->mip, dSA, xdSA, ydSA);
    //    cotnum(xdSA, stdout); cotnum(ydSA, stdout);
    epoint_get(para->mip, dSA, xdSA, ydSA);
    big_to_bytes(para->mip, BNLEN, xdSA, (char *)dsa, 1);
    big_to_bytes(para->mip, BNLEN, ydSA, (char *)(dsa + BNLEN), 1);

    redc(para->mip, Ppub.x.b, tmp);
    big_to_bytes(para->mip, BNLEN, tmp, (char *)Ppubs, 1);
    redc(para->mip, Ppub.x.a, tmp);
    big_to_bytes(para->mip, BNLEN, tmp, (char *)(Ppubs + BNLEN), 1);
    redc(para->mip, Ppub.y.b, tmp);
    big_to_bytes(para->mip, BNLEN, tmp, (char *)(Ppubs + BNLEN * 2), 1);
    redc(para->mip, Ppub.y.a, tmp);
    big_to_bytes(para->mip, BNLEN, tmp, (char *)(Ppubs + BNLEN * 3), 1);

    free(Z);
    return 0;
}

/****************************************************************
Function:       SM9_Sign
Description:    SM9 signature algorithm
Calls:          MIRACL functions,zzn12_init(),ecap(),member(),zzn12_ElementPrint(),
zzn12_pow(),LinkCharZzn12(),SM9_H2()
Called By:      SM9_SelfCheck()
Input:
hid:0x01
IDA          //identification of userA
message      //the message to be signed
len          //the length of message
rand         //a random number K lies in [1,N-1]
dSA          //signature private key
Ppubs        //signature public key

Output:         H,S        //signature result
Return:         0: success
1: asking for memory error
4: element is out of order q
5: R-ate calculation error
9: parameter L error
Others:
****************************************************************/
__FUNCTION_HEADER__ int SM9_Sign(struct SM9_Para *para, unsigned char hid[], unsigned char *IDA, unsigned char *message, int len, unsigned char rand[],
                                 unsigned char dsa[], unsigned char Ppub[], unsigned char H[], unsigned char S[])
{
    big h1, r, h, l, xdSA, ydSA;
    big xS, yS, tmp, zero;
    zzn12 g, w;
    epoint *s, *dSA;
    ecn2 Ppubs;
    int Zlen, buf;
    unsigned char *Z = NULL;

    // initiate
    h1 = mirvar(para->mip, 0);
    r = mirvar(para->mip, 0);
    h = mirvar(para->mip, 0);
    l = mirvar(para->mip, 0);
    tmp = mirvar(para->mip, 0);
    zero = mirvar(para->mip, 0);
    xS = mirvar(para->mip, 0);
    yS = mirvar(para->mip, 0);
    xdSA = mirvar(para->mip, 0);
    ydSA = mirvar(para->mip, 0);
    s = epoint_init(para->mip);
    dSA = epoint_init(para->mip);
    Ppubs.x.a = mirvar(para->mip, 0);
    Ppubs.x.b = mirvar(para->mip, 0);
    Ppubs.y.a = mirvar(para->mip, 0);
    Ppubs.y.b = mirvar(para->mip, 0);
    Ppubs.z.a = mirvar(para->mip, 0);
    Ppubs.z.b = mirvar(para->mip, 0);
    Ppubs.marker = MR_EPOINT_INFINITY;
    zzn12_init(para, &g);
    zzn12_init(para, &w);

    bytes_to_big(para->mip, BNLEN, (char *)rand, r);
    bytes_to_big(para->mip, BNLEN, (char *)dsa, xdSA);
    bytes_to_big(para->mip, BNLEN, (char *)(dsa + BNLEN), ydSA);
    epoint_set(para->mip, xdSA, ydSA, 0, dSA);
    bytes128_to_ecn2(para, Ppub, &Ppubs);

    // Step1:g = e(P1, Ppub-s)
    if (!ecap(para, Ppubs, para->P1, para->para_t, para->X, &g))
        return SM9_MY_ECAP_12A_ERR;
    // test if a ZZn12 element is of order q
    // if (!member(para, g, para->para_t, para->X))
    //     return SM9_MEMBER_ERR;

    //    printf("\n***********************g=e(P1,Ppubs):****************************\n");
    //    zzn12_ElementPrint(g);

    // Step2:calculate w=g(r)
    //    printf("\n***********************随机数 r:********************************\n");
    //    cotnum(r, stdout);
    w = zzn12_pow(para, g, r);
    //    printf("\n***************************w=g^r:**********************************\n");
    //    zzn12_ElementPrint(w);

    // Step3:calculate h=H2(M||w,N)
    Zlen = len + 32 * 12;
    Z = (unsigned char *)malloc(sizeof(char) * (Zlen + 1));
    if (Z == NULL)
        return SM9_ASK_MEMORY_ERR;

    LinkCharZzn12(para, message, len, w, Z, Zlen); // M||w
    buf = SM9_H2(para, Z, Zlen, para->N, h);
    if (buf != 0)
        return buf;
    //    printf("\n****************************h:*************************************\n");
    //    cotnum(h, stdout);

    // Step4:l=(r-h)mod N
    subtract(para->mip, r, h, l);       //(r-h)
    divide(para->mip, l, para->N, tmp); //(r-h)%N
    while (mr_compare(l, zero) < 0)
        add(para->mip, l, para->N, l);
    if (mr_compare(l, zero) == 0)
        return SM9_L_error;
    //    printf("\n**************************l=(r-h)mod N:****************************\n");
    //    cotnum(l, stdout);

    // Step5:S=[l]dSA=(xS,yS)
    ecurve_mult(para->mip, l, dSA, s); // 多倍点乘
    epoint_get(para->mip, s, xS, yS);
    //    printf("\n**************************S=[l]dSA=(xS,yS):*************************\n");
    //    cotnum(xS, stdout); cotnum(yS, stdout);

    big_to_bytes(para->mip, 32, h, (char *)H, 1);
    big_to_bytes(para->mip, 32, xS, (char *)S, 1);
    big_to_bytes(para->mip, 32, yS, (char *)(S + 32), 1);

    free(Z);
    return 0;
}
/****************************************************************
Function:       SM9_Verify
Description:    SM9 signature verification algorithm
Calls:          MIRACL functions,zzn12_init(),Test_Range(),Test_Point(),
ecap(),member(),zzn12_ElementPrint(),SM9_H1(),SM9_H2()
Called By:      SM9_SelfCheck()
Input:
H,S          //signature result used to be verified
hid          //identification
IDA          //identification of userA
message      //the message to be signed
len          //the length of message
Ppubs        //signature public key

Output:         NULL
Return:         0: success
1: asking for memory error
2: H is not in the range[1,N-1]
6: S is not on the SM9 curve
4: element is out of order q
5: R-ate calculation error
3: h2!=h,comparison error
Others:
****************************************************************/
__FUNCTION_HEADER__ int SM9_Verify(struct SM9_Para *para, unsigned char H[], unsigned char S[], unsigned char hid[], unsigned char *IDA, unsigned char *message, int len,
                                   unsigned char Ppub[])
{
    big h, xS, yS, h1, h2;
    epoint *S1;
    zzn12 g, t, u, w;
    ecn2 P, Ppubs;
    int Zlen1, Zlen2, buf;
    unsigned char *Z1 = NULL, *Z2 = NULL;

    h = mirvar(para->mip, 0);
    h1 = mirvar(para->mip, 0);
    h2 = mirvar(para->mip, 0);
    xS = mirvar(para->mip, 0);
    yS = mirvar(para->mip, 0);
    P.x.a = mirvar(para->mip, 0);
    P.x.b = mirvar(para->mip, 0);
    P.y.a = mirvar(para->mip, 0);
    P.y.b = mirvar(para->mip, 0);
    P.z.a = mirvar(para->mip, 0);
    P.z.b = mirvar(para->mip, 0);
    P.marker = MR_EPOINT_INFINITY;
    Ppubs.x.a = mirvar(para->mip, 0);
    Ppubs.x.b = mirvar(para->mip, 0);
    Ppubs.y.a = mirvar(para->mip, 0);
    Ppubs.y.b = mirvar(para->mip, 0);
    Ppubs.z.a = mirvar(para->mip, 0);
    Ppubs.z.b = mirvar(para->mip, 0);
    Ppubs.marker = MR_EPOINT_INFINITY;
    S1 = epoint_init(para->mip);
    zzn12_init(para, &g), zzn12_init(para, &t);
    zzn12_init(para, &u);
    zzn12_init(para, &w);

    bytes_to_big(para->mip, BNLEN, (char *)H, h);
    bytes_to_big(para->mip, BNLEN, (char *)S, xS);
    bytes_to_big(para->mip, BNLEN, (char *)(S + BNLEN), yS);
    bytes128_to_ecn2(para, Ppub, &Ppubs);

    // Step 1:test if h in the rangge [1,N-1]
    // if (Test_Range(para, h)) // 验证整数h是都在区间内
    //     return SM9_H_OUTRANGE;

    // Step 2:test if S is on G1
    epoint_set(para->mip, xS, yS, 0, S1); // 验证点是否在曲线上
    // if (Test_Point(para, S1))
    //     return SM9_S_NOT_VALID_G1;

    // Step3:g = e(P1, Ppub-s)
    if (!ecap(para, Ppubs, para->P1, para->para_t, para->X, &g))
        return SM9_MY_ECAP_12A_ERR;
    // test if a ZZn12 element is of order q
    // if (!member(para, g, para->para_t, para->X))
    //     return SM9_MEMBER_ERR;

    //    printf("\n***********************g=e(P1,Ppubs):****************************\n");
    //    zzn12_ElementPrint(g);

    // Step4:calculate t=g^h
    t = zzn12_pow(para, g, h);
    //    printf("\n***************************w=g^h:**********************************\n");
    //    zzn12_ElementPrint(t);

    // Step5:calculate h1=H1(IDA||hid,N)
    Zlen1 = cuda_strlen((const char *)IDA) + 1;
    Z1 = (unsigned char *)malloc(sizeof(char) * (Zlen1 + 1));
    if (Z1 == NULL)
        return SM9_ASK_MEMORY_ERR;

    memcpy(Z1, IDA, cuda_strlen((const char *)IDA));
    memcpy(Z1 + cuda_strlen((const char *)IDA), hid, 1);
    buf = SM9_H1(para, Z1, Zlen1, para->N, h1);
    if (buf != 0)
        return buf;
    //    printf("\n****************************h1:**********************************\n");
    //    cotnum(h1, stdout);

    // Step6:P=[h1]P2+Ppubs
    ecn2_copy(&(para->P2), &P);
    ecn2_mul(para->mip, h1, &P);
    ecn2_add(para->mip, &Ppubs, &P);

    // Step7:u=e(S1,P)
    if (!ecap(para, P, S1, para->para_t, para->X, &u))
        return SM9_MY_ECAP_12A_ERR;
    // test if a ZZn12 element is of order q
    // if (!member(para, u, para->para_t, para->X))
    //     return SM9_MEMBER_ERR;
    //    printf("\n************************** u=e(S1,P):*****************************\n");
    //    zzn12_ElementPrint(u);

    // Step8:w=u*t
    zzn12_mul(para, u, t, &w);
    //    printf("\n*************************  w=u*t: **********************************\n");
    //    zzn12_ElementPrint(w);

    // Step9:h2=H2(M||w,N)
    Zlen2 = len + 32 * 12;
    Z2 = (unsigned char *)malloc(sizeof(char) * (Zlen2 + 1));
    if (Z2 == NULL)
        return SM9_ASK_MEMORY_ERR;

    LinkCharZzn12(para, message, len, w, Z2, Zlen2);
    buf = SM9_H2(para, Z2, Zlen2, para->N, h2);
    if (buf != 0)
        return buf;
    //    printf("\n**************************** h2:***********************************\n");
    //    cotnum(h2, stdout);

    free(Z1);
    free(Z2);

    // for (int i = P.x.a->len - 1; i >= 0; i--)
    //     printf("%lX ", P.x.a->w[i]);
    // printf("\n");
    // for (int i = P.x.b->len - 1; i >= 0; i--)
    //     printf("%lX ", P.x.b->w[i]);
    // printf("\n");
    // for (int i = u.a.a.a->len - 1; i >= 0; i--)
    //     printf("%lX ", u.a.a.a->w[i]);
    // printf("\n");
    // for (int i = u.a.a.b->len - 1; i >= 0; i--)
    //     printf("%lX ", u.a.a.b->w[i]);
    // printf("\n");
    // for (int i = u.a.b.a->len - 1; i >= 0; i--)
    //     printf("%lX ", u.a.b.a->w[i]);
    // printf("\n");
    // for (int i = u.a.b.b->len - 1; i >= 0; i--)
    //     printf("%lX ", u.a.b.b->w[i]);
    // printf("\n");
    // for (int i = h1->len - 1; i >= 0; i--)
    //     printf("%lX ", h1->w[i]);
    // printf("\n");
    // for (int i = h2->len - 1; i >= 0; i--)
    //     printf("%lX ", h2->w[i]);
    // printf("\n");
    // for (int i = h->len - 1; i >= 0; i--)
    //     printf("%lX ", h->w[i]);
    // printf("\n");
    // for (int i = xS->len - 1; i >= 0; i--)
    //     printf("%lX ", xS->w[i]);
    // printf("\n");
    // for (int i = yS->len - 1; i >= 0; i--)
    //     printf("%lX ", yS->w[i]);
    // printf("\n");
    //    printf("\n签名验证结果：\n");
    if (mr_compare(h2, h) != 0)
    {
        printf("h 不等于 h2，验证失败！\n");
        return SM9_DATA_MEMCMP_ERR;
    }
    else
        //        printf("h 等于 h2，验证成功！\n\n");

        return 0;
}

/****************************************************************
Function:       set_frobenius_constant
Description:    calculate frobenius_constant X
see ake12bnx.cpp for details in MIRACL c++ source file
Calls:          MIRACL functions,zzn2_pow
Called By:      SM9_init
Input:          NULL
Output:         zzn2 *X
Return:         NULL
Others:
****************************************************************/
__FUNCTION_HEADER__ void set_frobenius_constant(struct SM9_Para *para, zzn2 *X)
{
    big p, zero, one, two;
    p = mirvar(para->mip, 0);
    zero = mirvar(para->mip, 0);
    one = mirvar(para->mip, 0);
    two = mirvar(para->mip, 0);

    convert(para->mip, 0, zero);
    convert(para->mip, 1, one);
    convert(para->mip, 2, two);

    para->mip = para->mip;
    copy(para->mip->modulus, p);

    switch (para->mip->pmod8)
    {
    case 5:
        zzn2_from_bigs(para->mip, zero, one, X); // = (sqrt(-2)^(p-1)/2
        break;
    case 3:
        zzn2_from_bigs(para->mip, one, one, X); // = (1+sqrt(-1))^(p-1)/2
        break;
    case 7:
        zzn2_from_bigs(para->mip, two, one, X); // = (2+sqrt(-1))^(p-1)/2
    default:
        break;
    }

    decr(para->mip, p, 1, p);
    subdiv(para->mip, p, 6, p);

    *X = zzn2_pow(para, *X, p);
}

/****************************************************************
Function:       SM9_SelfCheck
Description:    SM9 self check
Calls:          MIRACL functions,SM9_Init(),SM9_GenerateSignKey(),
SM9_Sign,SM9_Verify
Called By:
Input: 要签名的字符串
Output:
Return:         0: self-check success
1: asking for memory error
2: H is not in the range[1,N-1]
3: h2!=h,comparison error
4: element is out of order q
5: R-ate calculation error
6: S is not on the SM9 curve
7: base point P1 error
8: base point P2 error
9: parameter L error
A: public key generated error
B: private key generated error
C: signature result error
Others:
****************************************************************/
#include <stdio.h>
#define TEST_CNT 1
__FUNCTION_HEADER__ int SM9_SelfCheck(struct SM9_Para *para)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // time_t start, end;

    // the master private key
    unsigned char dA[32] = {0x00, 0x01, 0x30, 0xE7, 0x84, 0x59, 0xD7, 0x85, 0x45, 0xCB, 0x54, 0xC5, 0x87, 0xE0, 0x2C, 0xF4,
                            0x80, 0xCE, 0x0B, 0x66, 0x34, 0x0F, 0x31, 0x9F, 0x34, 0x8A, 0x1D, 0x5B, 0x1F, 0x2D, 0xC5, 0xF4};

    unsigned char rand[32] = {0x00, 0x03, 0x3C, 0x86, 0x16, 0xB0, 0x67, 0x04, 0x81, 0x32, 0x03, 0xDF, 0xD0, 0x09, 0x65, 0x02,
                              0x2E, 0xD1, 0x59, 0x75, 0xC6, 0x62, 0x33, 0x7A, 0xED, 0x64, 0x88, 0x35, 0xDC, 0x4B, 0x1C, 0xBE};

    unsigned char h[32], S[64]; // Signature
    unsigned char Ppub[128], dSA[64];

    // 提前算好的对于预设签名字符串“Chinese IBS standard”的签名值
    // 作用：签名后会验证h和std_h，S和std_S，若得到的签名值与这两个算好的不同，则说明签名出错
    unsigned char std_h[32] = {0x82, 0x3C, 0x4B, 0x21, 0xE4, 0xBD, 0x2D, 0xFE, 0x1E, 0xD9, 0x2C, 0x60, 0x66, 0x53, 0xE9, 0x96,
                               0x66, 0x85, 0x63, 0x15, 0x2F, 0xC3, 0x3F, 0x55, 0xD7, 0xBF, 0xBB, 0x9B, 0xD9, 0x70, 0x5A, 0xDB};
    unsigned char std_S[64] = {0x73, 0xBF, 0x96, 0x92, 0x3C, 0xE5, 0x8B, 0x6A, 0xD0, 0xE1, 0x3E, 0x96, 0x43, 0xA4, 0x06, 0xD8,
                               0xEB, 0x98, 0x41, 0x7C, 0x50, 0xEF, 0x1B, 0x29, 0xCE, 0xF9, 0xAD, 0xB4, 0x8B, 0x6D, 0x59, 0x8C,
                               0x85, 0x67, 0x12, 0xF1, 0xC2, 0xE0, 0x96, 0x8A, 0xB7, 0x76, 0x9F, 0x42, 0xA9, 0x95, 0x86, 0xAE,
                               0xD1, 0x39, 0xD5, 0xB8, 0xB3, 0xE1, 0x58, 0x91, 0x82, 0x7C, 0xC2, 0xAC, 0xED, 0x9B, 0xAA, 0x05};
    unsigned char std_Ppub[128] = {0x9F, 0x64, 0x08, 0x0B, 0x30, 0x84, 0xF7, 0x33, 0xE4, 0x8A, 0xFF, 0x4B, 0x41, 0xB5, 0x65, 0x01,
                                   0x1C, 0xE0, 0x71, 0x1C, 0x5E, 0x39, 0x2C, 0xFB, 0x0A, 0xB1, 0xB6, 0x79, 0x1B, 0x94, 0xC4, 0x08,
                                   0x29, 0xDB, 0xA1, 0x16, 0x15, 0x2D, 0x1F, 0x78, 0x6C, 0xE8, 0x43, 0xED, 0x24, 0xA3, 0xB5, 0x73,
                                   0x41, 0x4D, 0x21, 0x77, 0x38, 0x6A, 0x92, 0xDD, 0x8F, 0x14, 0xD6, 0x56, 0x96, 0xEA, 0x5E, 0x32,
                                   0x69, 0x85, 0x09, 0x38, 0xAB, 0xEA, 0x01, 0x12, 0xB5, 0x73, 0x29, 0xF4, 0x47, 0xE3, 0xA0, 0xCB,
                                   0xAD, 0x3E, 0x2F, 0xDB, 0x1A, 0x77, 0xF3, 0x35, 0xE8, 0x9E, 0x14, 0x08, 0xD0, 0xEF, 0x1C, 0x25,
                                   0x41, 0xE0, 0x0A, 0x53, 0xDD, 0xA5, 0x32, 0xDA, 0x1A, 0x7C, 0xE0, 0x27, 0xB7, 0xA4, 0x6F, 0x74,
                                   0x10, 0x06, 0xE8, 0x5F, 0x5C, 0xDF, 0xF0, 0x73, 0x0E, 0x75, 0xC0, 0x5F, 0xB4, 0xE3, 0x21, 0x6D};
    unsigned char
        std_dSA[64] = {0xA5, 0x70, 0x2F, 0x05, 0xCF, 0x13, 0x15, 0x30, 0x5E, 0x2D, 0x6E, 0xB6, 0x4B, 0x0D, 0xEB, 0x92,
                       0x3D, 0xB1, 0xA0, 0xBC, 0xF0, 0xCA, 0xFF, 0x90, 0x52, 0x3A, 0xC8, 0x75, 0x4A, 0xA6, 0x98, 0x20,
                       0x78, 0x55, 0x9A, 0x84, 0x44, 0x11, 0xF9, 0x82, 0x5C, 0x10, 0x9F, 0x5E, 0xE3, 0xF5, 0x2D, 0x72,
                       0x0D, 0xD0, 0x17, 0x85, 0x39, 0x2A, 0x72, 0x7B, 0xB1, 0x55, 0x69, 0x52, 0xB2, 0xB0, 0x13, 0xD3};

    unsigned char hid[] = {0x01};
    unsigned char *IDA = (unsigned char *)"Alice";
    unsigned char *message = (unsigned char *)"Chinese IBS standard"; // the message to be signed
    // unsigned char *message = "SM9 Identity-based cryptographic algorithms";
    int mlen = cuda_strlen((const char *)message), tmp; // the length of message
    big ks;

    // printf("ID为：%s\n", IDA);
    // printf("消息为：%s\n", message);

    tmp = SM9_Init(para);

    if (tmp != 0)
        return tmp;
    ks = mirvar(para->mip, 0);

    bytes_to_big(para->mip, 32, (const char *)dA, ks);

    printf("\n SM9 密钥生成 开始 - %d \n", index);

    //  start = clock();

    for (int i = 0; i < TEST_CNT; i++)
        tmp = SM9_GenerateSignKey(para, hid, IDA, cuda_strlen((const char *)IDA), ks, Ppub, dSA);
    if (tmp != 0)
        return tmp;
    if (cuda_memcmp(Ppub, std_Ppub, 128) != 0)
        return SM9_GEPUB_ERR;
    if (cuda_memcmp(dSA, std_dSA, 64) != 0)
        return SM9_GEPRI_ERR;

    //  end = clock();
    //  printf("SM9密钥%d次平均生成时间为：%lf ms\n", TEST_CNT, (double)(end - start) / CLOCKS_PER_SEC * 1000 / TEST_CNT);
    printf("\n SM9 密钥生成 结束 - %d \n", index);

    printf("\n SM9签名 开始 - %d \n", index);

    //  start = clock();
    for (int i = 0; i < TEST_CNT; i++)
        tmp = SM9_Sign(para, hid, IDA, message, mlen, rand, dSA, Ppub, h, S);
    if (tmp != 0)
        return tmp;

    //  end = clock();
    //  printf("SM9签名%d次平均时间为：%lf ms\n", TEST_CNT, (double)(end - start) / CLOCKS_PER_SEC * 1000 / TEST_CNT);
    printf("\n SM9签名 结束 - %d \n", index);

    // 此两句为验证签名值与预设字符串“Chinese IBS standard”得到的签名值是否相同
    // 若从外界输入签名字符串或文件，则注释掉即可
    if (cuda_memcmp(h, std_h, 32) != 0)
        return SM9_SIGN_ERR;
    if (cuda_memcmp(S, std_S, 64) != 0) // 为何消息一改动，此处就会返回SM9_SIGN_ERR？
        return SM9_SIGN_ERR;

    printf("\n SM9签名验证 开始 - %d \n", index);

    // //  start = clock();
    for (int i = 0; i < TEST_CNT; i++)
        tmp = SM9_Verify(para, h, S, hid, IDA, message, mlen, Ppub);
    if (tmp != 0)
        return tmp;

    //  end = clock();
    //  printf("SM9签名%d次平均验证时间为：%lf ms\n", TEST_CNT, (double)(end - start) / CLOCKS_PER_SEC * 1000 / TEST_CNT);

    printf("\n SM9签名验证 结束 - %d \n", index);

    return 0;
}