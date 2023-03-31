#include "SM9.cuh"

/****************************************************************
Function:       zzn2_pow
Description:    regular zzn2 powering
see zzn2.cpp for details in MIRACL c++ source file
Calls:          MIRACL functions
Called By:      set_frobenius_constant
Input:          zzn2 x,big k
Output:         null
Return:         zzn2
Others:
****************************************************************/
__FUNCTION_HEADER__ zzn2 zzn2_pow(struct SM9_Para *para, zzn2 x, big k)
{
    int i, j, nb, n, nbw, nzs;
    zzn2 res, u2, t[16];

    res.a = mirvar(para->mip, 0);
    res.b = mirvar(para->mip, 0);
    u2.a = mirvar(para->mip, 0);
    u2.b = mirvar(para->mip, 0);

    if (zzn2_iszero(&x))
    {
        zzn2_zero(&res);
        return res;
    }
    if (size(k) == 0)
    {
        zzn2_from_int(para->mip, 1, &res);
        return res;
    }
    if (size(k) == 1)
        return x;

    // Prepare table for windowing
    zzn2_mul(para->mip, &x, &x, &u2);
    t[0].a = mirvar(para->mip, 0);
    t[0].b = mirvar(para->mip, 0);
    zzn2_copy(&x, &t[0]);
    for (i = 1; i<16; i++)
    {
        t[i].a = mirvar(para->mip, 0);
        t[i].b = mirvar(para->mip, 0);
        zzn2_mul(para->mip, &t[i - 1], &u2, &t[i]);
    }

    // Left to right method - with windows
    zzn2_copy(&x, &res);
    nb = logb2(para->mip, k);
    if (nb>1) for (i = nb - 2; i >= 0;)
        {
            //Note new parameter of window_size=5. Default to 5, but reduce to 4 (or even 3) to save RAM
            n = mr_window(para->mip, k, i, &nbw, &nzs, 5);
            for (j = 0; j<nbw; j++) zzn2_mul(para->mip, &res, &res, &res);
            if (n>0) zzn2_mul(para->mip, &res, &t[n / 2], &res);
            i -= nbw;
            if (nzs)
            {
                for (j = 0; j<nzs; j++) zzn2_mul(para->mip, &res, &res, &res);
                i -= nzs;
            }
        }
    return res;
}

/****************************************************************
Function:       q_power_frobenius
Description:    F is frobenius_constant X
see ake12bnx.cpp for details in MIRACL c++ source file
Calls:          MIRACL functions
Called By:      fast_pairing
Input:          ecn2 A,zzn2 F
Output:         zzn2 A
Return:         NULL
Others:
****************************************************************/
__FUNCTION_HEADER__ void q_power_frobenius(struct SM9_Para *para, ecn2 A, zzn2 F)
{
    ecn2_get(para->mip, &A, &(para->fx), &(para->fy), &(para->fz));
    zzn2_copy(&F, &(para->fr));//(para->fr)=F
    if (para->mip->TWIST == MR_SEXTIC_M) zzn2_inv(para->mip, &(para->fr));  // could be precalculated
    zzn2_mul(para->mip, &(para->fr), &(para->fr), &(para->fw));//(para->fw)=(para->fr)*(para->fr)
    zzn2_conj(para->mip, &(para->fx), &(para->fx));
    zzn2_mul(para->mip, &(para->fw), &(para->fx), &(para->fx));
    zzn2_conj(para->mip, &(para->fy), &(para->fy));
    zzn2_mul(para->mip, &(para->fw), &(para->fr), &(para->fw));
    zzn2_mul(para->mip, &(para->fw), &(para->fy), &(para->fy));
    zzn2_conj(para->mip, &(para->fz), &(para->fz));
    ecn2_setxyz(para->mip, &(para->fx), &(para->fy), &(para->fz), &A);
}
/****************************************************************
Function:       line
Description:    Line from A to destination C. Let A=(x,y)
Line Y-slope.X-c=0, through A, so intercept c=y-slope.x
Line Y-slope.X-y+slope.x = (Y-y)-slope.(X-x) = 0
Now evaluate at Q -> return (Qy-y)-slope.(Qx-x)
see ake12bnx.cpp for details in MIRACL c++ source file
Calls:          MIRACL functions,zzn12_init
Called By:      g
Input:          ecn2 A,ecn2 *C,ecn2 *B,zzn2 slope,zzn2 extra,BOOL Doubling,big Qx,big Qy
Output:
Return:         zzn12
Others:
****************************************************************/
__FUNCTION_HEADER__ zzn12 line(struct SM9_Para *para, ecn2 A, ecn2 *C, ecn2 *B, zzn2 slope, zzn2 extra, BOOL Doubling, big Qx, big Qy)
{
    ecn2_getz(para->mip, C, &(para->w7));
    // Thanks to A. Menezes for pointing out this optimization...
    if (Doubling)
    {
        ecn2_get(para->mip, &A, &(para->w1), &(para->w2), &(para->w3));
        zzn2_mul(para->mip, &(para->w3), &(para->w3), &(para->w4)); //Z2=Z*Z

        //X=slope*X-extra
        zzn2_mul(para->mip, &slope, &(para->w1), &(para->w1));
        zzn2_sub(para->mip, &(para->w1), &extra, &(para->w1));

        zzn2_mul(para->mip, &(para->w7), &(para->w4), &(para->w5));

        //(-(Z*Z*slope)*Qx);
        nres(para->mip, Qx, para->b1);
        zzn2_mul(para->mip, &(para->w4), &slope, &(para->w2));
        zzn2_smul(para->mip, &(para->w2), para->b1, &(para->w2));
        zzn2_negate(para->mip, &(para->w2), &(para->w2));

        if (para->mip->TWIST == MR_SEXTIC_M)
        { // "multiplied across" by i to simplify
            zzn2_from_big(para->mip, Qy, &(para->w6));
            zzn2_txx(para->mip, &(para->w6));
            zzn2_mul(para->mip, &(para->w5), &(para->w6), &(para->w6));
            zzn4_from_zzn2s(&(para->w6), &(para->w1), &para->res_g.a);
            zzn2_copy(&(para->w2), &(para->res_g.c.b));
        }
        if (para->mip->TWIST == MR_SEXTIC_D)
        {
            zzn2_smul(para->mip, &(para->w5), Qy, &(para->w6));
            zzn4_from_zzn2s(&(para->w6), &(para->w1), &para->res_g.a);
            zzn2_copy(&(para->w2), &(para->res_g.b.b));
        }
    }
    else
    {   //slope*X-Y*Z
        ecn2_getxy(B, &(para->w1), &(para->w2));
        zzn2_mul(para->mip, &slope, &(para->w1), &(para->w1));
        zzn2_mul(para->mip, &(para->w2), &(para->w7), &(para->w2));
        zzn2_sub(para->mip, &(para->w1), &(para->w2), &(para->w1));

        //(-slope*Qx)
        nres(para->mip, Qx, para->b1);
        zzn2_smul(para->mip, &slope, para->b1, &(para->w3));
        zzn2_negate(para->mip, &(para->w3), &(para->w3));

        if (para->mip->TWIST == MR_SEXTIC_M)
        {
            zzn2_from_big(para->mip, Qy, &(para->w6));
            zzn2_txx(para->mip, &(para->w6));
            zzn2_mul(para->mip, &(para->w7), &(para->w6), &(para->w6));

            zzn4_from_zzn2s(&(para->w6), &(para->w1), &para->res_g.a);
            zzn2_copy(&(para->w3), &(para->res_g.c.b));
        }
        if (para->mip->TWIST == MR_SEXTIC_D)
        {
            zzn2_smul(para->mip, &(para->w7), Qy, &(para->w6));
            zzn4_from_zzn2s(&(para->w6), &(para->w1), &para->res_g.a);
            zzn2_copy(&(para->w3), &(para->res_g.b.b));
        }
    }
    return para->res_g;
}

/****************************************************************
Function:       g
Description:    Add A=A+B  (or A=A+A),Return line function value
see ake12bnx.cpp for details in MIRACL c++ source file
Calls:          MIRACL functions,zzn12_init,line
Called By:
Input:          ecn2 *A,ecn2 *B,big Qx,big Qy
Output:
Return:         zzn12
Others:
****************************************************************/
__FUNCTION_HEADER__ zzn12 g(struct SM9_Para *para, ecn2 *A, ecn2 *B, big Qx, big Qy)
{
    ecn2_copy(A, &(para->P));
    BOOL Doubling = ecn2_add2(para->mip, B, A, &(para->lam), &(para->extra));
    if (A->marker == MR_EPOINT_INFINITY)
    {
        zzn4_from_int(para->mip, 1, &para->res_g.a);
        para->res_g.miller = FALSE;
        para->res_g.unitary = TRUE;
        return para->res_g;
    }
    else
        return line(para, para->P, A, B, para->lam, para->extra, Doubling, Qx, Qy);
}
/****************************************************************
Function:       fast_pairing
Description:    R-ate Pairing G2 x G1 -> GT
P is a point of order q in G1. Q(x,y) is a point of order q in G2.
Note that P is a point on the sextic twist of the curve over Fp^2,
Q(x,y) is a point on the curve over the base field Fp
see ake12bnx.cpp for details in MIRACL c++ source file
Calls:          MIRACL functions,zzn12_init,g,q_power_frobenius
zzn12_copy,zzn12_conj,zzn12_div,zzn12_powq,zzn12_inverse
Called By:      ecap
Input:          ecn2 P,big Qx,big Qy,big x,zzn2 X
Output:         zzn12 *r
Return:         FALSE: r=0
TRUE: correct calculation
Others:
****************************************************************/
#include <time.h>
//__FUNCTION_HEADER__ BOOL fast_pairing(struct SM9_Para *para, ecn2 P, big Qx, big Qy, big x, zzn2 X, zzn12 *r)
//{
//    time_t start, end;
//    start = clock();
//
//    int i, nb;
//    zero(para->pair_res.a.a.a);zero(para->pair_res.a.a.b);zero(para->pair_res.a.b.a);zero(para->pair_res.a.b.b);
//    zero(para->pair_res.b.a.a);zero(para->pair_res.b.a.b);zero(para->pair_res.b.b.a);zero(para->pair_res.b.b.b);
//    zero(para->pair_res.c.a.a);zero(para->pair_res.c.a.b);zero(para->pair_res.c.b.a);zero(para->pair_res.c.b.b);
//    para->pair_res.unitary = FALSE;
//
//    premult(para->mip, x, 6, para->pair_n); incr(para->mip, para->pair_n, 2, para->pair_n);//para->pair_n=(6*x+2);
//    if (mr_compare(x, para->pair_zero)<0)  //x<0
//        negify(para->pair_n, para->pair_n);           //para->pair_n=-(6*x+2);
//
//    ecn2_copy(&P, &para->pair_A);
//    nb = logb2(para->mip, para->pair_n);
//    zzn4_from_int(para->mip, 1, &para->pair_res.a);
//    para->pair_res.unitary = TRUE; //para->pair_res=1
//    // Short Miller loop
//    para->pair_res.miller = TRUE;
//
//    for (i = nb - 2; i >= 0; i--)
//    {
//        zzn12_mul(para,para->pair_res, para->pair_res, &para->pair_res);
//        zzn12_mul(para,para->pair_res, g(para, &para->pair_A, &para->pair_A, Qx, Qy), &para->pair_res);
//        if (mr_testbit(para->mip, para->pair_n, i))
//            zzn12_mul(para,para->pair_res, g(para, &para->pair_A, &P, Qx, Qy), &para->pair_res);
//    }
//    // Combining ideas due to Longa, Aranha et al. and Naehrig
//    ecn2_copy(&P, &para->pair_KA);
//    q_power_frobenius(para, para->pair_KA, X);
//    if (mr_compare(x, para->pair_zero)<0)
//    {
//        ecn2_negate(para->mip, &para->pair_A, &para->pair_A);
//        zzn12_conj(para, &para->pair_res, &para->pair_res);
//    }
//    zzn12_mul(para, para->pair_res, g(para, &para->pair_A, &para->pair_KA, Qx, Qy), &para->pair_res);
//    q_power_frobenius(para, para->pair_KA, X);
//    ecn2_negate(para->mip, &para->pair_KA, &para->pair_KA);
//    zzn12_mul(para, para->pair_res, g(para, &para->pair_A, &para->pair_KA, Qx, Qy), &para->pair_res);
//
//    if (zzn4_iszero(&para->pair_res.a) && zzn4_iszero(&para->pair_res.b) && zzn4_iszero(&para->pair_res.c)) return FALSE;
//
//    // The final exponentiation
//    zzn12_copy(para, &para->pair_res, &para->pair_t0);//para->pair_t0=r;
//    zzn12_conj(para, &para->pair_res, &para->pair_res);
//    zzn12_div(para, para->pair_res, para->pair_t0, &para->pair_res);
//
//    para->pair_res.miller = FALSE; para->pair_res.unitary = FALSE;
//
//    zzn12_copy(para, &para->pair_res, &para->pair_t0);//para->pair_t0=r;
//    zzn12_powq(para, X, &para->pair_res);
//    zzn12_powq(para, X, &para->pair_res);
//    zzn12_mul(para, para->pair_res, para->pair_t0, &para->pair_res);// r^[(p^6-1)*(p^2+1)]
//    para->pair_res.miller = FALSE; para->pair_res.unitary = TRUE;
//
//    // Newer new idea...
//    // See "On the final exponentiation for calculating pairings on ordinary elliptic curves"
//    // Michael Scott and Naomi Benger and Manuel Charlemagne and Luis J. Dominguez Perez and Ezekiel J. Kachisa
//    zzn12_copy(para, &para->pair_res, &para->pair_t0); zzn12_powq(para, X, &para->pair_t0);
//    zzn12_copy(para, &para->pair_t0, &para->pair_x0); zzn12_powq(para, X, &para->pair_x0);   //para->pair_x0=para->pair_t0
//
//    zzn12_mul(para, para->pair_res, para->pair_t0, &para->pair_x1); zzn12_mul(para, para->pair_x0, para->pair_x1, &para->pair_x0);// para->pair_x0*=(para->pair_res*para->pair_t0);
//    zzn12_powq(para, X, &para->pair_x0);
//
//    para->pair_x1 = zzn12_inverse(para, para->pair_res);// just a conjugation!
//    negify(x, para->pair_negify_x);  para->pair_x4 = zzn12_pow(para, para->pair_res, para->pair_negify_x);//para->pair_negify_x=-x   x is sparse.
//    zzn12_copy(para, &para->pair_x4, &para->pair_x3); zzn12_powq(para, X, &para->pair_x3);
//
//    para->pair_x2 = zzn12_pow(para, para->pair_x4, para->pair_negify_x);
//    para->pair_x5 = zzn12_inverse(para, para->pair_x2);
//    para->pair_t0 = zzn12_pow(para, para->pair_x2, para->pair_negify_x);
//
//    zzn12_powq(para, X, &para->pair_x2);
//    zzn12_div(para, para->pair_x4, para->pair_x2, &para->pair_x4);
//
//    zzn12_powq(para, X, &para->pair_x2);
//    zzn12_copy(para, &para->pair_t0, &para->pair_res);// para->pair_res=para->pair_t0
//    zzn12_powq(para, X, &para->pair_res);
//    zzn12_mul(para, para->pair_t0, para->pair_res, &para->pair_t0);
//
//    zzn12_mul(para, para->pair_t0, para->pair_t0, &para->pair_t0); zzn12_mul(para, para->pair_t0, para->pair_x4, &para->pair_t0); zzn12_mul(para, para->pair_t0, para->pair_x5, &para->pair_t0);//para->pair_t0*=para->pair_t0;para->pair_t0*=para->pair_x4;para->pair_t0*=para->pair_x5;
//    zzn12_mul(para, para->pair_x3, para->pair_x5, &para->pair_res); zzn12_mul(para, para->pair_res, para->pair_t0, &para->pair_res);//para->pair_res=para->pair_x3*para->pair_x5;para->pair_res*=para->pair_t0;
//    zzn12_mul(para, para->pair_t0, para->pair_x2, &para->pair_t0);//para->pair_t0*=para->pair_x2;
//    zzn12_mul(para, para->pair_res, para->pair_res, &para->pair_res); zzn12_mul(para, para->pair_res, para->pair_t0, &para->pair_res); zzn12_mul(para, para->pair_res, para->pair_res, &para->pair_res);//para->pair_res*=para->pair_res;
//
//    //��������ʵ������ע�����ݣ��۲���һ����뼴�ɷ��֣�����PDFת������ʱ�����ˣ�����ֱ��ע�͵�����
//    //para->pair_res *= para->pair_t0;
//    //para->pair_res *= para->pair_res;
//
//    zzn12_mul(para, para->pair_res, para->pair_x1, &para->pair_t0);//  para->pair_t0=para->pair_res*para->pair_x1;
//    zzn12_mul(para, para->pair_res, para->pair_x0, &para->pair_res);//para->pair_res*=para->pair_x0;
//    zzn12_mul(para, para->pair_t0, para->pair_t0, &para->pair_t0); zzn12_mul(para, para->pair_t0, para->pair_res, &para->pair_t0);//para->pair_t0*=para->pair_t0;para->pair_t0*=para->pair_res;
//
//    zzn12_copy(para, &para->pair_t0, r);//r= para->pair_t0;
//
//
//    end = clock();
//    printf("双线性对：%lf ms\n", (double) (end - start) / CLOCKS_PER_SEC * 1000);
//
//    return TRUE;
//}

__FUNCTION_HEADER__ BOOL fast_pairing(struct SM9_Para *para, ecn2 P, big Qx, big Qy, big x, zzn2 X, zzn12 *r)
{
    int i, nb;
    big n, zero, negify_x;
    ecn2 A, KA;
    zzn12 t0, x0, x1, x2, x3, x4, x5, res;

    zero = mirvar(para->mip, 0); n = mirvar(para->mip, 0); negify_x = mirvar(para->mip, 0);
    A.x.a = mirvar(para->mip, 0); A.x.b = mirvar(para->mip, 0); A.y.a = mirvar(para->mip, 0); A.y.b = mirvar(para->mip, 0);
    A.z.a = mirvar(para->mip, 0); A.z.b = mirvar(para->mip, 0); A.marker = MR_EPOINT_INFINITY;
    KA.x.a = mirvar(para->mip, 0); KA.x.b = mirvar(para->mip, 0); KA.y.a = mirvar(para->mip, 0); KA.y.b = mirvar(para->mip, 0);
    KA.z.a = mirvar(para->mip, 0); KA.z.b = mirvar(para->mip, 0); KA.marker = MR_EPOINT_INFINITY;
    zzn12_init(para, &t0); zzn12_init(para, &x0); zzn12_init(para, &x1); zzn12_init(para, &x2);
    zzn12_init(para, &x3); zzn12_init(para, &x4); zzn12_init(para, &x5); zzn12_init(para, &res);

    premult(para->mip, x, 6, n); incr(para->mip, n, 2, n);//n=(6*x+2);
    if (mr_compare(x, zero)<0)  //x<0
        negify(n, n);           //n=-(6*x+2);

    ecn2_copy(&P, &A);
    nb = logb2(para->mip, n);
    zzn4_from_int(para->mip, 1, &res.a); res.unitary = TRUE; //res=1
    // Short Miller loop
    res.miller = TRUE;

    for (i = nb - 2; i >= 0; i--)
    {
        zzn12_mul(para,res, res, &res);
        zzn12_mul(para,res, g(para, &A, &A, Qx, Qy), &res);
        if (mr_testbit(para->mip, n, i))
            zzn12_mul(para,res, g(para, &A, &P, Qx, Qy), &res);
    }
    // Combining ideas due to Longa, Aranha et al. and Naehrig
    ecn2_copy(&P, &KA);
    q_power_frobenius(para, KA, X);
    if (mr_compare(x, zero)<0)
    {
        ecn2_negate(para->mip, &A, &A);
        zzn12_conj(para, &res, &res);
    }
    zzn12_mul(para, res, g(para, &A, &KA, Qx, Qy), &res);
    q_power_frobenius(para, KA, X);
    ecn2_negate(para->mip, &KA, &KA);
    zzn12_mul(para, res, g(para, &A, &KA, Qx, Qy), &res);

    if (zzn4_iszero(&res.a) && zzn4_iszero(&res.b) && zzn4_iszero(&res.c)) return FALSE;

    // The final exponentiation
    zzn12_copy(para, &res, &t0);//t0=r;
    zzn12_conj(para, &res, &res);
    zzn12_div(para, res, t0, &res);

    res.miller = FALSE; res.unitary = FALSE;

    zzn12_copy(para, &res, &t0);//t0=r;
    zzn12_powq(para, X, &res);
    zzn12_powq(para, X, &res);
    zzn12_mul(para, res, t0, &res);// r^[(p^6-1)*(p^2+1)]
    res.miller = FALSE; res.unitary = TRUE;

    // Newer new idea...
    // See "On the final exponentiation for calculating pairings on ordinary elliptic curves"
    // Michael Scott and Naomi Benger and Manuel Charlemagne and Luis J. Dominguez Perez and Ezekiel J. Kachisa
    zzn12_copy(para, &res, &t0); zzn12_powq(para, X, &t0);
    zzn12_copy(para, &t0, &x0); zzn12_powq(para, X, &x0);   //x0=t0

    zzn12_mul(para, res, t0, &x1); zzn12_mul(para, x0, x1, &x0);// x0*=(res*t0);
    zzn12_powq(para, X, &x0);

    x1 = zzn12_inverse(para, res);// just a conjugation!
    negify(x, negify_x);  x4 = zzn12_pow(para, res, negify_x);//negify_x=-x   x is sparse.
    zzn12_copy(para, &x4, &x3); zzn12_powq(para, X, &x3);

    x2 = zzn12_pow(para, x4, negify_x);
    x5 = zzn12_inverse(para, x2);
    t0 = zzn12_pow(para, x2, negify_x);

    zzn12_powq(para, X, &x2);
    zzn12_div(para, x4, x2, &x4);

    zzn12_powq(para, X, &x2);
    zzn12_copy(para, &t0, &res);// res=t0
    zzn12_powq(para, X, &res);
    zzn12_mul(para, t0, res, &t0);

    zzn12_mul(para, t0, t0, &t0); zzn12_mul(para, t0, x4, &t0); zzn12_mul(para, t0, x5, &t0);//t0*=t0;t0*=x4;t0*=x5;
    zzn12_mul(para, x3, x5, &res); zzn12_mul(para, res, t0, &res);//res=x3*x5;res*=t0;
    zzn12_mul(para, t0, x2, &t0);//t0*=x2;
    zzn12_mul(para, res, res, &res); zzn12_mul(para, res, t0, &res); zzn12_mul(para, res, res, &res);//res*=res;

    //��������ʵ������ע�����ݣ��۲���һ����뼴�ɷ��֣�����PDFת������ʱ�����ˣ�����ֱ��ע�͵�����
    //res *= t0;
    //res *= res;

    zzn12_mul(para, res, x1, &t0);//  t0=res*x1;
    zzn12_mul(para, res, x0, &res);//res*=x0;
    zzn12_mul(para, t0, t0, &t0); zzn12_mul(para, t0, res, &t0);//t0*=t0;t0*=res;

    zzn12_copy(para, &t0, r);//r= t0;

    return TRUE;
}


/****************************************************************
Function:       ecap
Description:    caculate Rate pairing
see ake12bnx.cpp for details in MIRACL c++ source file
Calls:          MIRACL functions,fast_pairing
Called By:      SM9_Sign,SM9_Verify
Input:          ecn2 P,epoint *Q,big x,zzn2 X
Output:         zzn12 *r
Return:         FALSE: calculation error
TRUE: correct calculation
Others:
****************************************************************/
__FUNCTION_HEADER__ BOOL ecap(struct SM9_Para *para, ecn2 P, epoint *Q, big x, zzn2 X, zzn12 *r)
{
    BOOL Ok;
    big Qx, Qy;
    Qx = mirvar(para->mip, 0); Qy = mirvar(para->mip, 0);

    ecn2_norm(para->mip, &P);
    epoint_get(para->mip, Q, Qx, Qy);

    Ok = fast_pairing(para, P, Qx, Qy, x, X, r);

    if (Ok) return TRUE;

    return FALSE;
}
/****************************************************************
Function:       member
Description:    ctest if a zzn12 element is of order q
test r^q = r^(p+1-t) =1, so test r^p=r^(t-1)
see ake12bnx.cpp for details in MIRACL c++ source file
Calls:          MIRACL functions,zzn12_init,zzn12_copy,zzn12_powq
Called By:      SM9_Sign,SM9_Verify
Input:          zzn12 r,big x,zzn2 F
Output:         NULL
Return:         FALSE: zzn12 element is not of order q
TRUE: zzn12 element is of order q
Others:
****************************************************************/
__FUNCTION_HEADER__ BOOL member(struct SM9_Para *para, zzn12 r, big x, zzn2 F)
{
    convert(para->mip, 6, para->member_six);
    zzn12_copy(para, &r, &para->member_w);//w=r
    zzn12_powq(para, F, &para->member_w);
    r = zzn12_pow(para, r, x);
    r = zzn12_pow(para, r, x);
    r = zzn12_pow(para, r, para->member_six); // t-1=6x^2
    if (zzn4_compare(&para->member_w.a, &r.a) && zzn4_compare(&para->member_w.a, &r.a) && zzn4_compare(&para->member_w.a, &r.a)) return TRUE;
    return FALSE;
}

