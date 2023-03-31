#include "SM9.cuh"

/****************************************************************
Function:       zzn12_init
Description:    Initiate struct zzn12
Calls:          MIRACL functions
Called By:
Input:          zzn12 *x
Output:         null
Return:         null
Others:
****************************************************************/

__FUNCTION_HEADER__ void zzn12_init(struct SM9_Para *para, zzn12 *x) {
    x->a.a.a = mirvar(para->mip, 0);
    x->a.a.b = mirvar(para->mip, 0);
    x->a.b.a = mirvar(para->mip, 0);
    x->a.b.b = mirvar(para->mip, 0);
    x->a.unitary = FALSE;
    x->b.a.a = mirvar(para->mip, 0);
    x->b.a.b = mirvar(para->mip, 0);
    x->b.b.a = mirvar(para->mip, 0);
    x->b.b.b = mirvar(para->mip, 0);
    x->b.unitary = FALSE;
    x->c.a.a = mirvar(para->mip, 0);
    x->c.a.b = mirvar(para->mip, 0);
    x->c.b.a = mirvar(para->mip, 0);
    x->c.b.b = mirvar(para->mip, 0);
    x->c.unitary = FALSE;
    x->miller = FALSE;
    x->unitary = FALSE;
}

/****************************************************************
Function:       zzn12_copy
Description:    copy y=x
Calls:          MIRACL functions
Called By:
Input:          zzn12 *x
Output:         zzn12 *y
Return:         null
Others:
****************************************************************/
__FUNCTION_HEADER__ void zzn12_copy(struct SM9_Para *para, zzn12 *x, zzn12 *y) {
    zzn4_copy(&x->a, &y->a);
    zzn4_copy(&x->b, &y->b);
    zzn4_copy(&x->c, &y->c);
    y->miller = x->miller;
    y->unitary = x->unitary;
}

/****************************************************************
Function:       zzn12_mul
Description:    z=x*y,see zzn12a.h and zzn12a.cpp for details in MIRACL c++ source file
Calls:          MIRACL functions
Called By:
Input:          zzn12 x,y
Output:         zzn12 *z
Return:         null
Others:
****************************************************************/

__FUNCTION_HEADER__ void zzn12_mul(struct SM9_Para *para, zzn12 x, zzn12 y, zzn12 *z) {
    // Karatsuba
//	zzn4 Z0, Z1, Z2, Z3, T0, T1;
    BOOL zero_c, zero_b;

    zzn12_copy(para, &x, z);

    if (zzn4_compare(&x.a, &y.a) && zzn4_compare(&x.a, &y.a) && zzn4_compare(&x.a, &y.a)) {
        if (x.unitary == TRUE) {
            zzn4_copy(&x.a, &(para->Z0));
            zzn4_mul(para->mip, &x.a, &x.a, &z->a);
            zzn4_copy(&z->a, &(para->Z3));
            zzn4_add(para->mip, &z->a, &z->a, &z->a);
            zzn4_add(para->mip, &z->a, &(para->Z3), &z->a);
            zzn4_conj(para->mip, &(para->Z0), &(para->Z0));
            zzn4_add(para->mip, &(para->Z0), &(para->Z0), &(para->Z0));
            zzn4_sub(para->mip, &z->a, &(para->Z0), &z->a);
            zzn4_copy(&x.c, &(para->Z1));
            zzn4_mul(para->mip, &(para->Z1), &(para->Z1), &(para->Z1));
            zzn4_tx(para->mip, &(para->Z1));
            zzn4_copy(&(para->Z1), &(para->Z3));
            zzn4_add(para->mip, &(para->Z1), &(para->Z1), &(para->Z1));
            zzn4_add(para->mip, &(para->Z1), &(para->Z3), &(para->Z1));
            zzn4_copy(&x.b, &(para->Z2));
            zzn4_mul(para->mip, &(para->Z2), &(para->Z2), &(para->Z2));
            zzn4_copy(&(para->Z2), &(para->Z3));
            zzn4_add(para->mip, &(para->Z2), &(para->Z2), &(para->Z2));
            zzn4_add(para->mip, &(para->Z2), &(para->Z3), &(para->Z2));

            zzn4_conj(para->mip, &x.b, &z->b);
            zzn4_add(para->mip, &z->b, &z->b, &z->b);
            zzn4_conj(para->mip, &x.c, &z->c);
            zzn4_add(para->mip, &z->c, &z->c, &z->c);
            zzn4_negate(para->mip, &z->c, &z->c);
            zzn4_add(para->mip, &z->b, &(para->Z1), &z->b);
            zzn4_add(para->mip, &z->c, &(para->Z2), &z->c);
        } else {
            if (!x.miller) {// Chung-Hasan SQR2
//                zzn4_copy(&x.a, &(para->Z0));
//                zzn4_mul(para->mip, &(para->Z0), &(para->Z0), &(para->Z0));
//                zzn4_mul(para->mip, &x.b, &x.c, &(para->Z1));
//                zzn4_add(para->mip, &(para->Z1), &(para->Z1), &(para->Z1));
//                zzn4_copy(&x.c, &(para->Z2));
//                zzn4_mul(para->mip, &(para->Z2), &(para->Z2), &(para->Z2));
//                zzn4_mul(para->mip, &x.a, &x.b, &(para->Z3));
//                zzn4_add(para->mip, &(para->Z3), &(para->Z3), &(para->Z3));
//                zzn4_add(para->mip, &x.a, &x.b, &z->c);
//                zzn4_add(para->mip, &z->c, &x.c, &z->c);
//                zzn4_mul(para->mip, &z->c, &z->c, &z->c);
//
//                zzn4_tx(para->mip, &(para->Z1));
//                zzn4_add(para->mip, &(para->Z0), &(para->Z1), &z->a);
//                zzn4_tx(para->mip, &(para->Z2));
//                zzn4_add(para->mip, &(para->Z3), &(para->Z2), &z->b);
//                zzn4_add(para->mip, &(para->Z0), &(para->Z1), &(para->T0));
//                zzn4_add(para->mip, &(para->T0), &(para->Z2), &(para->T0));
//                zzn4_add(para->mip, &(para->T0), &(para->Z3), &(para->T0));
//                zzn4_sub(para->mip, &z->c, &(para->T0), &z->c);
            } else {// Chung-Hasan SQR3 - actually calculate 2x^2 !
                // Slightly dangerous - but works as will be raised to p^{k/2}-1
                // which wipes out the 2.
                zzn4_copy(&x.a, &(para->Z0));
                zzn4_mul(para->mip, &(para->Z0), &(para->Z0), &(para->Z0));// a0^2    = S0
                zzn4_copy(&x.c, &(para->Z2));
                zzn4_mul(para->mip, &(para->Z2), &x.b, &(para->Z2));
                zzn4_add(para->mip, &(para->Z2), &(para->Z2), &(para->Z2)); // 2a1.a2  = S3
                zzn4_copy(&x.c, &(para->Z3));
                zzn4_mul(para->mip, &(para->Z3), &(para->Z3), &(para->Z3));;       // a2^2    = S4
                zzn4_add(para->mip, &x.c, &x.a, &z->c);           // a0+a2

                zzn4_copy(&x.b, &(para->Z1));
                zzn4_add(para->mip, &(para->Z1), &z->c, &(para->Z1));
                zzn4_mul(para->mip, &(para->Z1), &(para->Z1), &(para->Z1));// (a0+a1+a2)^2  =S1
                zzn4_sub(para->mip, &z->c, &x.b, &z->c);
                zzn4_mul(para->mip, &z->c, &z->c, &z->c);// (a0-a1+a2)^2  =S2
                zzn4_add(para->mip, &(para->Z2), &(para->Z2), &(para->Z2));
                zzn4_add(para->mip, &(para->Z0), &(para->Z0), &(para->Z0));
                zzn4_add(para->mip, &(para->Z3), &(para->Z3), &(para->Z3));

                zzn4_sub(para->mip, &(para->Z1), &z->c, &(para->T0));
                zzn4_sub(para->mip, &(para->T0), &(para->Z2), &(para->T0));
                zzn4_sub(para->mip, &(para->Z1), &(para->Z0), &(para->T1));
                zzn4_sub(para->mip, &(para->T1), &(para->Z3), &(para->T1));
                zzn4_add(para->mip, &z->c, &(para->T1), &z->c);
                zzn4_tx(para->mip, &(para->Z3));
                zzn4_add(para->mip, &(para->T0), &(para->Z3), &z->b);
                zzn4_tx(para->mip, &(para->Z2));
                zzn4_add(para->mip, &(para->Z0), &(para->Z2), &z->a);
            }
        }
    } else {
        // Karatsuba
        zero_b = zzn4_iszero(&y.b);
        zero_c = zzn4_iszero(&y.c);

        zzn4_mul(para->mip, &x.a, &y.a, &(para->Z0));  //9
        if (!zero_b) zzn4_mul(para->mip, &x.b, &y.b, &(para->Z2));  //+6

        zzn4_add(para->mip, &x.a, &x.b, &(para->T0));
        zzn4_add(para->mip, &y.a, &y.b, &(para->T1));
        zzn4_mul(para->mip, &(para->T0), &(para->T1), &(para->Z1)); //+9
        zzn4_sub(para->mip, &(para->Z1), &(para->Z0), &(para->Z1));
        if (!zero_b) zzn4_sub(para->mip, &(para->Z1), &(para->Z2), &(para->Z1));

        zzn4_add(para->mip, &x.b, &x.c, &(para->T0));
        zzn4_add(para->mip, &y.b, &y.c, &(para->T1));
        zzn4_mul(para->mip, &(para->T0), &(para->T1), &(para->Z3));//+6
        if (!zero_b) zzn4_sub(para->mip, &(para->Z3), &(para->Z2), &(para->Z3));

        zzn4_add(para->mip, &x.a, &x.c, &(para->T0));
        zzn4_add(para->mip, &y.a, &y.c, &(para->T1));
        zzn4_mul(para->mip, &(para->T0), &(para->T1), &(para->T0));//+9=39 for "special case"
        if (!zero_b) zzn4_add(para->mip, &(para->Z2), &(para->T0), &(para->Z2));
        else zzn4_copy(&(para->T0), &(para->Z2));
        zzn4_sub(para->mip, &(para->Z2), &(para->Z0), &(para->Z2));

        zzn4_copy(&(para->Z1), &z->b);
        if (!zero_c) { // exploit special form of BN curve line function
            zzn4_mul(para->mip, &x.c, &y.c, &(para->T0));
            zzn4_sub(para->mip, &(para->Z2), &(para->T0), &(para->Z2));
            zzn4_sub(para->mip, &(para->Z3), &(para->T0), &(para->Z3));
            zzn4_tx(para->mip, &(para->T0));
            zzn4_add(para->mip, &z->b, &(para->T0), &z->b);
        }

        zzn4_tx(para->mip, &(para->Z3));
        zzn4_add(para->mip, &(para->Z0), &(para->Z3), &z->a);
        zzn4_copy(&(para->Z2), &z->c);
        if (!y.unitary) z->unitary = FALSE;
    }
}

/****************************************************************
Function:       zzn12_conj
Description:    achieve conjugate complex
see zzn12a.h and zzn1212.cpp for details in MIRACL c++ source file
Calls:          MIRACL functions
Called By:
Input:          zzn12 x,y
Output:         zzn12 *z
Return:         null
Others:
****************************************************************/
__FUNCTION_HEADER__ void zzn12_conj(struct SM9_Para *para, zzn12 *x, zzn12 *y) {
    zzn4_conj(para->mip, &x->a, &y->a);
    zzn4_conj(para->mip, &x->b, &y->b);
    zzn4_negate(para->mip, &y->b, &y->b);
    zzn4_conj(para->mip, &x->c, &y->c);
    y->miller = x->miller;
    y->unitary = x->unitary;
}
/****************************************************************
Function:       zzn12_inverse
Description:    element inversion,
see zzn12a.h and zzn1212.cpp for details in MIRACL c++ source file
Calls:          MIRACL functions,zzn12_init,zzn12_conj
Called By:
Input:          zzn12 w
Output:
Return:         zzn12
Others:
****************************************************************/
__FUNCTION_HEADER__ zzn12 zzn12_inverse(struct SM9_Para *para, zzn12 w){
    zzn12 res;
    zzn12_init(para, &res);

    if (w.unitary) {
        zzn12_conj(para, &w, &res);
        return res;
    }
    //res.a=w.a*w.a-tx(w.b*w.c);
    zzn4_mul(para->mip, &w.a, &w.a, &res.a);
    zzn4_mul(para->mip, &w.b, &w.c, &res.b);
    zzn4_tx(para->mip, &res.b);
    zzn4_sub(para->mip, &res.a, &res.b, &res.a);

    //res.b=tx(w.c*w.c)-w.a*w.b;
    zzn4_mul(para->mip, &w.c, &w.c, &res.c);
    zzn4_tx(para->mip, &res.c);
    zzn4_mul(para->mip, &w.a, &w.b, &res.b);
    zzn4_sub(para->mip, &res.c, &res.b, &res.b);

    //res.c=w.b*w.b-w.a*w.c;
    zzn4_mul(para->mip, &w.b, &w.b, &res.c);
    zzn4_mul(para->mip, &w.a, &w.c, &para->tmp1);
    zzn4_sub(para->mip, &res.c, &para->tmp1, &res.c);

    //para->tmp1=tx(w.b*res.c)+w.a*res.a+tx(w.c*res.b);
    zzn4_mul(para->mip, &w.b, &res.c, &para->tmp1);
    zzn4_tx(para->mip, &para->tmp1);
    zzn4_mul(para->mip, &w.a, &res.a, &para->tmp2);
    zzn4_add(para->mip, &para->tmp1, &para->tmp2, &para->tmp1);
    zzn4_mul(para->mip, &w.c, &res.b, &para->tmp2);
    zzn4_tx(para->mip, &para->tmp2);
    zzn4_add(para->mip, &para->tmp1, &para->tmp2, &para->tmp1);

    zzn4_inv(para->mip, &para->tmp1);
    zzn4_mul(para->mip, &res.a, &para->tmp1, &res.a);
    zzn4_mul(para->mip, &res.b, &para->tmp1, &res.b);
    zzn4_mul(para->mip, &res.c, &para->tmp1, &res.c);

    return res;
}
/****************************************************************
Function:       zzn12_powq
Description:    Frobenius F=x^p. Assumes p=1 mod 6
see zzn12a.h and zzn1212.cpp for details in MIRACL c++ source file
Calls:          MIRACL functions
Called By:
Input:          zzn2 F
Output:         zzn12 *y
Return:         NULL
Others:
****************************************************************/
__FUNCTION_HEADER__ void zzn12_powq(struct SM9_Para *para, zzn2 F, zzn12 *y) {
    zzn2_mul(para->mip, &F, &F, &para->X2);
    zzn2_mul(para->mip, &para->X2, &F, &para->X3); //zzn2_mul(para->mip, &X2, &X, &X3); 经测试F=X

    zzn4_powq(para->mip, &para->X3, &y->a);
    zzn4_powq(para->mip, &para->X3, &y->b);
    zzn4_powq(para->mip, &para->X3, &y->c);
    zzn4_smul(para->mip, &y->b, &F, &y->b);
    zzn4_smul(para->mip, &y->c, &para->X2, &y->c);
}
/****************************************************************
Function:       zzn12_div
Description:    z=x/y
see zzn12a.h and zzn1212.cpp for details in MIRACL c++ source file
Calls:          MIRACL functions,zzn12_inverse,zzn12_mul
Called By:
Input:          zzn12 x,y
Output:         zzn12 *z
Return:         NULL
Others:
****************************************************************/
__FUNCTION_HEADER__ void zzn12_div(struct SM9_Para *para, zzn12 x, zzn12 y, zzn12 *z) {
    y = zzn12_inverse(para, y);
    zzn12_mul(para, x, y, z);
}

/****************************************************************
Function:       zzn12_pow
Description:    regular zzn12 powering,If k is low Hamming weight this will be just as good.
see zzn12a.h and zzn1212.cpp for details in MIRACL c++ source file
Calls:          MIRACL functions,zzn12_inverse,zzn12_mul,zzn12_copy,zzn12_init
Called By:
Input:          zzn12 x,big k
Output:
Return:         zzn12
Others:
****************************************************************/
__FUNCTION_HEADER__ zzn12 zzn12_pow(struct SM9_Para *para, zzn12 x, big k) {
    int nb, i;
    BOOL invert_it;
    zzn12 res;

//    para->zero = mirvar(para->mip, 0);
//    para->tmp11 = mirvar(para->mip, 0);
    zzn12_init(para, &res);
    copy(k, para->tmp11);
    invert_it = FALSE;

    if (mr_compare(para->tmp11, para->zero) == 0) {
        para->tmp000 = para->mip->one;
        zzn4_from_big(para->mip, para->tmp000, &res.a);
        return res;
    }
    if (mr_compare(para->tmp11, para->zero) < 0) {
        negify(para->tmp11, para->tmp11);
        invert_it = TRUE;
    }
    nb = logb2(para->mip, k);
    zzn12_copy(para, &x, &res);
    if (nb > 1)
        for (i = nb - 2; i >= 0; i--) {
            zzn12_mul(para, res, res, &res);
            if (mr_testbit(para->mip, k, i)) zzn12_mul(para, res, x, &res);
        }
    if (invert_it) res = zzn12_inverse(para, res);

    return res;
}

