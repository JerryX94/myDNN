#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "dataType.h"

void ReLU(Real *x, Real *y, const long len, const int derv);
void sigmoid(Real *x, Real *y, const long len, const int derv);
Real lgLoss(Real *a, Real *y, const long len, const int derv);
void zeros(Real *x, const long len);
void rands(Real *x, const long len, const long ni, const long no);
void transpose(Real *x, Real *y, const long nRows, const long nCols);
void matMul(Real *w, Real *x, Real *y,
            const long nRows,
            const long nShrs,
            const long nCols);
void vecPls(Real *v, Real *x, Real *y, const long nRows, const long nCols, const int axis);
void elePls(Real *m, Real *x, Real *y, const long len);
void eleMns(Real *m, Real *x, Real *y, const long len);
void eleMul(Real *m, Real *x, Real *y, const long len);
void scaMul(Real s, Real *x, Real *y, const long len);
void sumM2V(Real *m, Real *v, const long nRows, const long nCols, const int axis);
