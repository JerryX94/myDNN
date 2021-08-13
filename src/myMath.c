#include "../inc/myMath.h"

void ReLU(Real *x, Real *y, const long len, const int derv) {
    long i;
    Real leaky = 0.001;
    if (derv) {
        for (i = 0; i < len; i++) {
            y[i] = (x[i] > 0) ? 1 : leaky;
        }
    }
    else {
        for (i = 0; i < len; i++) {
            y[i] = (x[i] > 0) ? x[i] : leaky * x[i];
        }
    }
}

void sigmoid(Real *x, Real *y, const long len, const int derv) {
    long i;
    if (derv) {
        for (i = 0; i < len; i++) {
            Real e = exp(-x[i]);
            y[i] = e / ((1 + e) * (1 + e));
        }
    }
    else {
        for (i = 0; i < len; i++) {
            Real e = exp(-x[i]);
            y[i] = 1 / (1 + e);
        }
    }
}

Real lgLoss(Real *a, Real *y, const long len, const int derv) {
    long i;
    if (derv) {
        for (i = 0; i < len; i++) {
            a[i] = -(y[i] / a[i]) + ((1 - y[i]) / (1 - a[i]));
        }
        return 0;
    }
    else {
        Real loss = 0;
        for (i = 0; i < len; i++) {
            loss += y[i] * log(a[i]) + (1 - y[i]) * log(1 - a[i]);
        }
        return (-loss) / len;
    }
}

void zeros(Real *x, const long len) {
    long i;
    for (i = 0; i < len; i++)
        x[i] = 0;
}

void rands(Real *x, const long len, const long ni, const long no) {
    long i;
    Real range = sqrt(3 / (Real)ni);
    Real amplf = RAND_MAX / range;
    srand((unsigned)time(NULL));
    for (i = 0; i < len; i++) {
        x[i] = rand() % (int)(range * amplf);
        x[i] = x[i] / amplf - 0.5 * range;
    }
}

void transpose(Real *x, Real *y, const long nRows, const long nCols) {
    long i, j;
    for (j = 0; j < nRows; j++)
        for (i = 0; i < nCols; i++)
            y[i * nRows + j] = x[j * nCols + i];
}

void matMul(Real *w, Real *x, Real *y,
            const long nRows,
            const long nShrs,
            const long nCols
            ) {
    long i, j, k;
    zeros(y, nRows * nCols);
    for (j = 0; j < nRows; j++)
        for (k = 0; k < nShrs; k++)
            for (i = 0; i < nCols; i++)
                y[j * nCols + i] += w[j * nShrs + k] * x[k * nCols + i];
}

void vecPls(Real *v, Real *x, Real *y, const long nRows, const long nCols, const int axis) {
    long i, j;
    if (axis) {
        for (j = 0; j < nRows; j++)
            for (i = 0; i < nCols; i++)
                y[j * nCols + i] = x[j * nCols + i] + v[j];
    }
    else {
        for (j = 0; j < nRows; j++)
            for (i = 0; i < nCols; i++)
                y[j * nCols + i] = x[j * nCols + i] + v[i];
    }
}

void elePls(Real* m, Real* x, Real* y, const long len) {
    long i;
    for (i = 0; i < len; i++)
        y[i] = x[i] + m[i];
}

void eleMns(Real* m, Real* x, Real* y, const long len) {
    long i;
    for (i = 0; i < len; i++)
        y[i] = x[i] - m[i];
}

void eleMul(Real *m, Real *x, Real *y, const long len) {
    long i;
    for (i = 0; i < len; i++)
        y[i] = x[i] * m[i];
}

void scaMul(Real s, Real *x, Real *y, const long len) {
    long i;
    for (i = 0; i < len; i++)
        y[i] = x[i] * s;
}

void sumM2V(Real *m, Real *v, const long nRows, const long nCols, const int axis) {
    long i, j;
    if (axis) {
        for (j = 0; j < nRows; j++) {
            v[j] = 0;
            for (i = 0; i < nCols; i++)
                v[j] += m[j * nCols + i];
        }
    }
    else {
        zeros(v, nCols);
        for (j = 0; j < nRows; j++)
            for (i = 0; i < nCols; i++) {
                v[i] += m[j * nCols + i];
        }
    }
}
