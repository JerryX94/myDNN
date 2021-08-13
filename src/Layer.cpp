#include "../inc/Layer.h"

extern "C"
{
  #include "../inc/myMath.h"
}

Layer::Layer(const long nArgs,
             const long nCels,
             const long nSmps,
             void (*actvFunc)(Real *, Real *, const long, const int)
             ) {
    _actvFunc = actvFunc;
    _lRate = DEFAULT_LRATE;
    _pRegu = DEFAULT_PREGU;
    _nArgs = nArgs;
    _nCels = nCels;
    _nSmps = nSmps;
    _nTrns = (_nCels > _nSmps) ? _nCels : _nSmps;
    _w  = (Real *)malloc(_nCels * _nArgs * sizeof(Real));
    _dw = (Real *)malloc(_nCels * _nArgs * sizeof(Real));
    _b  = (Real *)malloc(_nCels * sizeof(Real));
    _db = (Real *)malloc(_nCels * sizeof(Real));
    _z  = (Real *)malloc(_nCels * _nSmps * sizeof(Real));
    _dx = (Real *)malloc(_nArgs * _nSmps * sizeof(Real));
    _a  = (Real *)malloc(_nCels * _nSmps * sizeof(Real));
    _tr = (Real *)malloc(_nArgs * _nTrns * sizeof(Real));
    rands(_w, _nCels * _nArgs, _nArgs, _nCels);
    rands(_b, _nCels, _nArgs, _nCels);
#ifdef TIMING
    _cMatMul = 0;
#endif
}

Layer::~Layer() {
    free(_w);
    free(_dw);
    free(_b);
    free(_db);
    free(_z);
    free(_dx);
    free(_a);
    free(_tr);
}

void Layer::fwp(Real *x) {
    _x = x;

#ifdef TIMING
    _cMatMul -= clock();
#endif
    matMul(_w, _x, _z, _nCels, _nArgs, _nSmps);
#ifdef TIMING
    _cMatMul += clock();
#endif
    vecPls(_b, _z, _z, _nCels, _nSmps, 1);

    (*_actvFunc)(_z, _a, _nCels * _nSmps, 0);
}

void Layer::bwp(Real *da) {
    _da = da;

    (*_actvFunc)(_z, _z, _nCels * _nSmps, 1);
    eleMul(_da, _z, _z, _nCels * _nSmps);

    transpose(_x, _tr, _nArgs, _nSmps);
#ifdef TIMING
    _cMatMul -= clock();
#endif
    matMul(_z, _tr, _dw, _nCels, _nSmps, _nArgs);
#ifdef TIMING
    _cMatMul += clock();
#endif
    scaMul(_pRegu, _w, _tr, _nCels * _nArgs);           // Regularization
    elePls(_tr, _dw, _dw, _nCels * _nArgs);             // Regularization
    scaMul(_lRate / _nSmps, _dw, _dw, _nCels * _nArgs);

    sumM2V(_z, _db, _nCels, _nSmps, 1);
    scaMul(_lRate / _nSmps, _db, _db, _nCels);

    transpose(_w, _tr, _nCels, _nArgs);
#ifdef TIMING
    _cMatMul -= clock();
#endif
    matMul(_tr, _z, _dx, _nArgs, _nCels, _nSmps);
#ifdef TIMING
    _cMatMul += clock();
#endif

    eleMns(_dw, _w, _w, _nCels * _nArgs);
    eleMns(_db, _b, _b, _nCels);
}

Real *Layer::run(Real *x) {
    _x = x;
    matMul(_w, _x, _z, _nCels, _nArgs, 1);
    vecPls(_b, _z, _z, _nCels, 1, 1);
    (*_actvFunc)(_z, _a, _nCels, 0);
    return _a;
}

void Layer::setLRate(Real lRate) {
    _lRate = lRate;
}

void Layer::setPRegu(Real pRegu) {
    _pRegu = pRegu;
}
