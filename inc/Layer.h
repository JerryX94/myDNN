#include <cstdlib>
#include "dataType.h"

#define DEFAULT_LRATE 0.1
#define DEFAULT_PREGU 0.0
#define TIMING

#ifdef TIMING
  #include <ctime>
#endif

class Layer {
    public:
        Layer(const long nArgs,
              const long nCels,
              const long nSmps,
              void (*actvFunc)(Real *, Real *, const long, const int)
              );
        ~Layer();
        void fwp(Real *x);
        void bwp(Real *da);
        Real *run(Real *x);
        void setLRate(Real lRate);
        void setPRegu(Real pRegu);
        inline Real *getA() const {return _a;};
        inline Real *getDx() const {return _dx;};
        inline Real *getW() const {return _w;};
        inline Real *getB() const {return _b;};
#ifdef TIMING
        inline Real outClock() const {return (Real)_cMatMul / CLOCKS_PER_SEC;};
#endif

    private:
        void (*_actvFunc)(Real *, Real *, const long, const int);
        long _nArgs;
        long _nCels;
        long _nSmps;
        long _nTrns;
        Real _lRate;
        Real _pRegu;
        Real *_w;
        Real *_b;
        Real *_x;
        Real *_a;
        Real *_z;
        Real *_dw;
        Real *_db;
        Real *_dx;
        Real *_da;
        Real *_tr;
#ifdef TIMING
        clock_t _cMatMul;
#endif
};
