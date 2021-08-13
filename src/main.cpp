#include <iostream>
#include <ctime>
#include "../inc/Layer.h"
#define NARGS 3
#define NHIDN 6
#define BATCH 64
#define NTEST 10000
#define NITER 1000000
#define LRATE 0.2
#define PREGU 0.1

extern "C"
{
  #include "../inc/myMath.h"
}

#ifdef TIMING
clock_t c_fwp, c_bwp, c_mlp;
#endif

int rng[NARGS] = { 10, 10, 10 };
Real radius2 = 25;

void renew(Real *x, Real *y) {
	// Generating Data
	for (long i = 0; i < BATCH; i++) {
		y[i] = 0;
		for (long j = 0; j < NARGS; j++) {
			x[j * BATCH + i] = rand() % rng[j] - rng[j] / 2;
			y[i] += x[j * BATCH + i] * x[j * BATCH + i];
			x[j * BATCH + i] /= rng[j];
		}
		y[i] = (y[i] > radius2) ? 1 : 0;
	}
}

int main() {
	Real x[NARGS * BATCH], y[BATCH];
	srand((unsigned)time(NULL));

	// Training
	Layer l1(NARGS, NHIDN, BATCH, ReLU);
	Layer l2(NHIDN, NHIDN, BATCH, ReLU);
	Layer l3(NHIDN, 1, BATCH, sigmoid);
	l1.setPRegu(PREGU);
	l2.setPRegu(PREGU);
	l3.setPRegu(PREGU);
#ifdef TIMING
	c_fwp = c_bwp = c_mlp = 0;
	c_mlp -= clock();
#endif
	for (long i = 0; i < NITER; i++) {
		if (i % 10000 == 0) {
			renew(x, y);
		}
#ifdef TIMING
		c_fwp -= clock();
#endif
		l1.fwp(x);
		l2.fwp(l1.getA());
		l3.fwp(l2.getA());
#ifdef TIMING
		c_fwp += clock();
#endif
		if (i % 1000 == 0) {
			std::cout << "Iter Step: " << i << ", Loss: " << lgLoss(l3.getA(), y, BATCH, 0) << std::endl;
		}
		lgLoss(l3.getA(), y, BATCH, 1);
#ifdef TIMING
		c_bwp -= clock();
#endif
		l3.bwp(l3.getA());
		l2.bwp(l3.getDx());
		l1.bwp(l2.getDx());
#ifdef TIMING
		c_bwp += clock();
#endif
		l1.setLRate(LRATE / sqrt(i + 1));
		l2.setLRate(LRATE / sqrt(i + 1));
		l3.setLRate(LRATE / sqrt(i + 1));
	}
#ifdef TIMING
	c_mlp += clock();
#endif

	// Test
	int nTotal = 0;
	int nCorrect = 0;
	for (long i = 0; i < NTEST; i++) {
		y[0] = 0;
		for (long j = 0; j < NARGS; j++) {
			x[j] = rand() % rng[j] - rng[j] / 2;
			y[0] += x[j] * x[j];
			x[j] /= rng[j];
		}
		y[0] = (y[0] > radius2) ? 1 : 0;
		Real *a = l3.run(l2.run(l1.run(x)));
		//Real* a = l3.run(l1.run(x));
		a[0] = (a[0] > 0.5) ? 1 : 0;
		if (a[0] == y[0]) nCorrect++;
		nTotal++;
	}

	std::cout << "\nTesting Accuracy:           " << (nCorrect / (Real)nTotal) * 100 << "%\n";
#ifdef TIMING
	std::cout << "Training Main Loop Time:    " << (Real)c_mlp / CLOCKS_PER_SEC << "s\n";
	std::cout << "Forward Propagation Time:   " << (Real)c_fwp / CLOCKS_PER_SEC << "s\n";
	std::cout << "Backward Propagation Time:  " << (Real)c_bwp / CLOCKS_PER_SEC << "s\n";
	std::cout << "Matrix Multiplication Time: " << l1.outClock() + l2.outClock() + l3.outClock() << "s\n";
#endif
	return 0;
}
