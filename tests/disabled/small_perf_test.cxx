
#include "misc/timer.hpp"
#include "containers/ndarray.hpp"

#include <random>
#include <iostream>

namespace compiler {
static void please_do_not_optimize_away([[maybe_unused]] void* p) { asm volatile("" : : "g"(p) : "memory"); }
}

double random_real() { 
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  return distribution(generator);
}

int main() {

  uint64_t n = 1000000;

  ndview in1(new double[n*3], {n, 3});
  ndview in2(new double[n*3*3], {n, 3, 3});

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      in1(i,j) = random_real();
      for (int k = 0; k < 3; k++) {
        in2(i,j,k) = random_real();
      }
    }
  }

  ndview out(new double[n*3], {n, 3});

  for (int m = 0; m < 50; m++) {
    std::cout << femto::time([&](){
      #pragma omp parallel for
      for (int i = 0; i < n; i++) {

        double A00 = in2(i,0,0);
        double A01 = in2(i,0,1);
        double A02 = in2(i,0,2);
        double A10 = in2(i,1,0);
        double A11 = in2(i,1,1);
        double A12 = in2(i,1,2);
        double A20 = in2(i,2,0);
        double A21 = in2(i,2,1);
        double A22 = in2(i,2,2);

        double inv_det = 1.0 / (
          A00 * A11 * A22 + A01 * A12 * A20 +
          A02 * A10 * A21 - A00 * A12 * A21 -
          A01 * A10 * A22 - A02 * A11 * A20
        );

        double tmp00 = (A11 * A22 - A12 * A21) * inv_det;
        double tmp01 = (A02 * A21 - A01 * A22) * inv_det;
        double tmp02 = (A01 * A12 - A02 * A11) * inv_det;
        double tmp10 = (A12 * A20 - A10 * A22) * inv_det;
        double tmp11 = (A00 * A22 - A02 * A20) * inv_det;
        double tmp12 = (A02 * A10 - A00 * A12) * inv_det;
        double tmp20 = (A10 * A21 - A11 * A20) * inv_det;
        double tmp21 = (A01 * A20 - A00 * A21) * inv_det;
        double tmp22 = (A00 * A11 - A01 * A10) * inv_det;

        out(i,0) = in1(i,0) * tmp00 + in1(i,1) * tmp10 + in1(i,2) * tmp20;
        out(i,1) = in1(i,0) * tmp01 + in1(i,1) * tmp11 + in1(i,2) * tmp21;
        out(i,2) = in1(i,0) * tmp02 + in1(i,1) * tmp12 + in1(i,2) * tmp22;
      }

    }) << "s" << std::endl;

    compiler::please_do_not_optimize_away(out.data);

  }

  delete[] in1.data;
  delete[] in2.data;
  delete[] out.data;

}