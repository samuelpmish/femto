#include <vector>
#include <iostream>

#include "misc/timer.hpp"

namespace compiler {
static void please_do_not_optimize_away([[maybe_unused]] void* p) { asm volatile("" : : "g"(p) : "memory"); }
} 

template < int dim >
struct Vector {
  double & operator()(int i) { return data[i]; }
  const double & operator()(int i) const { return data[i]; }
  double data[dim];
};

template < int dim >
struct Matrix {
  double & operator()(int i, int j) { return data[i][j]; }
  const double & operator()(int i, int j) const { return data[i][j]; }
  double data[dim][dim];
};

template < int dim >
Vector<dim> operator*(const Matrix<dim> & A, const Vector<dim> & x){
  Vector<dim> Ax{};
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      Ax(i) += A(i,j) * x(j);
    }
  }
  return Ax;
}

template < int dim >
double batched_matvec_test(size_t n, int runs) {

  std::vector< Vector<dim> > x(n);
  std::vector< Matrix<dim> > A(n);
  std::vector< Vector<dim> > Ax(n);

  femto::timer stopwatch;
  stopwatch.start();
  for (int k = 0; k < runs; k++) {
    for (int i = 0; i < n; i++) {
      Ax[i] = A[i] * x[i]; 
    }
    compiler::please_do_not_optimize_away(&Ax[0]);
  }
  stopwatch.stop();
  return stopwatch.elapsed() / runs;

}

int main() {
  size_t runs = 100;
  size_t n = 640000;
  std::cout << "average batched_matvec_test<2> runtime: " << batched_matvec_test<2>(n, runs) << std::endl;
  std::cout << "average batched_matvec_test<3> runtime: " << batched_matvec_test<3>(n, runs) << std::endl;
}