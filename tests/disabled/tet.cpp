#include "femto/quadrature.hpp"
#include "femto/interpolation.hpp"

#include "misc/timer.hpp"

#include <cstring>

//void GaussLobattoInterpolationTetrahedron(const double * xi, int n, double * output);

uint64_t T(uint64_t n) { return (n * (n + 1) * (n + 2)) / 6; }

double memcpy_ref(const std::vector < double > & x, std::vector< double > & y, uint64_t n, uint64_t q, uint64_t num_elements) {
  femto::timer stopwatch;
  stopwatch.start();
  std::memcpy(&y[0], &x[0], num_elements * T(n));
  stopwatch.stop();
  return stopwatch.elapsed();
}

double naive(const std::vector < double > & x, std::vector< double > & y, uint64_t n, uint64_t q, uint64_t num_elements) {

  femto::timer stopwatch;
  stopwatch.start();

  uint64_t Tn = T(n);
  uint64_t Tq = T(q);

  std::vector< double > qpts(3 * Tq);
  std::vector< double > qwts(Tq); // unused
  gauss_legendre_tetrahedron_rule(q, &qpts[0], &qwts[0]);

  std::vector< double > B(Tn);
  for (uint64_t e = 0; e < num_elements; e++) {
    std::vector< double > x_e(Tn, 0);
    for (uint64_t i = 0; i < Tn; i++) { x_e[i] = x[e * Tn + i]; }

    std::vector< double > y_e(Tn, 0);
    for (uint64_t i = 0; i < Tq; i++) {
      GaussLobattoInterpolationTetrahedron(&qpts[0] + 3 * i, n - 1, &B[0]);  
      double sum = 0;
      for (uint64_t j = 0; j < Tn; j++) { sum += x_e[j] * B[j]; } 
      for (uint64_t j = 0; j < Tn; j++) { y_e[j] += sum * B[j]; }
    }

    for (uint64_t i = 0; i < Tn; i++) { y[e * Tn + i]  = y_e[i]; }
  }

  stopwatch.stop();
  return stopwatch.elapsed();
}

double precompute_shape_fns(const std::vector < double > & x, std::vector< double > & y, uint64_t n, uint64_t q, uint64_t num_elements) {

  femto::timer stopwatch;
  stopwatch.start();

  uint64_t Tn = T(n);
  uint64_t Tq = T(q);

  std::vector< double > qpts(3 * Tq);
  std::vector< double > qwts(Tq); // unused
  gauss_legendre_tetrahedron_rule(q, &qpts[0], &qwts[0]);

  std::vector< double > B(Tn * Tq);
  for (uint64_t i = 0; i < Tq; i++) {
    GaussLobattoInterpolationTetrahedron(&qpts[0] + 3 * i, n - 1, &B[Tn * i]);  
  }

  for (uint64_t e = 0; e < num_elements; e++) {
    std::vector< double > x_e(Tn, 0);
    for (uint64_t i = 0; i < Tn; i++) { x_e[i] = x[e * Tn + i]; }

    std::vector< double > y_e(Tn, 0);
    for (uint64_t i = 0; i < Tq; i++) {
      double sum = 0;
      for (uint64_t j = 0; j < Tn; j++) { sum += x_e[j] * B[Tn * i + j]; } 
      for (uint64_t j = 0; j < Tn; j++) { y_e[j] += sum * B[Tn * i + j]; }
    }

    for (uint64_t i = 0; i < Tn; i++) { y[e * Tn + i]  = y_e[i]; }
  }

  stopwatch.stop();
  return stopwatch.elapsed();
}

double no_alloc_in_loop(const std::vector < double > & x, std::vector< double > & y, uint64_t n, uint64_t q, uint64_t num_elements) {

  femto::timer stopwatch;
  stopwatch.start();

  uint64_t Tn = T(n);
  uint64_t Tq = T(q);

  std::vector< double > qpts(3 * Tq);
  std::vector< double > qwts(Tq); // unused
  gauss_legendre_tetrahedron_rule(q, &qpts[0], &qwts[0]);

  std::vector< double > B(Tn * Tq);
  for (uint64_t i = 0; i < Tq; i++) {
    GaussLobattoInterpolationTetrahedron(&qpts[0] + 3 * i, n - 1, &B[Tn * i]);  
  }

  std::vector< double > x_e(Tn);
  std::vector< double > y_e(Tn);
  for (uint64_t e = 0; e < num_elements; e++) {
    for (uint64_t i = 0; i < Tn; i++) { 
      x_e[i] = x[e * Tn + i]; 
      y_e[i] = 0;
    }

    for (uint64_t i = 0; i < Tq; i++) {
      double sum = 0;
      for (uint64_t j = 0; j < Tn; j++) { sum += x_e[j] * B[Tn * i + j]; } 
      for (uint64_t j = 0; j < Tn; j++) { y_e[j] += sum * B[Tn * i + j]; }
    }

    for (uint64_t i = 0; i < Tn; i++) { y[e * Tn + i]  = y_e[i]; }
  }

  stopwatch.stop();
  return stopwatch.elapsed();
}

int main() {

  uint64_t n_max = 4;
  uint64_t q_max = 6;
  uint64_t num_elements = 300000;

  std::vector< double > in(num_elements * T(n_max), 1);
  std::vector< double > out(num_elements * T(n_max));

  for (int n = 2; n <= n_max; n++) {
    for (int q = 2; q <= q_max; q++) {
      std::cout << n << " " << q << " ";
      std::cout << memcpy_ref(in, out, n, q, num_elements) << " ";
      std::cout << naive(in, out, n, q, num_elements) << " ";
      std::cout << precompute_shape_fns(in, out, n, q, num_elements) << " ";
      std::cout << no_alloc_in_loop(in, out, n, q, num_elements) << " ";
      std::cout << std::endl;
    }
  }

}