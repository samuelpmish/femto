#include "fm/types/matrix.hpp"

#include <random>

#include "femto/threadpool.hpp"
#include "misc/timer.hpp"
#include "misc/nanobench.h"

#include "fm/types/vec.hpp"
#include "fm/types/matrix.hpp"
#include "fm/operations/random.hpp"

using namespace fm;

double tet_volume(const mat<4,3> & tet) {
  auto e1 = tet[1] - tet[0];
  auto e2 = tet[2] - tet[0];
  auto e3 = tet[3] - tet[0];
  return dot(cross(e1, e2), e3) / 6.0;
}

double tet_quality(const mat<4,3> & tet) {
  double L_rms = sqrt((norm_squared(tet[1] - tet[0]) + 
                       norm_squared(tet[2] - tet[1]) + 
                       norm_squared(tet[0] - tet[2]) + 
                       norm_squared(tet[0] - tet[3]) + 
                       norm_squared(tet[1] - tet[3]) + 
                       norm_squared(tet[2] - tet[3])) / 6.0); 
  double V = tet_volume(tet);
  return 6.0 * sqrt(2.0) *  V / (L_rms * L_rms * L_rms);
}

int main() {

  femto::timer stopwatch;

  int n = 1000000;
  std::vector< mat<4,3> > tets(n);
  for (int i = 0; i < n; i++) {
    tets[i] = fm::random_mat<4,3>();
  }

  std::vector< double > qualities(n);
  stopwatch.start();
  for (int i = 0; i < n; i++) {
    qualities[i] = tet_quality(tets[i]);
  }
  ankerl::nanobench::doNotOptimizeAway(qualities);
  stopwatch.stop();
  std::cout << stopwatch.elapsed() * 1000 << std::endl;

  stopwatch.start();
  femto::threadpool::parallel_for(n, [&](uint32_t i){
    qualities[i] = tet_quality(tets[i]);
  });
  stopwatch.stop();
  std::cout << stopwatch.elapsed() * 1000 << std::endl;

}