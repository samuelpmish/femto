
#include "forall.hpp"

#include "fm/operations/print.hpp"

using namespace femto;

int main() {

  uint32_t n = 100;
  nd::cpu_array<double, 3> v_q({n, 1, 2});
  nd::cpu_array<double, 3> A_q({n, 2, 2});

  for (int i = 0; i < n; i++) {
    v_q(i,0,0) = i + 1;
    v_q(i,0,1) = i + 2;

    A_q(i,0,0) = i + 1;
    A_q(i,0,1) = i + 2;
    A_q(i,1,0) = i + 3;
    A_q(i,1,1) = i + 4;
  }

  forall(+[](const vec2 & v, const mat2 & A){ 
    std::cout << v << std::endl;
    std::cout << A << std::endl;
    return v;
  }, v_q, A_q);

}