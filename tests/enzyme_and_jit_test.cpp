#if FEMTO_ENABLE_ENZYME && FEMTO_ENABLE_JIT
#include <iostream>

#include "JIT.hpp"

int main() {
  JIT jit({"-O3", ENZYME_PLUGIN_FLAG /* compilation flags */, "-v"});

  std::cout << "enzyme boilerplate" << std::endl;
  jit.compile(R"(
    int enzyme_dupnoneed;
    int enzyme_dup;
    int enzyme_out;
    int enzyme_const;

    template < typename return_type, typename ... T >
    extern return_type __enzyme_fwddiff(void*, T ... );

    template < typename return_type, typename ... T >
    extern return_type __enzyme_autodiff(void*, T ... );
  )");

  std::cout << "compiling function and its derivative" << std::endl;
  jit.compile(R"(
    extern "C" {
      double foo(double x) {
        return x * x;
      }

      double dfoo(double x) {
        return __enzyme_autodiff<double>((void*)foo, x);
      }
    }
  )");

  std::cout << "get function pointer to specialization" << std::endl;
  double (*foo)(double) = jit.lookup_function<double(*)(double)>("foo");
  double (*dfoo)(double) = jit.lookup_function<double(*)(double)>("dfoo");

  double x = 3.0;

  std::cout << "calling foo()" << std::endl;
  std::cout << foo(x) << ", expected: " << x * x << std::endl;

  std::cout << "calling dfoo()" << std::endl;
  std::cout << dfoo(x) << ", expected: " << 2 * x << std::endl;
 
  return 0;
}
#endif