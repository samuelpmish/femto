#if FEMTO_ENABLE_ENZYME
#include "gtest/gtest.h"

#include <stdio.h>
#include <tuple>

int enzyme_dupnoneed;
int enzyme_dup;
int enzyme_out;
int enzyme_const;

template < typename return_type, typename ... T >
extern return_type __enzyme_fwddiff(void*, T ... );

template < typename return_type, typename ... T >
extern return_type __enzyme_autodiff(void*, T ... );

namespace enzyme {

  template < typename T >
  struct inactive{ T value; };

  template < typename S, typename T >
  struct active{
    S value;
    T shadow;
  };

  template < typename S, typename T >
  active(S, T) -> active<S, T>;

  template < typename return_type, typename ... arg_types, typename ... T >
  auto forward(return_type (* func)(arg_types ...), T && ... args) {
    return func(args.value ...);
  }

}

namespace enz = enzyme;

extern double __enzyme_autodiff(void*, double);

double foo(void*, double);

extern double bar(void*, double);

double square(double x) {
  return x * x;
}

double dsquare(double x) {
  return __enzyme_autodiff((void*) square, x);
}

TEST(enzyme, basic_test) {
  for(double i=1; i<5; i++) {
    EXPECT_EQ(square(i), i*i);

    //EXPECT_EQ(std::invoke(bar, (void*)square, i*i), 2*i);
    //EXPECT_EQ(std::apply(__enzyme_autodiff, std::tuple((void*)(square), i*i)), 2*i);
    //EXPECT_EQ(std::apply(foo, std::tuple((void*)(square), i*i)), 2*i);
  }
}
#endif