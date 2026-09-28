#include "misc/timer.hpp"

#include <array>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <unordered_map>

template < typename int_t >
struct array_hasher {
  template<size_t n>
  std::size_t operator()(const std::array< int_t, n > & arr) const {
    int_t seed = 0;
    for(const auto elem : arr) {
      seed ^= std::hash<int_t>()(elem) + 0x9e3779b9 + (seed<<6) + (seed>>2);
    }
    return seed;
  }
};

template < std::size_t k, typename int_t >
double sort_test(int n) {

  static std::default_random_engine gen;
  static std::uniform_int_distribution< int_t > dist(0, n);

  std::vector< std::array< int_t, k > > values(n);

  for (int i = 0; i < n; i++) {
    std::array< int_t, k > v{};
    for (int j = 0; i < k; i++) {
      v[j] = dist(gen);
    }
    values[i] = v;
  }

  femto::timer stopwatch;
  stopwatch.start();
  std::sort(values.begin(), values.end());
  stopwatch.stop();

  return stopwatch.elapsed() * 1000.0;

}

template < std::size_t k, typename int_t >
double map_test(int n) {

  static std::default_random_engine gen;
  static std::uniform_int_distribution< int_t > dist(0, n);

  std::vector< std::array< int_t, k > > values(n);
  for (int i = 0; i < n; i++) {
    std::array< int_t, k > v{};
    for (int j = 0; i < k; i++) {
      v[j] = dist(gen);
    }
    values[i] = v;
  }

  femto::timer stopwatch;
  stopwatch.start();
  std::unordered_map< std::array< int_t, k >, int_t, array_hasher<int_t> > map_of_values(n);
  map_of_values.reserve(n);

  int_t count = 0;
  for (const auto & v : values) {
    if (!map_of_values.count(v)) {
      map_of_values[v] = count++;
    }
  }
  stopwatch.stop();

  return stopwatch.elapsed() * 1000.0;

}

template < std::size_t k >
double small_std_sort_test(int n) {

  static std::default_random_engine gen;
  static std::uniform_int_distribution< uint64_t > dist(0, n);

  std::vector< std::array< uint64_t, k > > values(n);
  for (int i = 0; i < n; i++) {
    std::array< uint64_t, k > v{};
    for (int j = 0; i < k; i++) {
      v[j] = dist(gen);
    }
    values[i] = v;
  }

  femto::timer stopwatch;
  stopwatch.start();
  for (auto & v : values) {
    std::sort(v.begin(), v.end());
  }
  stopwatch.stop();

  return stopwatch.elapsed() * 1000.0;

}

/**
 * A Functor class to create a sort for fixed sized arrays/containers with a
 * compile time generated Bose-Nelson sorting network.
 * \tparam NumElements  The number of elements in the array or container to sort.
 * \tparam T            The element type.
 * \tparam Compare      A comparator functor class that returns true if lhs < rhs.
 */
template <unsigned NumElements, class Compare = void> class StaticSort
{
    template <class A, class C> struct Swap
    {
        template <class T> inline void s(T &v0, T &v1)
        {
            T t = Compare()(v0, v1) ? v0 : v1; // Min
            v1 = Compare()(v0, v1) ? v1 : v0; // Max
            v0 = t;
        }

        inline Swap(A &a, const int &i0, const int &i1) { s(a[i0], a[i1]); }
    };

    template <class A> struct Swap <A, void>
    {
        template <class T> inline void s(T &v0, T &v1)
        {
            // Explicitly code out the Min and Max to nudge the compiler
            // to generate branchless code.
            T t = v0 < v1 ? v0 : v1; // Min
            v1 = v0 < v1 ? v1 : v0; // Max
            v0 = t;
        }

        inline Swap(A &a, const int &i0, const int &i1) { s(a[i0], a[i1]); }
    };

    template <class A, class C, int I, int J, int X, int Y> struct PB
    {
        inline PB(A &a)
        {
            enum { L = X >> 1, M = (X & 1 ? Y : Y + 1) >> 1, IAddL = I + L, XSubL = X - L };
            PB<A, C, I, J, L, M> p0(a);
            PB<A, C, IAddL, J + M, XSubL, Y - M> p1(a);
            PB<A, C, IAddL, J, XSubL, M> p2(a);
        }
    };

    template <class A, class C, int I, int J> struct PB <A, C, I, J, 1, 1>
    {
        inline PB(A &a) { Swap<A, C> s(a, I - 1, J - 1); }
    };

    template <class A, class C, int I, int J> struct PB <A, C, I, J, 1, 2>
    {
        inline PB(A &a) { Swap<A, C> s0(a, I - 1, J); Swap<A, C> s1(a, I - 1, J - 1); }
    };

    template <class A, class C, int I, int J> struct PB <A, C, I, J, 2, 1>
    {
        inline PB(A &a) { Swap<A, C> s0(a, I - 1, J - 1); Swap<A, C> s1(a, I, J - 1); }
    };

    template <class A, class C, int I, int M, bool Stop = false> struct PS
    {
        inline PS(A &a)
        {
            enum { L = M >> 1, IAddL = I + L, MSubL = M - L};
            PS<A, C, I, L, (L <= 1)> ps0(a);
            PS<A, C, IAddL, MSubL, (MSubL <= 1)> ps1(a);
            PB<A, C, I, IAddL, L, MSubL> pb(a);
        }
    };

    template <class A, class C, int I, int M> struct PS <A, C, I, M, true>
    {
        inline PS(A &a) {}
    };

public:
    /**
     * Sorts the array/container arr.
     * \param  arr  The array/container to be sorted.
     */
    template <class Container> inline void operator() (Container &arr) const
    {
        PS<Container, Compare, 1, NumElements, (NumElements <= 1)> ps(arr);
    };

    /**
     * Sorts the array arr.
     * \param  arr  The array to be sorted.
     */
    template <class T> inline void operator() (T *arr) const
    {
        PS<T*, Compare, 1, NumElements, (NumElements <= 1)> ps(arr);
    };
};

template < std::size_t k >
double small_bose_nelson_sort_test(int n) {

  static std::default_random_engine gen;
  static std::uniform_int_distribution< uint64_t > dist(0, n);

  std::vector< std::array< uint64_t, k > > values(n);
  for (int i = 0; i < n; i++) {
    std::array< uint64_t, k > v{};
    for (int j = 0; i < k; i++) {
      v[j] = dist(gen);
    }
    values[i] = v;
  }

  StaticSort<k> sorter;

  femto::timer stopwatch;
  stopwatch.start();
  for (auto & v : values) {
    sorter(v);
  }
  stopwatch.stop();

  return stopwatch.elapsed() * 1000.0;

}

int main() {

  std::cout << "sort (uint32_t):" << std::endl;
  for (auto n : {1 << 15, 1 << 20, 1 << 24}) {
    std::cout << "n = " << n << " k = 2: " << sort_test<2, uint32_t>(n) << "ms" << std::endl;
    std::cout << "n = " << n << " k = 3: " << sort_test<3, uint32_t>(n) << "ms" << std::endl;
    std::cout << "n = " << n << " k = 4: " << sort_test<4, uint32_t>(n) << "ms" << std::endl;
    std::cout << "n = " << n << " k = 5: " << sort_test<5, uint32_t>(n) << "ms" << std::endl;
  }

  std::cout << "map (uint32_t):" << std::endl;
  for (auto n : {1 << 15, 1 << 20, 1 << 24}) {
    std::cout << "n = " << n << " k = 2: " << map_test<2, uint32_t>(n) << "ms" << std::endl;
    std::cout << "n = " << n << " k = 3: " << map_test<3, uint32_t>(n) << "ms" << std::endl;
    std::cout << "n = " << n << " k = 4: " << map_test<4, uint32_t>(n) << "ms" << std::endl;
    std::cout << "n = " << n << " k = 5: " << map_test<5, uint32_t>(n) << "ms" << std::endl;
  }

  std::cout << "sort (uint64_t):" << std::endl;
  for (auto n : {1 << 15, 1 << 20, 1 << 24}) {
    std::cout << "n = " << n << " k = 2: " << sort_test<2, uint64_t>(n) << "ms" << std::endl;
    std::cout << "n = " << n << " k = 3: " << sort_test<3, uint64_t>(n) << "ms" << std::endl;
    std::cout << "n = " << n << " k = 4: " << sort_test<4, uint64_t>(n) << "ms" << std::endl;
    std::cout << "n = " << n << " k = 5: " << sort_test<5, uint64_t>(n) << "ms" << std::endl;
  }

  std::cout << "map (uint64_t):" << std::endl;
  for (auto n : {1 << 15, 1 << 20, 1 << 24}) {
    std::cout << "n = " << n << " k = 2: " << map_test<2, uint64_t>(n) << "ms" << std::endl;
    std::cout << "n = " << n << " k = 3: " << map_test<3, uint64_t>(n) << "ms" << std::endl;
    std::cout << "n = " << n << " k = 4: " << map_test<4, uint64_t>(n) << "ms" << std::endl;
    std::cout << "n = " << n << " k = 5: " << map_test<5, uint64_t>(n) << "ms" << std::endl;
  }

  std::cout << "small std::sort:" << std::endl;
  for (auto n : {1 << 15, 1 << 20, 1 << 24}) {
    std::cout << "n = " << n << " k = 2: " << small_std_sort_test<2>(n) << "ms" << std::endl;
    std::cout << "n = " << n << " k = 3: " << small_std_sort_test<3>(n) << "ms" << std::endl;
    std::cout << "n = " << n << " k = 4: " << small_std_sort_test<4>(n) << "ms" << std::endl;
    std::cout << "n = " << n << " k = 5: " << small_std_sort_test<5>(n) << "ms" << std::endl;
  }

  std::cout << "small bose-nelson sort:" << std::endl;
  for (auto n : {1 << 15, 1 << 20, 1 << 24}) {
    std::cout << "n = " << n << " k = 2: " << small_bose_nelson_sort_test<2>(n) << "ms" << std::endl;
    std::cout << "n = " << n << " k = 3: " << small_bose_nelson_sort_test<3>(n) << "ms" << std::endl;
    std::cout << "n = " << n << " k = 4: " << small_bose_nelson_sort_test<4>(n) << "ms" << std::endl;
    std::cout << "n = " << n << " k = 5: " << small_bose_nelson_sort_test<5>(n) << "ms" << std::endl;
  }

}