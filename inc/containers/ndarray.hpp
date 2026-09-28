#pragma once

#include <cstdio>  // for printf
#include <cstring> // for std::memcpy
#include <utility> // for std::integer_sequence
#include <functional> // for the deferred-write proxy, see operator=(writer)
#include <iostream>
#include <cinttypes>
#include <type_traits>

#include "misc/macros.hpp"
#include "containers/memory.hpp"
#include "containers/stack_array.hpp"

namespace nd {

  template < uint32_t dim >
  HOSTDEV uint32_t product(stack::array< uint32_t, dim > values) {
    uint32_t p = values[0];
    for (int i = 1; i < dim; i++) { p *= values[i]; }
    return p;
  }

  enum ordering{ row_major, col_major };

  template < uint32_t dim >
  HOSTDEV stack::array< uint32_t, dim > compute_strides(const stack::array< uint32_t, dim > & shape, ordering o, uint32_t m = 1) {
    stack::array< uint32_t, dim > strides{};
    int32_t k = (o == col_major) ? 0 : dim-1;
    int32_t s = (o == col_major) ? 1 : -1;
    for (uint32_t i = 0; i < dim; i++) {
      strides[k] = (i == 0) ? m : strides[k-s] * shape[k-s];
      k += s;
    }
    return strides;
  }

  // used for slicing arrays and views
  template < typename T >
  struct range { T begin; T end; };

  template < typename T >
  range(T,T) -> range<T>;

  template < typename T >
  HOSTDEV constexpr T begin(const range<T> & r) { return r.begin; }

  template < typename T >
  HOSTDEV constexpr T end(const range<T> & r) { return r.end; }

  template < typename T >
  HOSTDEV constexpr uint32_t is_range(T) { return 0; };

  template < typename T >
  HOSTDEV constexpr uint32_t is_range(range<T>) { return 1; };

  HOSTDEV constexpr int32_t begin(int32_t x) { return x; }
  HOSTDEV constexpr int64_t begin(int64_t x) { return x; }
  HOSTDEV constexpr uint32_t begin(uint32_t x) { return x; }
  HOSTDEV constexpr uint64_t begin(uint64_t x) { return x; }

  HOSTDEV constexpr int32_t end(int32_t x) { return x; }
  HOSTDEV constexpr int64_t end(int64_t x) { return x; }
  HOSTDEV constexpr uint32_t end(uint32_t x) { return x; }
  HOSTDEV constexpr uint64_t end(uint64_t x) { return x; }

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

  template < typename T, uint32_t dim = 1, memory::space mem_space = memory::space::cpu >
  struct view {
    static_assert(dim > 0);
    static constexpr memory::space memory_space = mem_space;

    static constexpr auto iseq = std::make_integer_sequence< uint32_t, dim >();

    HOSTDEV constexpr view() {
      values = nullptr;
      sz = 0;
      shape = {};
      stride = {};
    }

    HOSTDEV constexpr view(T * input, const stack::array< uint32_t, dim > & dimensions) {
      values = input;
      for (uint32_t i = 0; i < dim; i++) {
        uint32_t id = dim - 1 - i;
        shape[id] = dimensions[id];
        stride[id] = (id == dim - 1) ? 1 : stride[id+1] * shape[id+1];
      }
      sz = product(shape);
    }

    HOSTDEV constexpr view(T * input, const stack::array< uint32_t, dim > & dimensions, const stack::array< uint32_t, dim > & strides) {
      values = input;
      shape = dimensions;
      stride = strides;
      sz = product(shape);
    }

    template < typename ... index_types >
    HOSTDEV uint32_t index(index_types ... indices) const { 
      static_assert(sizeof ... (indices) == dim);
      return values[index(iseq, indices...)];
    }

    template < uint32_t ... I, typename ... index_types >
    HOSTDEV uint32_t index(std::integer_sequence<uint32_t, I...>, index_types ... indices) const {
      #ifdef NDARRAY_ENABLE_BOUNDS_CHECKING
        // note: the cast to int32_t is a way to avoid warnings 
        // about pointless comparison between unsigned integer types and 0
        if (((int32_t(indices) < 0 || indices >= shape[I]) || ... )) {
          printf("array index out of bounds\n");
        };
      #endif
      return ((indices * stride[I]) + ...);
    }

    template < typename ... index_types >
    HOSTDEV decltype(auto) operator()(index_types ... indices) const { 
      constexpr uint32_t num_args = sizeof ... (indices);
      static_assert(num_args <= dim);
      constexpr uint32_t rank = (is_range(index_types{}) + ... ) + (dim - num_args);
      if constexpr (rank == 0) {
        return (T&)values[index(iseq, indices...)];
      } else {
        constexpr uint32_t is_a_range[] = {is_range(index_types{}) ... };
        stack::array<uint32_t, rank> slice_shape{};
        stack::array<uint32_t, rank> slice_stride{};

        uint32_t beginnings[num_args] = {uint32_t(nd::begin(indices)) ... };
        uint32_t endings[num_args] = {uint32_t(nd::end(indices)) ... };
        int k = 0;
        int offset = 0;
        for (int i = 0; i < dim; i++) {
          if (i >= num_args) {
            slice_shape[k] = shape[i];
            slice_stride[k] = stride[i];
            k++;
          } else {
            offset += stride[i] * beginnings[i];
            if (is_a_range[i]) {
              slice_shape[k] = endings[i] - beginnings[i];
              slice_stride[k] = stride[i];
              k++;
            }
          }
        }
        return view<T, rank, mem_space>{&values[offset], slice_shape, slice_stride};
      }
    }

    template < typename ... index_types >
    HOSTDEV decltype(auto) operator()(index_types ... indices) { 
      constexpr uint32_t num_args = sizeof ... (indices);
      static_assert(num_args <= dim);
      constexpr uint32_t rank = (is_range(index_types{}) + ... ) + (dim - num_args);
      if constexpr (rank == 0) {
        return (T&)values[index(iseq, indices...)];
      } else {
        constexpr uint32_t is_a_range[] = {is_range(index_types{}) ... };
        stack::array<uint32_t, rank> slice_shape{};
        stack::array<uint32_t, rank> slice_stride{};

        uint32_t beginnings[num_args] = {uint32_t(nd::begin(indices)) ... };
        uint32_t endings[num_args] = {uint32_t(nd::end(indices)) ... };
        int k = 0;
        int offset = 0;
        for (int i = 0; i < dim; i++) {
          if (i >= num_args) {
            slice_shape[k] = shape[i];
            slice_stride[k] = stride[i];
            k++;
          } else {
            offset += stride[i] * beginnings[i];
            if (is_a_range[i]) {
              slice_shape[k] = endings[i] - beginnings[i];
              slice_stride[k] = stride[i];
              k++;
            }
          }
        }
        return view<T, rank, mem_space>{&values[offset], slice_shape, slice_stride};
      }
    }

    template < typename index_type >
    HOSTDEV auto & operator[](index_type i) { return values[i]; }

    template < typename index_type >
    HOSTDEV auto & operator[](index_type i) const { return values[i]; }

    HOSTDEV operator view<const T, dim, mem_space>() const {
      return view<const T, dim, mem_space>{values, shape, stride};
    }

    template < memory::space target_space, typename = std::enable_if_t<mem_space == memory::space::unified && target_space == memory::space::cpu> >
    HOSTDEV operator view<T, dim, target_space>() const {
      return view<T, dim, target_space>{values, shape, stride};
    }

    template < memory::space target_space, typename = std::enable_if_t<mem_space == memory::space::unified && target_space == memory::space::cpu> >
    HOSTDEV operator view<const T, dim, target_space>() const {
      return view<const T, dim, target_space>{values, shape, stride};
    }

    HOSTDEV T * data() { return &values[0]; }
    HOSTDEV const T * data() const { return &values[0]; }

    HOSTDEV uint32_t size() const { return product(shape); }

    HOSTDEV T * begin() const { return &values[0]; }
    HOSTDEV T * end() const { return &values[size()]; }

    T * values;
    uint64_t sz;
    stack::array< uint32_t, dim > shape;
    stack::array< uint32_t, dim > stride;

  };

  template < typename T, uint32_t dim, memory::space mem_space >
  struct view< const T, dim, mem_space >{
    static_assert(dim > 0);
    static constexpr memory::space memory_space = mem_space;

    static constexpr auto iseq = std::make_integer_sequence< uint32_t, dim >();

    HOSTDEV constexpr view() {
      values = nullptr;
      sz = 0;
      shape = {};
      stride = {};
    }

    HOSTDEV constexpr view(const T * input, const stack::array< uint32_t, dim > & dimensions) {
      values = input;
      for (uint32_t i = 0; i < dim; i++) {
        uint32_t id = dim - 1 - i;
        shape[id] = dimensions[id];
        stride[id] = (id == dim - 1) ? 1 : stride[id+1] * shape[id+1];
      }
      sz = product(shape);
    }

    HOSTDEV constexpr view(const T * input, const stack::array< uint32_t, dim > & dimensions, const stack::array< uint32_t, dim > & strides) {
      values = input;
      shape = dimensions;
      stride = strides;
      sz = product(shape);
    }

    template < typename ... index_types >
    HOSTDEV uint32_t index(index_types ... indices) const { 
      static_assert(sizeof ... (indices) == dim);
      return values[index(iseq, indices...)];
    }

    template < uint32_t ... I, typename ... index_types >
    HOSTDEV uint32_t index(std::integer_sequence<uint32_t, I...>, index_types ... indices) const {
      #ifdef NDARRAY_ENABLE_BOUNDS_CHECKING
        // note: the cast to int32_t is a way to avoid warnings 
        // about pointless comparison between unsigned integer types and 0
        if (((int32_t(indices) < 0 || indices >= shape[I]) || ... )) {
          printf("array index out of bounds\n");
        };
      #endif
      return ((indices * stride[I]) + ...);
    }

    template < typename ... index_types >
    HOSTDEV decltype(auto) operator()(index_types ... indices) const { 
      constexpr uint32_t num_args = sizeof ... (indices);
      static_assert(num_args <= dim);
      constexpr uint32_t rank = (is_range(index_types{}) + ... ) + (dim - num_args);
      if constexpr (rank == 0) {
        return (T&)values[index(iseq, indices...)];
      } else {
        constexpr uint32_t is_a_range[] = {is_range(index_types{}) ... };
        stack::array<uint32_t, rank> slice_shape{};
        stack::array<uint32_t, rank> slice_stride{};

        uint32_t beginnings[num_args] = {uint32_t(nd::begin(indices)) ... };
        uint32_t endings[num_args] = {uint32_t(nd::end(indices)) ... };
        int k = 0;
        int offset = 0;
        for (int i = 0; i < dim; i++) {
          if (i >= num_args) {
            slice_shape[k] = shape[i];
            slice_stride[k] = stride[i];
            k++;
          } else {
            offset += stride[i] * beginnings[i];
            if (is_a_range[i]) {
              slice_shape[k] = endings[i] - beginnings[i];
              slice_stride[k] = stride[i];
              k++;
            }
          }
        }
        return view<const T, rank, mem_space>{&values[offset], slice_shape, slice_stride};
      }
    }

    template < typename index_type >
    HOSTDEV const T & operator[](index_type i) const { return values[i]; }

    template < memory::space target_space, typename = std::enable_if_t<mem_space == memory::space::unified && target_space == memory::space::cpu> >
    HOSTDEV operator view<const T, dim, target_space>() const {
      return view<const T, dim, target_space>{values, shape, stride};
    }

    HOSTDEV const T * data() const { return &values[0]; }

    HOSTDEV uint32_t size() const { return product(shape); }

    HOSTDEV const T * begin() const { return &values[0]; }
    HOSTDEV const T * end() const { return &values[size()]; }

    const T * values;
    uint64_t sz;
    stack::array< uint32_t, dim > shape;
    stack::array< uint32_t, dim > stride;

  };

  template < typename T, uint32_t dim, memory::space mem_space = memory::space::cpu >
  using const_view = view<const T, dim, mem_space>;

  template < typename T, uint32_t dim, memory::space mem_space >
  struct array : public view< T, dim, mem_space > {
    static constexpr memory::space memory_space = mem_space;

    using view<T,dim,mem_space>::iseq;
    using view<T,dim,mem_space>::values;
    using view<T,dim,mem_space>::shape;
    using view<T,dim,mem_space>::stride;
    using view<T,dim,mem_space>::sz;

    array() : view<T,dim,mem_space>() {}
    
    array(stack::array< uint32_t, dim > dimensions) : view<T,dim,mem_space>() { resize(dimensions); }

    array(stack::array< uint32_t, dim > dimensions,
          stack::array< uint32_t, dim > strides) : view<T,dim,mem_space>() {
      resize(dimensions, strides);
    }

    array(const array & other)  {
      sz = other.sz;
      shape = other.shape;
      stride = other.stride;
      allocate(sz);
      memory::memcpy<T, mem_space, mem_space>(values, other.values, other.sz);
    }

    template < memory::space other_space >
    array(const array<T, dim, other_space> & other)  {
      sz = other.sz;
      shape = other.shape;
      stride = other.stride;
      allocate(sz);
      memory::memcpy<T, mem_space, other_space>(values, other.values, other.sz);
    }

    void operator=(const array & other) {
      shape = other.shape;
      stride = other.stride;
      _resize(other.sz);
      memory::memcpy<T, mem_space, mem_space>(values, other.values, sz);
    }

    template < memory::space other_space >
    void operator=(const array<T, dim, other_space> & other) {
      shape = other.shape;
      stride = other.stride;
      _resize(other.sz);
      memory::memcpy<T, mem_space, other_space>(values, other.values, sz);
    }

    array(array && other) {
      sz = other.sz;
      shape = other.shape;
      values = other.values;
      stride = other.stride;
      other.values = nullptr;
    }

    void operator=(array && other) {
      sz = other.sz;
      shape = other.shape;

      deallocate();
      values = other.values;

      stride = other.stride;
      other.values = nullptr;
    }

    // A producer that fills an array returns one of these instead of an array, so
    // that `arr = producer(...)` writes into arr's existing storage rather than
    // allocating a new buffer and throwing the old one away.  sparse_matrix has
    // carried the same pair since assembly was written; see integrate_spmat.hpp.
    // The writer runs immediately, so it may hold references to anything alive
    // for the full-expression it was created in.
    //
    // It wraps the std::function rather than being one: view::operator() is an
    // unconstrained variadic template, so an implicit array -> std::function
    // conversion would be a candidate for every `array x = ...` and hard-error
    // while std::function checked whether an array is callable.
    struct writer {
      std::function< void(array &) > fill;
      // a real constructor rather than an aggregate, so that a multi-element
      // `array a({m, n})` cannot list-initialize a writer and go ambiguous with
      // the shape constructor.  One case still is: a 1D `array a({0})`, because
      // a literal zero is a null pointer constant and std::function has a
      // constructor taking one.  Default-construct instead -- shape is {0} already.
      explicit writer(std::function< void(array &) > f) : fill(std::move(f)) {}
    };

    array(const writer & w) : view<T,dim,mem_space>() { w.fill(*this); }

    void operator=(const writer & w) { w.fill(*this); }

    ~array() { 
      deallocate(); 
    }

    void resize(uint32_t new_size) {
      static_assert(dim == 1, "resize(uint32_t) only defined for 1D arrays");
      stride[0] = 1;
      shape[0] = new_size;
      _resize(new_size);
    }

    void resize(const stack::array< uint32_t, dim > & new_shape) {
      shape = new_shape;
      _resize(product(shape));
      for (uint64_t i = 0; i < dim; i++) {
        uint64_t id = dim - 1 - i;
        stride[id] = (id == dim - 1) ? 1 : stride[id+1] * shape[id+1];
      }
    }

    void resize(const stack::array< uint32_t, dim > & new_shape, 
                const stack::array< uint32_t, dim > & new_stride) {
      shape = new_shape;
      stride = new_stride;
      _resize(product(shape));
    }

   private:

    void allocate(uint32_t n) { 
      values = memory::allocate<T, mem_space>(n);
    }

    void deallocate() { 
      if (values) { 
        memory::deallocate<T, mem_space>(values);
        values = nullptr;
      } 
    }

    void _resize(uint32_t new_sz) {
      if (new_sz != sz) {
        deallocate();
        allocate(new_sz);
        sz = new_sz;
        memory::zero<T, mem_space>(values, sz);
      }
    }

  };

  template < memory::space target_space, typename T, uint32_t dim, memory::space source_space >
  array<T, dim, target_space> copy_to(const array<T, dim, source_space> & input) {
    return array<T, dim, target_space>(input);
  }

  template < typename T, uint32_t dim >
  using cpu_array = array<T, dim, memory::space::cpu>;

#ifdef NDARRAY_ENABLE_CUDA
  template < typename T, uint32_t dim >
  using gpu_array = array<T, dim, memory::space::gpu>;
#endif

  template < typename T, uint32_t dim >
  view(T*, stack::array<uint32_t,dim>) -> view<T,dim>;

  template < typename T, uint32_t dim >
  view(T*, stack::array<uint32_t,dim>, stack::array<uint32_t,dim>) -> view<T,dim>;

  /////////////////////////////////////////////////////////////////////////////

  template < typename T, uint32_t dim, memory::space mem_space >
  void zero(array<T, dim, mem_space> & arr) {
    memory::zero<T, mem_space>(arr.data(), arr.sz);
  }

//  template < typename T, uint32_t dim > 
//  void fill(array<T, dim> & arr, T value) {
//    for (uint32_t i = 0; i < arr.size(); i++) {
//      arr[i] = value;
//    }
//  }

  /////////////////////////////////////////////////////////////////////////////

  template < typename T, uint32_t dim, memory::space mem_space >
  HOSTDEV view<const T,1,mem_space> flatten(const array<T, dim, mem_space> & arr) {
    return view<const T,1,mem_space>{&arr.values[0], {product(arr.shape)}};
  }

  template < typename T, uint32_t dim, memory::space mem_space >
  HOSTDEV view<T,1,mem_space> flatten(array<T, dim, mem_space> & arr) {
    return view<T,1,mem_space>{&arr.values[0], {product(arr.shape)}};
  }

  template < typename T, uint32_t dim, memory::space mem_space >
  HOSTDEV view<T,1,mem_space> flatten(view<T, dim, mem_space> arr) {
    return view<T,1,mem_space>{arr.values, {product(arr.shape)}};
  }

  /////////////////////////////////////////////////////////////////////////////

  template < uint32_t dim, typename T, memory::space mem_space >
  HOSTDEV view<T, dim, mem_space> reshape(view<T, 1, mem_space> v, stack::array< uint32_t, dim > new_dimensions, ordering o = ordering::row_major) {
    #ifdef NDARRAY_ENABLE_BOUNDS_CHECKING
      if (product(new_dimensions) != v.shape[0]) {
        printf("reshaping view into incompatible shape");
      };
    #endif

    auto strides = nd::compute_strides(new_dimensions, o, v.stride[0]); 

    return view<T,dim,mem_space>{v.data(), new_dimensions, strides};
  }

  /////////////////////////////////////////////////////////////////////////////

};

double relative_error(nd::view<const double, 1> a, nd::view<const double,1> b);
double relative_error(nd::view<const double, 2> a, nd::view<const double,2> b);
double relative_error(nd::view<const double, 3> a, nd::view<const double,3> b);
double relative_error(nd::view<const double, 4> a, nd::view<const double,4> b);
double relative_error(nd::view<const double, 5> a, nd::view<const double,5> b);

#ifdef NDARRAY_ENABLE_CUDA
double relative_error(nd::view<const double, 1, memory::space::gpu> a, nd::view<const double, 1, memory::space::gpu> b);
double relative_error(nd::view<const double, 2, memory::space::gpu> a, nd::view<const double, 2, memory::space::gpu> b);
double relative_error(nd::view<const double, 3, memory::space::gpu> a, nd::view<const double, 3, memory::space::gpu> b);
double relative_error(nd::view<const double, 4, memory::space::gpu> a, nd::view<const double, 4, memory::space::gpu> b);
double relative_error(nd::view<const double, 5, memory::space::gpu> a, nd::view<const double, 5, memory::space::gpu> b);
#endif

namespace nd {
template < typename T > 
struct printer;

template <> struct printer<float>{ static HOSTDEV void print(float x) { printf("%f", x); } };
template <> struct printer<double>{ static HOSTDEV void print(double x) { printf("%f", x); } };
template <> struct printer<int8_t>{ static HOSTDEV void print(int8_t x) { printf("%d", x); } };
template <> struct printer<uint8_t>{ static HOSTDEV void print(uint8_t x) { printf("%u", x); } };
template <> struct printer<int32_t>{ static HOSTDEV void print(int32_t x) { printf("%d", x); } };
template <> struct printer<uint32_t>{ static HOSTDEV void print(uint32_t x) { printf("%u", x); } };
template <> struct printer<int64_t>{ static HOSTDEV void print(int64_t x) { printf("%" PRId64, x); } };
template <> struct printer<uint64_t>{ static HOSTDEV void print(uint64_t x) { printf("%" PRIu64, x); } };

template < typename T, uint32_t dim, memory::space mem_space >
HOSTDEV void print_recursive(nd::view< T, dim, mem_space > arr, int depth) {
  using S = typename std::remove_const<T>::type;

  if constexpr (dim == 1) {
    for (int i = 0; i < depth; i++) printf("  ");
    printf("{");
    for (int i = 0; i < arr.shape[0]; i++) {
      printer<S>::print(arr(i));
      if (i != arr.shape[0] - 1) { printf(","); }
    }
    printf("}");
  } else {
    const T * ptr = arr.data();
    stack::array< uint32_t, dim - 1 > shape;
    stack::array< uint32_t, dim - 1 > stride;
    for (int i = 0; i < dim - 1; i++) {
      shape[i] = arr.shape[i+1];
      stride[i] = arr.stride[i+1];
    }
    nd::view<const T, dim - 1, mem_space> slice{ptr, shape, stride};

    for (int i = 0; i < depth; i++) printf("  ");
    printf("{");
    if (arr.shape[0] == 1) {
      print_recursive(slice, 0);
    } else {
      printf("\n");
      for (int i = 0; i < arr.shape[0]; i++) {
        print_recursive(slice, depth+1);
        if (i != arr.shape[0] - 1) { printf(","); }
        printf("\n");
        slice.values += arr.stride[0];
      }
      for (int i = 0; i < depth; i++) printf("  ");
    }
    printf("}");
  }
  if (depth == 0) { printf("\n"); }
}

}

template < typename T, uint32_t dim, memory::space mem_space >
HOSTDEV void print(nd::view< T, dim, mem_space > arr) {
  nd::print_recursive(arr, 0);
}

template < typename T, uint32_t dim, memory::space mem_space >
std::ostream& operator<<(std::ostream & out, const nd::view< T, dim, mem_space > & arr) {
  print(nd::view<const T, dim, mem_space>(arr));
  return out;
}

template < typename T, uint32_t dim, memory::space mem_space >
std::ostream& operator<<(std::ostream & out, const nd::array< T, dim, mem_space > & arr) {
  print(nd::view<const T, dim, mem_space>(arr));
  return out;
}
