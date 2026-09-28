#pragma once

#include "femto/assert.hpp"
#include "containers/memory.hpp"

#include <cmath>
#include <vector>
#include <iostream>
#include <cinttypes>
#include <functional>
#include <initializer_list>
#include <utility>

namespace femto {

  template < typename T, memory::space mem_space = memory::space::cpu >
  struct Tvector;

  template < typename T, memory::space mem_space = memory::space::cpu >
  struct Tvector_slice;

  template < typename T, memory::space mem_space = memory::space::cpu >
  struct Tvector_expression {
    std::function< void (T *) > evaluate;
    uint32_t sz;
  };

  template < typename T, memory::space mem_space = memory::space::cpu >
  struct Tvector_view{
    Tvector_view() { 
      sz = 0;
      ptr = nullptr;
    }

    Tvector_view(const Tvector<T, mem_space> & arr) {
      ptr = arr.ptr;
      sz = arr.sz;
    }

    Tvector_view(const Tvector_view<T, mem_space> & arr) {
      ptr = arr.ptr;
      sz = arr.sz;
    }

    Tvector_view(T * ptr_, uint32_t sz_) : ptr(ptr_), sz(sz_) {}

    void operator=(T value) {
      for (int i = 0; i < sz; i++) { ptr[i] = value; }
    }

    template < memory::space other_space >
    void operator=(const Tvector<T, other_space> & v) {
      CHECK_FOR_SIZE_MISMATCH(sz, v.sz);
      memory::memcpy<T, mem_space, other_space>(ptr, v.ptr, sz);
    }

    void operator=(const Tvector_expression<T, mem_space> & other) {
      CHECK_FOR_SIZE_MISMATCH(sz, other.sz);
      other.evaluate(ptr);
    }

    operator Tvector_view<const T, mem_space>() const {
      Tvector_view<const T, mem_space> output;
      output.ptr = ptr;
      output.sz = sz;
      return output;
    } 

    T & operator[](uint32_t i) { return ptr[i]; }
    const T & operator[](uint32_t i) const { return ptr[i]; }

    Tvector_slice<T, mem_space> operator[](const std::vector<int> & ids) { return {ptr, &ids[0], uint32_t(ids.size())}; }
    Tvector_slice<const T, mem_space> operator[](const std::vector<int> & ids) const { return {ptr, &ids[0], uint32_t(ids.size())}; }

    uint32_t size() const { return sz; }

    T * ptr;
    uint32_t sz;
  };

  template < typename T, memory::space mem_space >
  struct Tvector_view< const T, mem_space >{
    Tvector_view() { 
      sz = 0;
      ptr = nullptr;
    }

    explicit Tvector_view(const Tvector<T, mem_space> & arr) {
      ptr = arr.ptr;
      sz = arr.sz;
    }

    Tvector_view(const Tvector_view<const T, mem_space> & arr) {
      ptr = arr.ptr;
      sz = arr.sz;
    }

    Tvector_view(const T * ptr_, uint32_t sz_) : ptr(ptr_), sz(sz_) {}

    const T & operator[](uint32_t i) const { return ptr[i]; }

    Tvector_slice<const T, mem_space> operator[](const std::vector<int> & ids) const { return {ptr, &ids[0], uint32_t(ids.size())}; }

    uint32_t size() const { return sz; }

    const T * ptr;
    uint32_t sz;
  };

  template < typename T, memory::space mem_space >
  struct Tvector_slice {

    void operator=(T value) {
      for (int i = 0; i < sz; i++) { ptr[ids[i]] = value; }
    }

    void operator=(const Tvector<T, mem_space> & v) {
      CHECK_FOR_SIZE_MISMATCH(sz, v.sz);
      for (int i = 0; i < sz; i++) { ptr[ids[i]] = v[i]; }
    }

    // slice assignment is unusual-- it forwards the copying through the slices
    // rather than modifying the actual slice object itself
    void operator=(const Tvector_slice<T, mem_space> & v) {
      CHECK_FOR_SIZE_MISMATCH(sz, v.sz);
      for (int i = 0; i < sz; i++) { ptr[ids[i]] = v.ptr[v.ids[i]]; }
    }

    operator Tvector_slice<const T, mem_space>() const {
      return Tvector_slice<const T, mem_space>{ptr, ids, sz};
    }

    T * ptr;
    const int * ids;
    uint32_t sz;
  };

  template < typename T, memory::space mem_space >
  struct Tvector_slice< const T, mem_space >{
    const T * ptr;
    const int * ids;
    uint32_t sz;
  };

  // represents an expression of the form:
  // 
  // a_1 * v_1 + a_2 * v_2 + ... + a_n * v_n
  // 
  // where a_i are scalars and v_i are vectors
  template < typename T, memory::space mem_space = memory::space::cpu >
  struct Tlinear_combination {

    struct view_term {
      T coefficient;
      Tvector_view<const T, mem_space> view;
    };

    struct slice_term {
      T coefficient;
      Tvector_slice<const T, mem_space> slice;
    };

    uint32_t sz;
    std::vector< view_term > view_terms;
    std::vector< slice_term > slice_terms;

    void operator()(T * x) const;

  };

  template < typename T, memory::space mem_space >
  struct Tvector : public Tvector_view<T, mem_space> {

    using Tvector_view<T, mem_space>::sz;
    using Tvector_view<T, mem_space>::ptr;

    Tvector() : Tvector_view<T, mem_space>() {}

    explicit Tvector(uint32_t n) : Tvector_view<T, mem_space>() {
      resize(n);
    }

    Tvector(const Tvector<T, mem_space> & other) : Tvector_view<T, mem_space>() {
      resize(other.sz);
      memory::memcpy<T, mem_space, mem_space>(ptr, other.ptr, sz);
    }

    template < memory::space other_space >
    Tvector(const Tvector<T, other_space> & other) : Tvector_view<T, mem_space>() {
      resize(other.sz);
      memory::memcpy<T, mem_space, other_space>(ptr, other.ptr, sz);
    }

    Tvector(Tvector<T, mem_space> && other) : Tvector_view<T, mem_space>() {
      ptr = other.ptr;
      sz = other.sz;
      other.ptr = nullptr;
      other.sz = 0;
    }

    Tvector(const Tvector_view<T, mem_space> & other) : Tvector_view<T, mem_space>() {
      resize(other.sz);
      memory::memcpy<T, mem_space, mem_space>(ptr, other.ptr, sz);
    }

    template < memory::space other_space >
    Tvector(const Tvector_view<T, other_space> & other) : Tvector_view<T, mem_space>() {
      resize(other.sz);
      memory::memcpy<T, mem_space, other_space>(ptr, other.ptr, sz);
    }

    Tvector(const Tvector_view<const T, mem_space> & other) : Tvector_view<T, mem_space>() {
      resize(other.sz);
      memory::memcpy<T, mem_space, mem_space>(ptr, other.ptr, sz);
    }

    template < memory::space other_space >
    Tvector(const Tvector_view<const T, other_space> & other) : Tvector_view<T, mem_space>() {
      resize(other.sz);
      memory::memcpy<T, mem_space, other_space>(ptr, other.ptr, sz);
    }

    Tvector(const Tlinear_combination<T, mem_space> & expr) : Tvector_view<T, mem_space>() {
      resize(expr.sz);
      expr(ptr);
    }

    Tvector(const Tvector_slice<T, mem_space> & other) : Tvector_view<T, mem_space>() {
      resize(other.sz);
      for (uint32_t i = 0; i < sz; i++) {
        ptr[i] = other.ptr[other.ids[i]];
      }
    }

    Tvector(const Tvector_expression<T, mem_space> & other) : Tvector_view<T, mem_space>() {
      resize(other.sz);
      other.evaluate(ptr);
    }

    template < int n >
    Tvector(const T (&values)[n]) : Tvector_view<T, mem_space>() {
      copy_from_cpu(values, n);
    }

    Tvector(std::initializer_list<T> values) : Tvector_view<T, mem_space>() {
      std::vector<T> buffer(values);
      copy_from_cpu(buffer.data(), buffer.size());
    }

    Tvector(std::initializer_list<std::initializer_list<T>> values) : Tvector_view<T, mem_space>() {
      uint32_t size = 0;
      for (const auto & row : values) {
        size += row.size();
      }

      std::vector<T> buffer;
      buffer.reserve(size);
      for (const auto & row : values) {
        for (T value : row) {
          buffer.push_back(value);
        }
      }
      copy_from_cpu(buffer.data(), buffer.size());
    }

    ~Tvector() {
      if (ptr) {
        memory::deallocate<T, mem_space>(ptr);
      }
    }

    Tvector<T, mem_space>& operator=(const Tvector_view<T, mem_space> & other) {
      resize(other.sz);
      memory::memcpy<T, mem_space, mem_space>(ptr, other.ptr, sz);
      return *this;
    }

    template < memory::space other_space >
    Tvector<T, mem_space>& operator=(const Tvector_view<T, other_space> & other) {
      resize(other.sz);
      memory::memcpy<T, mem_space, other_space>(ptr, other.ptr, sz);
      return *this;
    }

    Tvector<T, mem_space>& operator=(const Tvector_view<const T, mem_space> & other) {
      resize(other.sz);
      memory::memcpy<T, mem_space, mem_space>(ptr, other.ptr, sz);
      return *this;
    }

    template < memory::space other_space >
    Tvector<T, mem_space>& operator=(const Tvector_view<const T, other_space> & other) {
      resize(other.sz);
      memory::memcpy<T, mem_space, other_space>(ptr, other.ptr, sz);
      return *this;
    }

    Tvector<T, mem_space>& operator=(const Tvector<T, mem_space> & other) {
      resize(other.sz);
      memory::memcpy<T, mem_space, mem_space>(ptr, other.ptr, sz);
      return *this;
    }

    template < memory::space other_space >
    Tvector<T, mem_space>& operator=(const Tvector<T, other_space> & other) {
      resize(other.sz);
      memory::memcpy<T, mem_space, other_space>(ptr, other.ptr, sz);
      return *this;
    }

    Tvector<T, mem_space>& operator=(Tvector<T, mem_space> && other) {
      if (this != &other) {
        if (ptr) {
          memory::deallocate<T, mem_space>(ptr);
        }
        ptr = other.ptr;
        sz = other.sz;
        other.ptr = nullptr;
        other.sz = 0;
      }
      return *this;
    }

    auto & operator[](uint32_t i) { return ptr[i]; }
    const auto & operator[](uint32_t i) const { return ptr[i]; }

    Tvector_slice<T, mem_space> operator[](const std::vector<int> & ids) { return {ptr, &ids[0], uint32_t(ids.size())}; }
    Tvector_slice<const T, mem_space> operator[](const std::vector<int> & ids) const { return {ptr, &ids[0], uint32_t(ids.size())}; }

   private:
    void copy_from_cpu(const T * values, uint32_t new_sz) {
      resize(new_sz);
      if (sz > 0) {
        memory::memcpy<T, mem_space, memory::space::cpu>(ptr, values, sz);
      }
    }

    void resize(uint32_t new_sz) {
      if (sz != new_sz) {
        if (ptr) {
          memory::deallocate<T, mem_space>(ptr);
          ptr = nullptr;
        }
        if (new_sz > 0) { ptr = memory::allocate<T, mem_space>(new_sz); }
        sz = new_sz;
      }
    }

  };

  using vector = Tvector<double, memory::space::cpu>;
  using vector_view = Tvector_view<double, memory::space::cpu>;
  using vector_slice = Tvector_slice<double, memory::space::cpu>;
  using vector_expr = Tvector_expression<double, memory::space::cpu>;
  using const_vector_view = Tvector_view<const double, memory::space::cpu>;
  using const_vector_slice = Tvector_slice<const double, memory::space::cpu>;
  using linear_combination = Tlinear_combination<double, memory::space::cpu>;

  using gpu_vector = Tvector<double, memory::space::gpu>;
  using gpu_vector_view = Tvector_view<double, memory::space::gpu>;
  using gpu_const_vector_view = Tvector_view<const double, memory::space::gpu>;

  vector import_vector(std::string filename);
  void export_vector(const_vector_view view, std::string filename);

  vector ones(int n);
  vector zeros(int n);

  void operator+=(vector_view u, const const_vector_view v);
  void operator-=(vector_view u, const const_vector_view v);
  void operator*=(vector_view u, double scale);
  void operator/=(vector_view u, double scale);

  double dot(const const_vector_view & u, const const_vector_view & v);

////////////////////////////////////////////////////////////////////////////////

  std::ostream & operator<<(std::ostream & out, const_vector_view v);
  std::ostream & operator<<(std::ostream & out, const_vector_slice v);
  std::ostream & operator<<(std::ostream & out, linear_combination && v);

////////////////////////////////////////////////////////////////////////////////

  double total(const_vector_view v);
  double total(const_vector_slice v);
  double total(linear_combination && v);

////////////////////////////////////////////////////////////////////////////////

  double norm(const_vector_view v);
  double norm(const_vector_slice v);
  double norm(linear_combination && v);

////////////////////////////////////////////////////////////////////////////////

  linear_combination operator-(const_vector_view u);
  linear_combination operator-(const_vector_slice u);
  linear_combination&& operator-(linear_combination&& c);

////////////////////////////////////////////////////////////////////////////////

  linear_combination operator+(const_vector_view c1, const_vector_view c2);
  linear_combination operator+(const_vector_view c1, const_vector_slice c2);
  linear_combination&& operator+(const_vector_view c1, linear_combination && c2);
  linear_combination operator+(const_vector_slice c1, const_vector_view c2);
  linear_combination operator+(const_vector_slice c1, const_vector_slice c2);
  linear_combination&& operator+(const_vector_slice c1, linear_combination && c2);
  linear_combination&& operator+(linear_combination && c1, const_vector_view c2);
  linear_combination&& operator+(linear_combination && c1, const_vector_slice c2);
  linear_combination&& operator+(linear_combination && c1, linear_combination && c2);

////////////////////////////////////////////////////////////////////////////////

  linear_combination operator-(const_vector_view c1, const_vector_view c2);
  linear_combination operator-(const_vector_view c1, const_vector_slice c2);
  linear_combination&& operator-(const_vector_view c1, linear_combination && c2);
  linear_combination operator-(const_vector_slice c1, const_vector_view c2);
  linear_combination operator-(const_vector_slice c1, const_vector_slice c2);
  linear_combination&& operator-(const_vector_slice c1, linear_combination && c2);
  linear_combination&& operator-(linear_combination && c1, const_vector_view c2);
  linear_combination&& operator-(linear_combination && c1, const_vector_slice c2);
  linear_combination&& operator-(linear_combination && c1, linear_combination && c2);

////////////////////////////////////////////////////////////////////////////////

  linear_combination operator*(const_vector_view v, double scale);
  linear_combination operator*(double scale, const_vector_view v);
  linear_combination operator*(const_vector_slice v, double scale);
  linear_combination operator*(double scale, const_vector_slice v);
  linear_combination&& operator*(linear_combination&& v, double scale);
  linear_combination&& operator*(double scale, linear_combination&& v);

////////////////////////////////////////////////////////////////////////////////

  linear_combination operator/(const_vector_view v, double scale);
  linear_combination operator/(const_vector_slice v, double scale);
  linear_combination&& operator/(linear_combination&& v, double scale);

////////////////////////////////////////////////////////////////////////////////

}

