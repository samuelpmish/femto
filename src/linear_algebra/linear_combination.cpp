#include "linear_algebra/vector.hpp"

#include "femto/assert.hpp"

namespace femto {

  using u32 = uint32_t;

  template <>
  void Tlinear_combination<double>::operator()(double * x) const { 
    for (int i = 0; i < sz; i++) {
      double sum = 0.0;
      for (auto & [c, v] : view_terms) {
        sum += c * v[i];
      }

      for (auto & [c, v] : slice_terms) {
        sum += c * v.ptr[v.ids[i]];
      }
      x[i] = sum;
    }
  }

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

  linear_combination operator-(const_vector_view u) {
    linear_combination c{};
    c.sz = u.sz;
    c.view_terms.push_back({-1.0, u});
    return c;
  }

  linear_combination operator-(const_vector_slice u) {
    linear_combination c{};
    c.sz = u.sz;
    c.slice_terms.push_back({-1.0, u});
    return c;
  }

  linear_combination&& operator-(linear_combination&& c) {
    for (auto & [a, v] : c.view_terms) { a *= -1; };
    for (auto & [a, v] : c.slice_terms) { a *= -1; };
    return std::move(c);
  }

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

  linear_combination operator+(const_vector_view c1, const_vector_view c2) {
    CHECK_FOR_SIZE_MISMATCH(c1.sz, c2.sz);
    linear_combination out{};
    out.sz = c1.sz;
    out.view_terms.push_back({1.0, c1});
    out.view_terms.push_back({1.0, c2});
    return out;
  }

  linear_combination operator+(const_vector_view c1, const_vector_slice c2) {
    CHECK_FOR_SIZE_MISMATCH(c1.sz, c2.sz);
    linear_combination out{};
    out.sz = c1.sz;
    out.view_terms.push_back({1.0, c1});
    out.slice_terms.push_back({1.0, c2});
    return out;
  }

  linear_combination&& operator+(const_vector_view c1, linear_combination && c2) {
    CHECK_FOR_SIZE_MISMATCH(c1.sz, c2.sz);
    c2.view_terms.push_back({1.0, c1});
    return std::move(c2);
  }

  linear_combination operator+(const_vector_slice c1, const_vector_view c2) {
    CHECK_FOR_SIZE_MISMATCH(c1.sz, c2.sz);
    linear_combination out{};
    out.sz = c1.sz;
    out.slice_terms.push_back({1.0, c1});
    out.view_terms.push_back({1.0, c2});
    return out;
  }

  linear_combination operator+(const_vector_slice c1, const_vector_slice c2) {
    CHECK_FOR_SIZE_MISMATCH(c1.sz, c2.sz);
    linear_combination out{};
    out.sz = c1.sz;
    out.slice_terms.push_back({1.0, c1});
    out.slice_terms.push_back({1.0, c2});
    return out;
  }

  linear_combination&& operator+(const_vector_slice c1, linear_combination && c2) {
    CHECK_FOR_SIZE_MISMATCH(c1.sz, c2.sz);
    c2.slice_terms.push_back({1.0, c1});
    return std::move(c2);
  };

  linear_combination&& operator+(linear_combination && c1, const_vector_view c2) {
    CHECK_FOR_SIZE_MISMATCH(c1.sz, c2.sz);
    c1.view_terms.push_back({1.0, c2});
    return std::move(c1);
  }

  linear_combination&& operator+(linear_combination && c1, const_vector_slice c2) {
    CHECK_FOR_SIZE_MISMATCH(c1.sz, c2.sz);
    c1.slice_terms.push_back({1.0, c2});
    return std::move(c1);
  }

  linear_combination&& operator+(linear_combination && c1, linear_combination && c2) {
    CHECK_FOR_SIZE_MISMATCH(c1.sz, c1.sz);
    linear_combination c{};
    for (auto & [a, v] : c2.view_terms) { c1.view_terms.push_back({a, v}); };
    for (auto & [a, v] : c2.slice_terms) { c1.slice_terms.push_back({a, v}); };
    return std::move(c1);
  }

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

  linear_combination operator-(const_vector_view c1, const_vector_view c2) {
    CHECK_FOR_SIZE_MISMATCH(c1.sz, c2.sz);
    linear_combination out{};
    out.sz = c1.sz;
    out.view_terms.push_back({ 1.0, c1});
    out.view_terms.push_back({-1.0, c2});
    return out;
  }

  linear_combination operator-(const_vector_view c1, const_vector_slice c2) {
    CHECK_FOR_SIZE_MISMATCH(c1.sz, c2.sz);
    linear_combination out{};
    out.sz = c1.sz;
    out.view_terms.push_back({ 1.0, c1});
    out.slice_terms.push_back({-1.0, c2});
    return out;
  }

  linear_combination&& operator-(const_vector_view c1, linear_combination && c2) {
    CHECK_FOR_SIZE_MISMATCH(c1.sz, c2.sz);
    for (auto & [a, v] : c2.view_terms) { a *= -1; };
    for (auto & [a, v] : c2.slice_terms) { a *= -1; };
    c2.view_terms.push_back({1.0, c1});
    return std::move(c2);
  }

  linear_combination operator-(const_vector_slice c1, const_vector_view c2) {
    CHECK_FOR_SIZE_MISMATCH(c1.sz, c2.sz);
    linear_combination out{};
    out.sz = c1.sz;
    out.slice_terms.push_back({ 1.0, c1});
    out.view_terms.push_back({-1.0, c2});
    return out;
  }

  linear_combination operator-(const_vector_slice c1, const_vector_slice c2) {
    CHECK_FOR_SIZE_MISMATCH(c1.sz, c2.sz);
    linear_combination out{};
    out.sz = c1.sz;
    out.slice_terms.push_back({ 1.0, c1});
    out.slice_terms.push_back({-1.0, c2});
    return out;
  }

  linear_combination&& operator-(const_vector_slice c1, linear_combination && c2) {
    CHECK_FOR_SIZE_MISMATCH(c1.sz, c2.sz);
    for (auto & [a, v] : c2.view_terms) { a *= -1; };
    for (auto & [a, v] : c2.slice_terms) { a *= -1; };
    c2.slice_terms.push_back({1.0, c1});
    return std::move(c2);
  };

  linear_combination&& operator-(linear_combination && c1, const_vector_view c2) {
    CHECK_FOR_SIZE_MISMATCH(c1.sz, c2.sz);
    c1.view_terms.push_back({-1.0, c2});
    return std::move(c1);
  }

  linear_combination&& operator-(linear_combination && c1, const_vector_slice c2) {
    CHECK_FOR_SIZE_MISMATCH(c1.sz, c2.sz);
    c1.slice_terms.push_back({-1.0, c2});
    return std::move(c1);
  }

  linear_combination&& operator-(linear_combination && c1, linear_combination && c2) {
    CHECK_FOR_SIZE_MISMATCH(c1.sz, c1.sz);
    linear_combination c{};
    for (auto & [a, v] : c2.view_terms) { c1.view_terms.push_back({-a, v}); };
    for (auto & [a, v] : c2.slice_terms) { c1.slice_terms.push_back({-a, v}); };
    return std::move(c1);
  }

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

  linear_combination operator*(const_vector_view v, double scale) {
    linear_combination out{};
    out.sz = v.sz;
    out.view_terms.push_back({scale, v});
    return out;
  }

  linear_combination operator*(double scale, const_vector_view v) {
    linear_combination out{};
    out.sz = v.sz;
    out.view_terms.push_back({scale, v});
    return out;
  }

  linear_combination operator*(const_vector_slice v, double scale) {
    linear_combination out{};
    out.sz = v.sz;
    out.slice_terms.push_back({scale, v});
    return out;
  }

  linear_combination operator*(double scale, const_vector_slice v) {
    linear_combination out{};
    out.sz = v.sz;
    out.slice_terms.push_back({scale, v});
    return out;
  }

  linear_combination&& operator*(linear_combination&& v, double scale) {
    for (auto & [a, _] : v.view_terms) { a *= scale; }
    for (auto & [a, _] : v.slice_terms) { a *= scale; }
    return std::move(v);
  }

  linear_combination&& operator*(double scale, linear_combination&& v) {
    for (auto & [a, _] : v.view_terms) { a *= scale; }
    for (auto & [a, _] : v.slice_terms) { a *= scale; }
    return std::move(v);
  }

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

  linear_combination operator/(const_vector_view v, double scale) {
    linear_combination out{};
    out.sz = v.sz;
    out.view_terms.push_back({1.0 / scale, v});
    return out;
  }

  linear_combination operator/(const_vector_slice v, double scale) {
    linear_combination out{};
    out.sz = v.sz;
    out.slice_terms.push_back({1.0 / scale, v});
    return out;
  }

  linear_combination&& operator/(linear_combination&& v, double scale) {
    double inv_scale = 1.0 / scale;
    for (auto & [a, _] : v.view_terms) { a *= inv_scale; }
    for (auto & [a, _] : v.slice_terms) { a *= inv_scale; }
    return std::move(v);
  }

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

  double norm(linear_combination && expr) {
    double sq_sum = {};
    for (int i = 0; i < expr.sz; i++) {
      double component = 0.0;
      for (auto & [c, v] : expr.view_terms) {
        component += c * v[i];
      }

      for (auto & [c, v] : expr.slice_terms) {
        component += c * v.ptr[v.ids[i]];
      }
      sq_sum += component * component;
    }
    return sqrt(sq_sum);
  }

////////////////////////////////////////////////////////////////////////////////

  double total(linear_combination expr) {
    double sum;
    for (int i = 0; i < expr.sz; i++) {
      double component = 0.0;
      for (auto & [c, v] : expr.view_terms) {
        component += c * v[i];
      }

      for (auto & [c, v] : expr.slice_terms) {
        component += c * v.ptr[v.ids[i]];
      }
      sum += component;
    }
    return sum;
  }

////////////////////////////////////////////////////////////////////////////////

#if 0
  linear_combination   foo(const_vector_view c1, const_vector_view c2);
  linear_combination   foo(const_vector_view c1, const_vector_slice c2);
  linear_combination&& foo(const_vector_view c1, linear_combination && c2);

  linear_combination   foo(const_vector_slice c1, const_vector_view c2);
  linear_combination   foo(const_vector_slice c1, const_vector_slice c2);
  linear_combination&& foo(const_vector_slice c1, linear_combination && c2);

  linear_combination&& foo(linear_combination && c1, const_vector_view c2);
  linear_combination&& foo(linear_combination && c1, const_vector_slice c2);
  linear_combination&& foo(linear_combination && c1, linear_combination && c2);
#endif

}