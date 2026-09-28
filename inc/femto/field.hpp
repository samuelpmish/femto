#pragma once

#include <functional>

#include "containers/ndarray.hpp"
#include "femto/finite_element.hpp"
#include "linear_algebra/vector.hpp"

#include "femto/assert.hpp"

namespace femto {

template < memory::space mem_space = memory::space::cpu >
struct Mesh;

template < Family f, memory::space mem_space = memory::space::cpu >
struct Field {

  static constexpr Family family = f;

  Field() = default;

  template < memory::space other_space >
  Field(const Field<f, other_space> & other) {
    *this = other;
  }

  void operator=(nd::array<double, 2, mem_space> new_data) {
    FEMTO_ASSERT((new_data.shape[0] == data.shape[0]) && (new_data.shape[1] == data.shape[1]),
                 "error: incompatible dimensions for assignment");
    data = new_data;
  }

  template < memory::space other_space >
  void operator=(const Field<f, other_space> & other) {
    degree = other.degree;
    data = other.data;
    offsets = other.offsets;
  }

  uint32_t size() const { return data.shape[0] * data.shape[1]; };
  uint32_t num_nodes() const { return data.shape[0]; };

  femto::Tvector_slice<double, mem_space> v(const std::vector<int> & ids) {
    return {data.data(), ids.data(), uint32_t(ids.size())}; 
  }
  femto::Tvector_slice<const double, mem_space> v(const std::vector<int> & ids) const {
    return {data.data(), ids.data(), uint32_t(ids.size())}; 
  }

  femto::Tvector_view<double, mem_space> v() { return {data.data(), size()}; }
  femto::Tvector_view<const double, mem_space> v() const { return {data.data(), size()}; }

  uint32_t degree;
  nd::array<double, 2, mem_space> data;
  GeometryInfo offsets;

};

template < Family family, memory::space mem_space >
Field<family, mem_space> create_field(const Mesh<mem_space> & mesh, uint32_t degree, uint32_t components = 1) {
  uint32_t gdim = mesh.geometry_dimension;

  GeometryInfo nodes_per = interior_nodes_per_geom(FunctionSpace{family, degree}, gdim);
  GeometryInfo counts = mesh.geometry_counts();

  Field<family, mem_space> output;
  output.degree = degree;
  output.data = nd::array<double, 2, mem_space>({total(nodes_per * counts), components});
  output.offsets = scan(nodes_per * counts);
  return output;
}

template < memory::space target_space, Family family, memory::space source_space >
Field<family, target_space> copy_to(const Field<family, source_space> & input) {
  return Field<family, target_space>(input);
}

template < Family f >
struct BasisFunction {
  static constexpr Family family = f;

  FunctionSpace space;

  template < memory::space mem_space >
  BasisFunction(const Field<f, mem_space> & u) : space(f, u.degree, u.data.shape[1]) {}
  BasisFunction(uint32_t degree, uint32_t components = 1) : space(f, degree, components) {}

  bool operator==(const BasisFunction & other) const {
    return (space.components == other.space.components) && (space.degree == other.space.degree);
  }
};

template < Family f, memory::space mem_space >
BasisFunction(const Field<f, mem_space> &) -> BasisFunction<f>;

// the dual of Field: a functional on a discrete function space, so it carries
// the same compile-time family
template < Family f, memory::space mem_space = memory::space::cpu >
struct Residual {
  static constexpr Family family = f;

  Residual() {};

  template < memory::space mesh_space >
  Residual(FunctionSpace, const Mesh<mesh_space>& mesh);

  // shape this residual for the given space and zero it, keeping the existing
  // allocation when the size already matches
  template < memory::space mesh_space >
  void reset(FunctionSpace, const Mesh<mesh_space>& mesh);

  template < memory::space other_space >
  Residual(const Residual<f, other_space> & other) {
    *this = other;
  }

  Residual& operator=(const Residual & other) = default;

  // see nd::array::writer -- lets integrate(...) refill an existing Residual
  // instead of allocating and discarding one per call
  struct writer {
    std::function< void(Residual &) > fill;
    explicit writer(std::function< void(Residual &) > fn) : fill(std::move(fn)) {}
  };

  Residual(const writer & w) { w.fill(*this); }

  Residual& operator=(const writer & w) { w.fill(*this); return *this; }

  template < memory::space other_space >
  Residual& operator=(const Residual<f, other_space> & other) {
    space = other.space;
    offsets = other.offsets;
    data = other.data;
    return *this;
  }

  uint32_t size() const { return data.shape[0] * data.shape[1]; };
  femto::Tvector_view<double, mem_space> v() { return {data.data(), size()}; }
  femto::Tvector_view<const double, mem_space> v() const { return {data.data(), size()}; }

  FunctionSpace space;
  GeometryInfo offsets;
  nd::array<double, 2, mem_space> data;
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template < Family family >
nd::array<double,2,memory::space::cpu> nodes_for(const Field<family> & u, const Mesh<> & mesh);
template < Family family >
nd::array<double,2,memory::space::cpu> directions_for(const Field<family> & u, const Mesh<> & mesh);

#ifdef NDARRAY_ENABLE_CUDA
template < Family family >
nd::array<double,2,memory::space::gpu> nodes_for(
  const Field<family, memory::space::gpu> & u,
  const Mesh<memory::space::gpu> & mesh);

template < Family family >
nd::array<double,2,memory::space::gpu> directions_for(
  const Field<family, memory::space::gpu> & u,
  const Mesh<memory::space::gpu> & mesh);
#endif

////////////////////////////////////////////////////////////////////////////////

enum class Modifier { NONE, DIAGONAL, SYM };

enum class DerivedQuantity {
  VALUE,
  VALUE_AVERAGE,
  VALUE_JUMP,
  VALUE_TWO_SIDED,
  GRAD,
  GRAD_AVERAGED,
  GRAD_TWO_SIDED,
  CURL,
  DIV,
  STRAIN  // generalized (membrane, bending, shear) strains of a Family::MITC shell field
};

// grad(u) on a Field or BasisFunction produces one of these: the derived
// quantity and the family are part of the type, so evaluate() / integrate()
// hand them to the kernels as template parameters directly
template < DerivedQuantity dq, Family f, memory::space mem_space = memory::space::cpu >
struct FieldOp {
  static constexpr DerivedQuantity op = dq;
  static constexpr Family family = f;
  const Field<f, mem_space> & field;
};

template < DerivedQuantity dq, Family f >
struct BasisFunctionOp {
  static constexpr DerivedQuantity op = dq;
  static constexpr Family family = f;
  BasisFunction<f> function;
  Modifier mod = Modifier::NONE;
};

// a bare Field / BasisFunction means its VALUE
template < Family f, memory::space mem_space >
FieldOp<DerivedQuantity::VALUE, f, mem_space> as_op(const Field<f, mem_space> & u) { return {u}; }
template < DerivedQuantity dq, Family f, memory::space mem_space >
FieldOp<dq, f, mem_space> as_op(const FieldOp<dq, f, mem_space> & du) { return du; }
template < Family f >
BasisFunctionOp<DerivedQuantity::VALUE, f> as_op(const BasisFunction<f> & phi) { return {phi}; }
template < DerivedQuantity dq, Family f >
BasisFunctionOp<dq, f> as_op(const BasisFunctionOp<dq, f> & dphi) { return dphi; }

////////////////////////////////////////////////////////////////////////////////

template < Family f, memory::space mem_space >
FieldOp<DerivedQuantity::GRAD, f, mem_space> grad(const Field<f, mem_space> & u) {
  static_assert(is_scalar_valued(f), "grad(Field) only supports scalar-valued function spaces");
  return {u};
}

template < Family f, memory::space mem_space >
FieldOp<DerivedQuantity::CURL, f, mem_space> curl(const Field<f, mem_space> & u) {
  static_assert(f == Family::Hcurl, "curl(Field) only supports Family::Hcurl");
  return {u};
}

template < Family f, memory::space mem_space >
FieldOp<DerivedQuantity::STRAIN, f, mem_space> strain(const Field<f, mem_space> & u) {
  static_assert(f == Family::MITC, "strain(Field) only supports Family::MITC");
  return {u};
}

template < Family f, memory::space mem_space >
FieldOp<DerivedQuantity::DIV, f, mem_space> div(const Field<f, mem_space> & u) {
  static_assert(f == Family::Hdiv, "div(Field) only supports Family::Hdiv");
  return {u};
}

template < Family f, memory::space mem_space >
FieldOp<DerivedQuantity::VALUE_JUMP, f, mem_space> jump(const Field<f, mem_space> & u) {
  static_assert(f == Family::DG, "jump(Field) only supports Family::DG");
  return {u};
}

template < Family f, memory::space mem_space >
FieldOp<DerivedQuantity::VALUE_AVERAGE, f, mem_space> average(const Field<f, mem_space> & u) {
  static_assert(f == Family::DG, "average(Field) only supports Family::DG");
  return {u};
}

template < Family f >
BasisFunctionOp<DerivedQuantity::GRAD, f> grad(const BasisFunction<f> & phi) {
  static_assert(is_scalar_valued(f), "grad(BasisFunction) only supports scalar-valued function spaces");
  return {phi};
}

template < Family f >
BasisFunctionOp<DerivedQuantity::CURL, f> curl(const BasisFunction<f> & phi) {
  static_assert(f == Family::Hcurl, "curl(BasisFunction) only supports Family::Hcurl");
  return {phi};
}

template < Family f >
BasisFunctionOp<DerivedQuantity::STRAIN, f> strain(const BasisFunction<f> & phi) {
  static_assert(f == Family::MITC, "strain(BasisFunction) only supports Family::MITC");
  return {phi};
}

template < Family f >
BasisFunctionOp<DerivedQuantity::DIV, f> div(const BasisFunction<f> & phi) {
  static_assert(f == Family::Hdiv, "div(BasisFunction) only supports Family::Hdiv");
  return {phi};
}

template < Family f >
BasisFunctionOp<DerivedQuantity::VALUE_JUMP, f> jump(const BasisFunction<f> & phi) {
  static_assert(f == Family::DG, "jump(BasisFunction) only supports Family::DG");
  return {phi};
}

template < Family f >
BasisFunctionOp<DerivedQuantity::VALUE_AVERAGE, f> average(const BasisFunction<f> & phi) {
  static_assert(f == Family::DG, "average(BasisFunction) only supports Family::DG");
  return {phi};
}

////////////////////////////////////////////////////////////////////////////////

// for integrating sparse matrices (e.g. mass, stiffness)
template < typename T1, typename T2, typename T3 >
struct WeightedIntegrand {
  const T1 test;
  const T2 & qdata; 
  const T3 trial;
};

// for integrating residual vectors
template < typename T1, typename T2 >
struct WeightedIntegrand< T1, T2, void > {
  const T1 test; 
  const T2 & qdata;
};

template < typename T >
struct DiagonalOnly : public T {};

template < typename T >
DiagonalOnly(T) -> DiagonalOnly<T>;

template < typename T, uint32_t n, memory::space mem_space, Family f >
auto operator*(const nd::array<T,n,mem_space> & data, const BasisFunction<f> & phi) {
  return WeightedIntegrand< BasisFunctionOp<DerivedQuantity::VALUE, f>, nd::array<T,n,mem_space>, void >{as_op(phi), data};
}

template < typename T, uint32_t n, memory::space mem_space, DerivedQuantity dq, Family f >
auto operator*(const nd::array<T,n,mem_space> & data, const BasisFunctionOp<dq, f> & dphi) {
  return WeightedIntegrand< BasisFunctionOp<dq, f>, nd::array<T,n,mem_space>, void >{dphi, data};
}

template < typename T, uint32_t n, memory::space mem_space, DerivedQuantity dq, Family f >
auto dot(const nd::array<T,n,mem_space> & data, const BasisFunctionOp<dq, f> & dphi) {
  return WeightedIntegrand< BasisFunctionOp<dq, f>, nd::array<T,n,mem_space>, void >{dphi, data};
}

template < typename T, uint32_t n, memory::space mem_space, Family f >
auto dot(const nd::array<T,n,mem_space> & data, const BasisFunction<f> & phi) {
  return WeightedIntegrand< BasisFunctionOp<DerivedQuantity::VALUE, f>, nd::array<T,n,mem_space>, void >{as_op(phi), data};
}

template < typename T, uint32_t n, memory::space mem_space, DerivedQuantity dq, Family f >
auto diagonal(WeightedIntegrand< BasisFunctionOp<dq, f>, nd::array<T,n,mem_space>, BasisFunctionOp<dq, f> > integrand) {
  FEMTO_ASSERT(
    integrand.test.function == integrand.trial.function,
    "diag(...) requires same test and trial space"
  );

  return DiagonalOnly{integrand};
}

////////////////////////////////////////////////////////////////////////////////

template < typename test_t, typename T, uint32_t n, memory::space mem_space, typename trial_t >
auto dot(const test_t & psi, const nd::array<T,n,mem_space> & data, const trial_t & phi) {
  auto test = as_op(phi);
  auto trial = as_op(psi);
  return WeightedIntegrand< decltype(test), nd::array<T,n,mem_space>, decltype(trial) >{test, data, trial};
}

////////////////////////////////////////////////////////////////////////////////

template < Family f >
double dot(const Residual<f> & r, const Field<f> & u) {
  FEMTO_ASSERT(r.data.shape == u.data.shape, "invalid array shapes");

  double sum = 0.0;
  for (uint32_t i = 0; i < r.data.shape[0]; i++) {
    for (uint32_t j = 0; j < r.data.shape[1]; j++) {
      sum += r.data(i,j) * u.data(i,j);
    }
  }
  return sum;
}

template < Family f >
double dot(const Field<f> & u, const Residual<f> & r) { return dot(r, u); }

// +------------+------+-------+------+----+
// |            |  H1  | Hcurl | Hdiv | DG |
// +------------+------+-------+------+----+
// | value (1D) |   1  |   1   |   1  |  1 |
// +------------+------+-------+------+----+
// | deriv (1D) |   1  |   1   |   1  |  1 |
// +------------+------+-------+------+----+
// | value (2D) |   1  |   2   |   2  |  1 |
// +------------+------+-------+------+----+
// | deriv (2D) |   2  |   1   |   1  |  2 |
// +------------+------+-------+------+----+
// | value (3D) |   1  |   3   |   3  |  1 |
// +------------+------+-------+------+----+
// | deriv (3D) |   3  |   3   |   1  |  3 |
// +------------+------+-------+------+----+
__host__ __device__ constexpr uint32_t qshape(Family f, DerivedQuantity op, uint32_t gdim) {
  if (op == DerivedQuantity::VALUE) { return is_vector_valued(f) ? gdim : 1; } 
  if (op == DerivedQuantity::GRAD) { return gdim; }
  if (op == DerivedQuantity::CURL) { return (gdim == 2) ? 1 : gdim; }
  if (op == DerivedQuantity::DIV) { return 1; }
  if (op == DerivedQuantity::STRAIN) { return 8; }

  return (1u<<31);
}

template < typename T, uint32_t n >
stack::array<T, n> remove_ones(const stack::array<T, n> & x) { 
  uint32_t rank = 0;
  stack::array<T, n> copy{};
  for (int i = 0; i < n; i++) {
    if (x[i] > 1) copy[rank++] = x[i];
  }
  return copy;
}

template < typename T, uint32_t m, uint32_t n >
bool compatible_shapes(const stack::array<T, m> & x, 
                       const stack::array<T, n> & y) {
  auto x_filtered = remove_ones(x);
  auto y_filtered = remove_ones(y);
  for (int i = 0; i < std::max(m, n); i++) {
    auto xval = (i >= m) ? 0 : x_filtered[i];
    auto yval = (i >= n) ? 0 : y_filtered[i];
    if (xval != yval) { return false; }
  }
  return true;
}

}
