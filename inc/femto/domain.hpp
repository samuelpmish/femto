#pragma once

#include <map>

#include "femto/mesh.hpp"
#include "femto/field.hpp"

#include "linear_algebra/sparse_matrix.hpp"

namespace femto {

// DO NOT USE
void set_elems_per_block(uint32_t);

template < memory::space mem_space = memory::space::cpu >
struct Domain {

  // precompute_jacobians == false skips the dxi_dX tables, for internal
  // temporary domains that only exist to build gather tables
  Domain(const Mesh<mem_space> & m, MeshQuadratureRule qrule, bool precompute_jacobians = true)
      : mesh(m), geometry_dimension(m.geometry_dimension), active_elements{}, rule(qrule) {
    foreach_geometry([&](auto geom){
      uint32_t num_elements = mesh[geom].shape[0];
      if (geometry_dimension == dimension(geom) && num_elements > 0) {
        nd::array<int, 1, memory::space::cpu> arr({num_elements});
        for (int i = 0; i < num_elements; i++) {
          arr(i) = i;
        }
        active_elements[geom] = arr;
      }
    });

    num_qpts = geometry_counts() * qpts_per_geom(qrule, geometry_dimension);

    if (precompute_jacobians) { compute_jacobian_inverses(); }
  }

  // a domain over a subset of the mesh's entities, e.g. its boundary facets:
  //
  //   Domain bdr(boundary_of(mesh), MeshQuadratureRule(q));
  //
  // Fields still live on the parent mesh; only the elements iterated over
  // change, so evaluate()/integrate() output is laid out over the facets'
  // quadrature points.  Facets are embedded in a higher-dimensional space, so
  // dX_dxi is not square there: values, isoparametric derivatives, and the
  // integration of H1/DG values (against the facet measure) are supported;
  // spatial derivatives and vector-valued (Hcurl) fields on facets are not.
  Domain(const SubMesh<mem_space> & sub, MeshQuadratureRule qrule, bool precompute_jacobians = true)
      : mesh(*sub.parent), geometry_dimension(sub.geometry_dimension), active_elements{}, rule(qrule) {
    foreach_geometry([&](auto geom){
      uint32_t num_elements = sub[geom].shape[0];
      if (geometry_dimension == dimension(geom) && num_elements > 0) {
        // SubMesh holds uint32_t ids, active_elements int: convert on the host
        nd::array<uint32_t, 1, memory::space::cpu> ids = sub[geom];
        nd::array<int, 1, memory::space::cpu> arr({num_elements});
        for (int i = 0; i < num_elements; i++) {
          arr(i) = ids(i);
        }
        active_elements[geom] = arr;
      }
    });

    num_qpts = geometry_counts() * qpts_per_geom(qrule, geometry_dimension);

    if (precompute_jacobians) { compute_jacobian_inverses(); }
  }

  GeometryInfo geometry_counts() const {
    return GeometryInfo{
      active_elements.vert.shape[0], 
      active_elements.edge.shape[0], 
      active_elements.tri.shape[0], 
      active_elements.quad.shape[0], 
      active_elements.tet.shape[0], 
      active_elements.hex.shape[0]
    };
  }

  void spatial_sort();

  const Mesh<mem_space> & mesh;

  // the dimension of the elements integrated over: the mesh's, or one less
  // for a boundary domain.  Dof numbering always follows mesh.geometry_dimension
  uint32_t geometry_dimension;

  // DG dofs live on the top-dimensional cells only, so a DG field has no dofs
  // of its own on a boundary domain's facets (its trace there belongs to the
  // neighbouring cell, which none of the kernels look up yet)
  void check_family(Family f) const {
    FEMTO_ASSERT(f != Family::DG || geometry_dimension == mesh.geometry_dimension,
                 "DG fields are not supported on boundary domains");
  }

  GeometryData< nd::array< int, 1, mem_space > > active_elements;

  MeshQuadratureRule rule;
  GeometryInfo num_qpts;

  // dxi_dX(q, i, j) = d(xi_i)/d(X_j): the inverse of the isoparametric jacobian
  // at every quadrature point, laid out in the same geometry-by-geometry order
  // as evaluate() output, so integration kernels can load it rather than gather
  // the coordinate field and invert per quadrature point.  det_dX_dxi(q) is the
  // determinant of the (uninverted) jacobian, so the weighted piola
  // transformations are multiplications rather than determinant evaluations.
  // On a boundary domain the jacobian J is (sdim x gdim) and has no inverse:
  // there dxi_dX(q) is its pseudo-inverse (JᵀJ)⁻¹Jᵀ, shaped (gdim, sdim), and
  // det_dX_dxi(q) the facet measure sqrt(det(JᵀJ))
  nd::array<double, 3, mem_space> dxi_dX;
  nd::array<double, 1, mem_space> det_dX_dxi;
  void compute_jacobian_inverses();

  struct AssemblyLUT {
    nd::array<uint32_t, 1, mem_space> ids;
    nd::array<uint32_t, 1, mem_space> offsets;
  };

  const AssemblyLUT & get(Geometry g, Family f, uint32_t p) const;

  mutable std::map< std::tuple< Geometry, Family, uint32_t >, AssemblyLUT > gather_tables;

};

enum class DomainType { ISOPARAMETRIC, SPATIAL };

template < memory::space mem_space = memory::space::cpu >
struct DomainWithType {
  const Domain<mem_space> & domain;
  const DomainType type;

  DomainWithType(const Domain<mem_space> & d) : domain(d), type(DomainType::SPATIAL) {};
  DomainWithType(DomainType t, const Domain<mem_space> & d) : domain(d), type(t) {};
};

template < memory::space mem_space >
inline DomainWithType<mem_space> isoparametric(const Domain<mem_space> & domain) {
  return DomainWithType<mem_space>(DomainType::ISOPARAMETRIC, domain);
}

// how the sparse matrix assembly kernels evaluate *test* basis functions.
// Tabulating them trades arithmetic for memory traffic; on the GPU the data can
// additionally live in shared memory or stay in const global memory and rely on
// the L1 cache to amortize reuse within a block.
enum class TestShapeMode : uint32_t {
  MinimalShared  = 0,  // minimal tables (1D factors for tensor-product elements), staged in shared memory
  MinimalGlobal  = 1,  // the same tables, read from const global memory
  OnTheFlyShared = 2,  // only quadrature points staged, shape functions evaluated on demand
  OnTheFlyGlobal = 3,  // quadrature points in const global memory, evaluated on demand
};

TestShapeMode get_test_shape_mode();
void set_test_shape_mode(TestShapeMode mode);

namespace impl {

template < Family family, DerivedQuantity op >
void evaluate(nd::array<double, 3, memory::space::cpu> & output, const Field<family> & u, const Domain<> & domain, const DomainType & type);

#ifdef NDARRAY_ENABLE_CUDA
template < Family family, DerivedQuantity op >
void evaluate_cuda_impl(nd::array<double, 3, memory::space::gpu> & output,
                        const Field<family, memory::space::gpu> & u,
                        const Domain<memory::space::gpu> & domain,
                        const DomainType & type);
#endif

} // namespace impl

// evaluate() returns a writer rather than an array: `u_q = evaluate(...)` fills
// u_q's existing storage, so a repeated evaluation (a Newton loop, a time step)
// stops allocating and discarding a quadrature-point array per call.  Assign to
// an nd::array to run it; the writer holds the FieldOp and DomainWithType handles
// by value, so it is only valid while the field and domain they name are alive.
// The family and derived quantity are part of the FieldOp type, so they reach
// the kernels as template parameters with no runtime dispatch.
template < DerivedQuantity dq, Family f >
nd::array<double, 3, memory::space::cpu>::writer evaluate(const FieldOp<dq, f> input, const DomainWithType<> & d) {
  using writer = nd::array<double, 3, memory::space::cpu>::writer;
  return writer([input, d](nd::array<double, 3, memory::space::cpu> & u_q) {
    uint32_t gdim = d.domain.geometry_dimension;

    // no-op when u_q is already the right size; the kernels write every
    // quadrature point of every geometry, so there is nothing to zero
    u_q.resize({total(d.domain.num_qpts), input.field.data.shape[1], qshape(f, dq, gdim)});

    impl::evaluate<f, dq>(u_q, input.field, d.domain, d.type);
  });
}

template < Family f >
nd::array<double, 3, memory::space::cpu>::writer evaluate(const Field<f> & u, const DomainWithType<> & d) {
  return evaluate(FieldOp<DerivedQuantity::VALUE, f>{u}, d);
}

#ifdef NDARRAY_ENABLE_CUDA
template < DerivedQuantity dq, Family f >
nd::array<double, 3, memory::space::gpu>::writer evaluate(const FieldOp<dq, f, memory::space::gpu> input,
                                                          const DomainWithType<memory::space::gpu> & d) {
  // see the cpu overload above -- reusing the device buffer matters more here,
  // since cudaFree synchronizes the device
  using writer = nd::array<double, 3, memory::space::gpu>::writer;
  return writer([input, d](nd::array<double, 3, memory::space::gpu> & u_q) {
    uint32_t gdim = d.domain.geometry_dimension;
    u_q.resize({total(d.domain.num_qpts), input.field.data.shape[1], qshape(f, dq, gdim)});
    impl::evaluate_cuda_impl<f, dq>(u_q, input.field, d.domain, d.type);
  });
}

template < Family f >
nd::array<double, 3, memory::space::gpu>::writer evaluate(const Field<f, memory::space::gpu> & u,
                                                          const DomainWithType<memory::space::gpu> & d) {
  return evaluate(FieldOp<DerivedQuantity::VALUE, f, memory::space::gpu>{u}, d);
}
#endif

template < typename T >
auto integrate(const T & integrand, const DomainWithType<> & domain);

#ifdef NDARRAY_ENABLE_CUDA
template < typename T >
auto integrate(const T & integrand, const DomainWithType<memory::space::gpu> & domain);
#endif

template < typename T >
auto integrate_vjp(const T & d_dresidual, const Domain<> & domain);

femto::sparse_matrix<> blank_sparse_matrix(FunctionSpace test, FunctionSpace trial, const Domain<> &domain);

#ifdef NDARRAY_ENABLE_CUDA
femto::sparse_matrix<memory::space::gpu> blank_sparse_matrix(
  FunctionSpace test,
  FunctionSpace trial,
  const Domain<memory::space::gpu> & domain);
#endif

template < DerivedQuantity test_dq, Family test_f, DerivedQuantity trial_dq, Family trial_f, memory::space mem_space >
auto blank_sparse_matrix(const BasisFunctionOp<test_dq, test_f> & test,
                         const BasisFunctionOp<trial_dq, trial_f> & trial,
                         const Domain<mem_space> & domain) {
  return blank_sparse_matrix(test.function.space, trial.function.space, domain);
}

} // namespace femto

#include "femto/integrate.hpp"
