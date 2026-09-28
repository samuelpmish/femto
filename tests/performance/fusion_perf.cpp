// Kernel fusion experiment, CPU driver.
//
// Compares three implementations of the residual / stiffness calculation:
//   separated : the existing library pipeline (evaluate / forall / integrate),
//               which allocates its intermediate arrays on every call
//   staged    : the same three phases as custom loops writing into
//               intermediate buffers preallocated ahead of time, so the
//               timings contain only the computation itself
//   fused     : all three phases in a single loop over elements, with no
//               intermediate staging at all
//
// Cases: {poisson, neohookean} x {tet, hex} x {p=1, p=2} x {medium, large, huge}
//
// Compile-time isolation:
//   -DFUSION_ONLY_SEPARATED : compile only the 3-phase (separated + staged) code
//   -DFUSION_ONLY_FUSED     : compile only the fused kernels

#include "fusion_common.hpp"

#include "femto/threadpool.hpp"
#include "forall.hpp"

#include <atomic>
#include <cstring>
#include <vector>

using namespace femto;
using namespace fusion;

static inline void atomic_add(double & target, double v) {
  std::atomic_ref<double> ref(target);
  ref.fetch_add(v, std::memory_order_relaxed);
}

////////////////////////////////////////////////////////////////////////////////
// element-local helpers shared by the fused and staged loops
////////////////////////////////////////////////////////////////////////////////

// note: the coordinate field X is linear (mesh geometry), so its gather and
// jacobian use the p=1 element and its own shape gradient table G_X, while the
// solution field u uses degree P
template < Geometry geom, uint32_t NC, uint32_t P >
struct ElementScratch {
  static constexpr uint32_t NN = nodes_per_element<geom, P>();
  static constexpr uint32_t NNX = nodes_per_element<geom, 1>();
  uint32_t ids[NN];
  uint32_t X_ids[NNX];
  double X_e[NNX][3];
  double u_e[NN][NC];

  void gather(const FiniteElement< geom, Family::H1 > & el,
              const FiniteElement< geom, Family::H1 > & X_el,
              const Field<Family::H1> & u, const Field<Family::H1> & X,
              nd::view<const Connection, 2> conn, int elem_id,
              bool with_u = true) {
    el.indices(u.offsets, &conn(elem_id, 0), ids);
    X_el.indices(X.offsets, &conn(elem_id, 0), X_ids);
    for (uint32_t i = 0; i < NNX; i++) {
      for (uint32_t d = 0; d < 3; d++) { X_e[i][d] = X.data(X_ids[i], d); }
    }
    if (with_u) {
      for (uint32_t i = 0; i < NN; i++) {
        for (uint32_t c = 0; c < NC; c++) { u_e[i][c] = u.data(ids[i], c); }
      }
    }
  }

  mat3 jacobian(uint32_t q, const nd::array<double, 3, memory::space::cpu> & G_X) const {
    mat3 dX_dxi{};
    for (uint32_t i = 0; i < NNX; i++) {
      vec3 g = {G_X(q, i, 0), G_X(q, i, 1), G_X(q, i, 2)};
      for (uint32_t d = 0; d < 3; d++) {
        dX_dxi[d] += X_e[i][d] * g;
      }
    }
    return dX_dxi;
  }

  grad_t<NC> gradient_xi(uint32_t q, const nd::array<double, 3, memory::space::cpu> & G) const {
    grad_t<NC> du_dxi{};
    for (uint32_t i = 0; i < NN; i++) {
      vec3 g = {G(q, i, 0), G(q, i, 1), G(q, i, 2)};
      if constexpr (NC == 1) {
        du_dxi += u_e[i][0] * g;
      } else {
        for (uint32_t c = 0; c < NC; c++) {
          du_dxi[c] += u_e[i][c] * g;
        }
      }
    }
    return du_dxi;
  }
};

#ifndef FUSION_ONLY_SEPARATED

////////////////////////////////////////////////////////////////////////////////
// fused residual / stiffness (CPU)
////////////////////////////////////////////////////////////////////////////////

template < Geometry geom, uint32_t NC, uint32_t P, typename Material >
void fused_residual_cpu(Residual<Family::H1> & r,
                        const Field<Family::H1> & u,
                        const Domain<> & domain,
                        Material material,
                        const nd::array<double, 3, memory::space::cpu> & G,
                        const nd::array<double, 3, memory::space::cpu> & G_X,
                        const nd::array<double, 1, memory::space::cpu> & W) {

  constexpr uint32_t NN = nodes_per_element<geom, P>();
  constexpr uint32_t NQ = qpts_per_element<geom, P>();
  FiniteElement< geom, Family::H1 > el{P};
  FiniteElement< geom, Family::H1 > X_el{1};
  nd::view<const Connection, 2> conn = domain.mesh[geom];
  nd::view<const int> elements = domain.active_elements[geom];

  zero(r.data);

  threadpool::block_parallel_for(elements.size(), [&](uint32_t e_start, uint32_t e_end) {
    ElementScratch<geom, NC, P> s;
    double r_e[NN][NC];

    for (uint32_t e = e_start; e < e_end; e++) {
      s.gather(el, X_el, u, domain.mesh.X, conn, elements(e));
      std::memset(r_e, 0, sizeof(r_e));

      for (uint32_t q = 0; q < NQ; q++) {
        mat3 dX_dxi = s.jacobian(q, G_X);
        mat3 A = adj(dX_dxi);                    // A = det(dX_dxi) * inv(dX_dxi)
        grad_t<NC> du_dX = dot(s.gradient_xi(q, G), inv(dX_dxi));
        grad_t<NC> flux = material(du_dX);
        double w = W(q);

        for (uint32_t i = 0; i < NN; i++) {
          vec3 g = {G(q, i, 0), G(q, i, 1), G(q, i, 2)};
          vec3 JdN = dot(g, A);
          for (uint32_t c = 0; c < NC; c++) {
            r_e[i][c] += w * flux_component_dot(flux, c, JdN);
          }
        }
      }

      for (uint32_t i = 0; i < NN; i++) {
        for (uint32_t c = 0; c < NC; c++) {
          atomic_add(r.data(s.ids[i], c), r_e[i][c]);
        }
      }
    }
  });
}

template < Geometry geom, uint32_t NC, uint32_t P, typename MaterialJac >
void fused_stiffness_cpu(sparse_matrix<> & K,
                         const Field<Family::H1> & u,
                         const Domain<> & domain,
                         MaterialJac material_jac,
                         const nd::array<double, 3, memory::space::cpu> & G,
                         const nd::array<double, 3, memory::space::cpu> & G_X,
                         const nd::array<double, 1, memory::space::cpu> & W) {

  constexpr uint32_t NN = nodes_per_element<geom, P>();
  constexpr uint32_t NQ = qpts_per_element<geom, P>();
  FiniteElement< geom, Family::H1 > el{P};
  FiniteElement< geom, Family::H1 > X_el{1};
  nd::view<const Connection, 2> conn = domain.mesh[geom];
  nd::view<const int> elements = domain.active_elements[geom];

  zero(K.values);

  threadpool::block_parallel_for(elements.size(), [&](uint32_t e_start, uint32_t e_end) {
    ElementScratch<geom, NC, P> s;
    vec3 dNdX[NN];
    std::vector<double> K_e(NN * NC * NN * NC);

    for (uint32_t e = e_start; e < e_end; e++) {
      s.gather(el, X_el, u, domain.mesh.X, conn, elements(e));
      std::fill(K_e.begin(), K_e.end(), 0.0);

      for (uint32_t q = 0; q < NQ; q++) {
        mat3 dX_dxi = s.jacobian(q, G_X);
        mat3 dxi_dX = inv(dX_dxi);
        double wJ = W(q) * det(dX_dxi);

        for (uint32_t i = 0; i < NN; i++) {
          vec3 g = {G(q, i, 0), G(q, i, 1), G(q, i, 2)};
          dNdX[i] = dot(g, dxi_dX);
        }

        grad_t<NC> du_dX = dot(s.gradient_xi(q, G), dxi_dX);
        jac_t<NC> D = material_jac(du_dX);

        for (uint32_t I = 0; I < NN; I++) {
          for (uint32_t J = 0; J < NN; J++) {
            for (uint32_t i = 0; i < NC; i++) {
              for (uint32_t j = 0; j < NC; j++) {
                K_e[((I * NC + i) * NN * NC) + J * NC + j] += wJ * jac_contract(D, i, j, dNdX[I], dNdX[J]);
              }
            }
          }
        }
      }

      for (uint32_t I = 0; I < NN; I++) {
        for (uint32_t i = 0; i < NC; i++) {
          int row = int(s.ids[I] * NC + i);
          int row_start = K.row_ptr[row];
          int row_end = K.row_ptr[row + 1];
          for (uint32_t J = 0; J < NN; J++) {
            for (uint32_t j = 0; j < NC; j++) {
              int col = int(s.ids[J] * NC + j);
              const int * lo = &K.col_ind[row_start];
              const int * hi = &K.col_ind[row_end];
              int position = int(std::lower_bound(lo, hi, col) - &K.col_ind[0]);
              atomic_add(K.values[position], K_e[((I * NC + i) * NN * NC) + J * NC + j]);
            }
          }
        }
      }
    }
  });
}

#endif // !FUSION_ONLY_SEPARATED

#ifndef FUSION_ONLY_FUSED

////////////////////////////////////////////////////////////////////////////////
// staged (kernel-only) 3-phase variant (CPU): same loops as the fused
// implementation, split into phases with preallocated intermediate buffers
////////////////////////////////////////////////////////////////////////////////

template < Geometry geom, uint32_t NC, uint32_t P >
void staged_interpolate_cpu(nd::array<double, 3, memory::space::cpu> & du_q,
                            const Field<Family::H1> & u,
                            const Domain<> & domain,
                            const nd::array<double, 3, memory::space::cpu> & G,
                            const nd::array<double, 3, memory::space::cpu> & G_X) {

  constexpr uint32_t NQ = qpts_per_element<geom, P>();
  FiniteElement< geom, Family::H1 > el{P};
  FiniteElement< geom, Family::H1 > X_el{1};
  nd::view<const Connection, 2> conn = domain.mesh[geom];
  nd::view<const int> elements = domain.active_elements[geom];

  threadpool::block_parallel_for(elements.size(), [&](uint32_t e_start, uint32_t e_end) {
    ElementScratch<geom, NC, P> s;
    for (uint32_t e = e_start; e < e_end; e++) {
      s.gather(el, X_el, u, domain.mesh.X, conn, elements(e));
      for (uint32_t q = 0; q < NQ; q++) {
        grad_t<NC> du_dX = dot(s.gradient_xi(q, G), inv(s.jacobian(q, G_X)));
        for (uint32_t c = 0; c < NC; c++) {
          for (uint32_t d = 0; d < 3; d++) {
            if constexpr (NC == 1) {
              du_q(e * NQ + q, c, d) = du_dX[d];
            } else {
              du_q(e * NQ + q, c, d) = du_dX[c][d];
            }
          }
        }
      }
    }
  });
}

template < typename Model, typename in_t, typename out_t >
void staged_material_cpu(uint32_t n, Model model, out_t * output, const in_t * input) {
  threadpool::parallel_for(n, [&](int i) {
    output[i] = model(input[i]);
  });
}

template < Geometry geom, uint32_t NC, uint32_t P >
void staged_integrate_residual_cpu(Residual<Family::H1> & r,
                                   const nd::array<double, 3, memory::space::cpu> & flux_q,
                                   const Field<Family::H1> & u,
                                   const Domain<> & domain,
                                   const nd::array<double, 3, memory::space::cpu> & G,
                                   const nd::array<double, 3, memory::space::cpu> & G_X,
                                   const nd::array<double, 1, memory::space::cpu> & W) {

  constexpr uint32_t NN = nodes_per_element<geom, P>();
  constexpr uint32_t NQ = qpts_per_element<geom, P>();
  FiniteElement< geom, Family::H1 > el{P};
  FiniteElement< geom, Family::H1 > X_el{1};
  nd::view<const Connection, 2> conn = domain.mesh[geom];
  nd::view<const int> elements = domain.active_elements[geom];

  zero(r.data);

  threadpool::block_parallel_for(elements.size(), [&](uint32_t e_start, uint32_t e_end) {
    ElementScratch<geom, NC, P> s;
    double r_e[NN][NC];

    for (uint32_t e = e_start; e < e_end; e++) {
      s.gather(el, X_el, u, domain.mesh.X, conn, elements(e), /*with_u=*/false);
      std::memset(r_e, 0, sizeof(r_e));

      for (uint32_t q = 0; q < NQ; q++) {
        mat3 A = adj(s.jacobian(q, G_X));
        double w = W(q);

        grad_t<NC> flux;
        for (uint32_t c = 0; c < NC; c++) {
          vec3 f = {flux_q(e * NQ + q, c, 0), flux_q(e * NQ + q, c, 1), flux_q(e * NQ + q, c, 2)};
          if constexpr (NC == 1) { flux = f; } else { flux[c] = f; }
        }

        for (uint32_t i = 0; i < NN; i++) {
          vec3 g = {G(q, i, 0), G(q, i, 1), G(q, i, 2)};
          vec3 JdN = dot(g, A);
          for (uint32_t c = 0; c < NC; c++) {
            r_e[i][c] += w * flux_component_dot(flux, c, JdN);
          }
        }
      }

      for (uint32_t i = 0; i < NN; i++) {
        for (uint32_t c = 0; c < NC; c++) {
          atomic_add(r.data(s.ids[i], c), r_e[i][c]);
        }
      }
    }
  });
}

template < Geometry geom, uint32_t NC, uint32_t P >
void staged_integrate_stiffness_cpu(sparse_matrix<> & K,
                                    const nd::array<double, 2, memory::space::cpu> & dflux_q,
                                    const Field<Family::H1> & u,
                                    const Domain<> & domain,
                                    const nd::array<double, 3, memory::space::cpu> & G,
                                    const nd::array<double, 3, memory::space::cpu> & G_X,
                                    const nd::array<double, 1, memory::space::cpu> & W) {

  constexpr uint32_t NN = nodes_per_element<geom, P>();
  constexpr uint32_t NQ = qpts_per_element<geom, P>();
  FiniteElement< geom, Family::H1 > el{P};
  FiniteElement< geom, Family::H1 > X_el{1};
  nd::view<const Connection, 2> conn = domain.mesh[geom];
  nd::view<const int> elements = domain.active_elements[geom];

  zero(K.values);

  threadpool::block_parallel_for(elements.size(), [&](uint32_t e_start, uint32_t e_end) {
    ElementScratch<geom, NC, P> s;
    vec3 dNdX[NN];
    std::vector<double> K_e(NN * NC * NN * NC);

    for (uint32_t e = e_start; e < e_end; e++) {
      s.gather(el, X_el, u, domain.mesh.X, conn, elements(e), /*with_u=*/false);
      std::fill(K_e.begin(), K_e.end(), 0.0);

      for (uint32_t q = 0; q < NQ; q++) {
        mat3 dX_dxi = s.jacobian(q, G_X);
        mat3 dxi_dX = inv(dX_dxi);
        double wJ = W(q) * det(dX_dxi);

        for (uint32_t i = 0; i < NN; i++) {
          vec3 g = {G(q, i, 0), G(q, i, 1), G(q, i, 2)};
          dNdX[i] = dot(g, dxi_dX);
        }

        const jac_t<NC> & D = *reinterpret_cast<const jac_t<NC> *>(&dflux_q(e * NQ + q, 0));

        for (uint32_t I = 0; I < NN; I++) {
          for (uint32_t J = 0; J < NN; J++) {
            for (uint32_t i = 0; i < NC; i++) {
              for (uint32_t j = 0; j < NC; j++) {
                K_e[((I * NC + i) * NN * NC) + J * NC + j] += wJ * jac_contract(D, i, j, dNdX[I], dNdX[J]);
              }
            }
          }
        }
      }

      for (uint32_t I = 0; I < NN; I++) {
        for (uint32_t i = 0; i < NC; i++) {
          int row = int(s.ids[I] * NC + i);
          int row_start = K.row_ptr[row];
          int row_end = K.row_ptr[row + 1];
          for (uint32_t J = 0; J < NN; J++) {
            for (uint32_t j = 0; j < NC; j++) {
              int col = int(s.ids[J] * NC + j);
              const int * lo = &K.col_ind[row_start];
              const int * hi = &K.col_ind[row_end];
              int position = int(std::lower_bound(lo, hi, col) - &K.col_ind[0]);
              atomic_add(K.values[position], K_e[((I * NC + i) * NN * NC) + J * NC + j]);
            }
          }
        }
      }
    }
  });
}

#endif // !FUSION_ONLY_FUSED

////////////////////////////////////////////////////////////////////////////////
// case driver
////////////////////////////////////////////////////////////////////////////////

struct PhaseTimes {
  double interpolate = 0.0, material = 0.0, integrate = 0.0;
  double total() const { return interpolate + material + integrate; }
};

static constexpr uint64_t cpu_memory_budget = 40ull * 1000 * 1000 * 1000;

template < Geometry geom, uint32_t NC, uint32_t P, typename Flux, typename Jac >
void run_case_cpu(CaseInfo info, const Mesh<> & mesh, Flux flux_model, Jac jac_model,
                  int warmup, int reps) {

  Domain<> domain(mesh, MeshQuadratureRule(P + 1));
  Field u = create_field<Family::H1>(mesh, P, NC);
  fill_solution_field(u, mesh);
  BasisFunction<Family::H1> phi(P, NC);

  info.degree = P;
  info.num_elements = domain.mesh[geom].shape[0];
  info.num_qpts = total(domain.num_qpts);
  info.num_unknowns = u.size();

  const uint64_t nq = info.num_qpts;
  const uint64_t nelem = info.num_elements;
  constexpr uint32_t NN = nodes_per_element<geom, P>();

  FEMTO_ASSERT(mesh.X.degree == 1, "experiment assumes linear mesh geometry");
  nd::array<double, 3, memory::space::cpu> G, G_X;
  nd::array<double, 1, memory::space::cpu> W, W_unused;
  build_qtables<geom>(domain, P, G, W);
  build_qtables<geom>(domain, 1, G_X, W_unused);

  printf("# case: %s %s %s %s p%u: %llu elements, %llu qpts, %llu unknowns\n",
         info.device.c_str(), info.physics.c_str(), info.geom.c_str(), info.size.c_str(), P,
         (unsigned long long)nelem, (unsigned long long)nq, (unsigned long long)info.num_unknowns);

  //////////////////////////////////////////////////////////////////////////////
  // residual
  //////////////////////////////////////////////////////////////////////////////

  Residual<Family::H1> r_sep;
  Residual<Family::H1> r_staged(FunctionSpace{Family::H1, P, NC}, mesh);
  Residual<Family::H1> r_fused(FunctionSpace{Family::H1, P, NC}, mesh);

#ifndef FUSION_ONLY_FUSED
  {
    PhaseTimes best;
    double best_total = 1.0e30;
    for (int rep = 0; rep < warmup + reps; rep++) {
      PhaseTimes pt;
      timer t;

      t.start();
      nd::array<double, 3, memory::space::cpu> du_q = evaluate(grad(u), domain);
      t.stop();
      pt.interpolate = t.elapsed();

      t.start();
      auto flux_q = forall(flux_model, du_q);
      t.stop();
      pt.material = t.elapsed();

      t.start();
      r_sep = integrate(dot(flux_q, grad(phi)), domain);
      t.stop();
      pt.integrate = t.elapsed();

      if (rep >= warmup && pt.total() < best_total) { best_total = pt.total(); best = pt; }
    }
    report_time(info, "residual", "separated", "interpolate", best.interpolate);
    report_time(info, "residual", "separated", "material", best.material);
    report_time(info, "residual", "separated", "integrate", best.integrate);
    report_time(info, "residual", "separated", "total", best.total());

    report_mem(info, "residual", "separated", "du_dX_q", nq * NC * 3 * 8);
    report_mem(info, "residual", "separated", "flux_q", nq * NC * 3 * 8);
    report_mem(info, "residual", "separated", "element_residuals", nelem * NN * NC * 8);
  }

  // staged: preallocated intermediates, computation-only timings
  {
    nd::array<double, 3, memory::space::cpu> du_q({uint32_t(nq), NC, 3u});
    nd::array<double, 3, memory::space::cpu> flux_q({uint32_t(nq), NC, 3u});

    PhaseTimes best;
    double best_total = 1.0e30;
    for (int rep = 0; rep < warmup + reps; rep++) {
      PhaseTimes pt;
      timer t;

      t.start();
      staged_interpolate_cpu<geom, NC, P>(du_q, u, domain, G, G_X);
      t.stop();
      pt.interpolate = t.elapsed();

      t.start();
      staged_material_cpu(uint32_t(nq), flux_model,
                          reinterpret_cast<grad_t<NC> *>(flux_q.data()),
                          reinterpret_cast<const grad_t<NC> *>(du_q.data()));
      t.stop();
      pt.material = t.elapsed();

      t.start();
      staged_integrate_residual_cpu<geom, NC, P>(r_staged, flux_q, u, domain, G, G_X, W);
      t.stop();
      pt.integrate = t.elapsed();

      if (rep >= warmup && pt.total() < best_total) { best_total = pt.total(); best = pt; }
    }
    report_time(info, "residual", "staged", "interpolate", best.interpolate);
    report_time(info, "residual", "staged", "material", best.material);
    report_time(info, "residual", "staged", "integrate", best.integrate);
    report_time(info, "residual", "staged", "total", best.total());
  }
#endif

#ifndef FUSION_ONLY_SEPARATED
  {
    double t = bench([&]() {
      fused_residual_cpu<geom, NC, P>(r_fused, u, domain, flux_model, G, G_X, W);
    }, warmup, reps);
    report_time(info, "residual", "fused", "total", t);
    report_mem(info, "residual", "fused", "qtables", G.size() * 8 + W.size() * 8);
  }
#endif

#if !defined(FUSION_ONLY_FUSED) && !defined(FUSION_ONLY_SEPARATED)
  report_verify(info, "residual", "separated_vs_staged",
                rel_l2_diff(&r_sep.data[0], &r_staged.data[0], r_sep.size()), 1.0e-12);
  report_verify(info, "residual", "separated_vs_fused",
                rel_l2_diff(&r_sep.data[0], &r_fused.data[0], r_sep.size()), 1.0e-12);
#endif

  //////////////////////////////////////////////////////////////////////////////
  // stiffness (one shared CSR matrix; approaches verified via host copies)
  //////////////////////////////////////////////////////////////////////////////

  uint64_t est_nnz = info.num_unknowns * couplings_per_node(geom, P) * NC;
  uint64_t est_bytes = est_nnz * (12 + 8)                 // K + reference copy
                     + 2 * nq * NC * 3 * NC * 3 * 8       // dflux (staged + library)
                     + nq * NC * 3 * 8;                   // du
  if (est_bytes > cpu_memory_budget) {
    report_skip(info, "stiffness", est_bytes);
    return;
  }

  sparse_matrix<> K = blank_sparse_matrix(grad(phi), grad(phi), domain);

  report_mem(info, "stiffness", "both", "csr_matrix", K.nnz * 12 + (K.nrows + 1) * 4);

#ifndef FUSION_ONLY_FUSED
  {
    PhaseTimes best;
    double best_total = 1.0e30;
    for (int rep = 0; rep < warmup + reps; rep++) {
      PhaseTimes pt;
      timer t;

      t.start();
      nd::array<double, 3, memory::space::cpu> du_q = evaluate(grad(u), domain);
      t.stop();
      pt.interpolate = t.elapsed();

      t.start();
      auto dflux_q = forall(jac_model, du_q);
      t.stop();
      pt.material = t.elapsed();

      t.start();
      auto fill = integrate(dot(grad(phi), dflux_q, grad(phi)), domain);
      fill(K);
      t.stop();
      pt.integrate = t.elapsed();

      if (rep >= warmup && pt.total() < best_total) { best_total = pt.total(); best = pt; }
    }
    report_time(info, "stiffness", "separated", "interpolate", best.interpolate);
    report_time(info, "stiffness", "separated", "material", best.material);
    report_time(info, "stiffness", "separated", "integrate", best.integrate);
    report_time(info, "stiffness", "separated", "total", best.total());

    report_mem(info, "stiffness", "separated", "du_dX_q", nq * NC * 3 * 8);
    report_mem(info, "stiffness", "separated", "dflux_q", nq * NC * 3 * NC * 3 * 8);
  }
#endif

#if !defined(FUSION_ONLY_FUSED) && !defined(FUSION_ONLY_SEPARATED)
  std::vector<double> K_ref(&K.values[0], &K.values[0] + K.nnz);
#endif

#ifndef FUSION_ONLY_FUSED
  // staged: preallocated intermediates, computation-only timings
  {
    nd::array<double, 3, memory::space::cpu> du_q({uint32_t(nq), NC, 3u});
    nd::array<double, 2, memory::space::cpu> dflux_q({uint32_t(nq), NC * 3 * NC * 3});

    PhaseTimes best;
    double best_total = 1.0e30;
    for (int rep = 0; rep < warmup + reps; rep++) {
      PhaseTimes pt;
      timer t;

      t.start();
      staged_interpolate_cpu<geom, NC, P>(du_q, u, domain, G, G_X);
      t.stop();
      pt.interpolate = t.elapsed();

      t.start();
      staged_material_cpu(uint32_t(nq), jac_model,
                          reinterpret_cast<jac_t<NC> *>(dflux_q.data()),
                          reinterpret_cast<const grad_t<NC> *>(du_q.data()));
      t.stop();
      pt.material = t.elapsed();

      t.start();
      staged_integrate_stiffness_cpu<geom, NC, P>(K, dflux_q, u, domain, G, G_X, W);
      t.stop();
      pt.integrate = t.elapsed();

      if (rep >= warmup && pt.total() < best_total) { best_total = pt.total(); best = pt; }
    }
    report_time(info, "stiffness", "staged", "interpolate", best.interpolate);
    report_time(info, "stiffness", "staged", "material", best.material);
    report_time(info, "stiffness", "staged", "integrate", best.integrate);
    report_time(info, "stiffness", "staged", "total", best.total());
  }
#endif

#if !defined(FUSION_ONLY_FUSED) && !defined(FUSION_ONLY_SEPARATED)
  report_verify(info, "stiffness", "separated_vs_staged",
                rel_l2_diff(K_ref.data(), &K.values[0], K.nnz), 1.0e-12);
#endif

#ifndef FUSION_ONLY_SEPARATED
  {
    double t = bench([&]() {
      fused_stiffness_cpu<geom, NC, P>(K, u, domain, jac_model, G, G_X, W);
    }, warmup, reps);
    report_time(info, "stiffness", "fused", "total", t);
    report_mem(info, "stiffness", "fused", "qtables", G.size() * 8 + W.size() * 8);
  }
#endif

#if !defined(FUSION_ONLY_FUSED) && !defined(FUSION_ONLY_SEPARATED)
  report_verify(info, "stiffness", "separated_vs_fused",
                rel_l2_diff(K_ref.data(), &K.values[0], K.nnz), 1.0e-12);
#endif
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char * argv[]) {

  int max_size = (argc > 1) ? atoi(argv[1]) : 2;   // 0 = medium only, 2 = all
  int warmup = (argc > 2) ? atoi(argv[2]) : 1;
  int reps = (argc > 3) ? atoi(argv[3]) : 3;

  const char * size_labels[3] = {"medium", "large", "huge"};
  uint32_t hex_n[3] = {32, 64, 96};
  uint32_t tet_n[3] = {20, 40, 64};

  for (int s = 0; s <= max_size; s++) {
    // hexahedra
    {
      Mesh<> mesh = Mesh<>::cuboid({hex_n[s], hex_n[s], hex_n[s]}, vec3{1.0, 1.0, 1.0});
      CaseInfo poisson{"cpu", "poisson", "hex", size_labels[s]};
      CaseInfo elasticity{"cpu", "elasticity", "hex", size_labels[s]};
      run_case_cpu<Geometry::Hexahedron, 1, 1>(poisson, mesh, PoissonModel{kappa0}, PoissonJacobianModel{kappa0}, warmup, reps);
      run_case_cpu<Geometry::Hexahedron, 1, 2>(poisson, mesh, PoissonModel{kappa0}, PoissonJacobianModel{kappa0}, warmup, reps);
      run_case_cpu<Geometry::Hexahedron, 3, 1>(elasticity, mesh, NeoHookeanModel{lambda0, mu0}, NeoHookeanJacobianModel{lambda0, mu0}, warmup, reps);
      run_case_cpu<Geometry::Hexahedron, 3, 2>(elasticity, mesh, NeoHookeanModel{lambda0, mu0}, NeoHookeanJacobianModel{lambda0, mu0}, warmup, reps);
    }

    // tetrahedra
    {
      Mesh<> mesh = tet_cuboid(tet_n[s]);
      CaseInfo poisson{"cpu", "poisson", "tet", size_labels[s]};
      CaseInfo elasticity{"cpu", "elasticity", "tet", size_labels[s]};
      run_case_cpu<Geometry::Tetrahedron, 1, 1>(poisson, mesh, PoissonModel{kappa0}, PoissonJacobianModel{kappa0}, warmup, reps);
      run_case_cpu<Geometry::Tetrahedron, 1, 2>(poisson, mesh, PoissonModel{kappa0}, PoissonJacobianModel{kappa0}, warmup, reps);
      run_case_cpu<Geometry::Tetrahedron, 3, 1>(elasticity, mesh, NeoHookeanModel{lambda0, mu0}, NeoHookeanJacobianModel{lambda0, mu0}, warmup, reps);
      run_case_cpu<Geometry::Tetrahedron, 3, 2>(elasticity, mesh, NeoHookeanModel{lambda0, mu0}, NeoHookeanJacobianModel{lambda0, mu0}, warmup, reps);
    }
  }

  return 0;
}
