// CPU sparse matrix / residual integration prototypes (linear-geometry tets),
// comparing the femto library path against FFCx-inspired kernels:
//   cpu:fixed   compile-time sizes, restrict, stack A_e, one pass over qdata,
//               one scatter per element via precomputed CSR positions
//   cpu:affine  the same + the p1 quadrature collapse
//               A_e = detJ * dN_I . (sum_q w_q C_q) . dN_J
// Each prototype is compiled twice: with the build's own flags (-O3, baseline
// x86-64) and again under `#pragma GCC target("avx2,fma")`, to separate the
// refactor from the instruction set.
//
// usage: spmat_cpu_perf [size(0=smoke,1=full)] [warmup] [reps]

#include "fusion_common.hpp"

#include "forall.hpp"

#include "femto/threadpool.hpp"

#include <cstring>
#include <mutex>
#include <vector>

using namespace femto;
using namespace fusion;

static constexpr memory::space cpu_mem = memory::space::cpu;

#define PROTO_NS plain
#include "spmat_cpu_kernels.inl"
#undef PROTO_NS

#pragma GCC push_options
#pragma GCC target("avx2,fma")
#define PROTO_NS avx2
#include "spmat_cpu_kernels.inl"
#undef PROTO_NS
#pragma GCC pop_options

////////////////////////////////////////////////////////////////////////////////

template < Geometry geom, uint32_t NC, uint32_t P, typename JacModel >
void run_case(CaseInfo info, const Mesh<> & mesh, JacModel jac_model, int warmup, int reps) {
  constexpr uint32_t NN = nodes_per_element<geom, P>();
  constexpr uint32_t NQ = qpts_per_element<geom, P>();
  constexpr uint32_t DSZ = NC * 3 * NC * 3;

  Domain<> h_domain(mesh, MeshQuadratureRule(P + 1));
  Field h_u = create_field<Family::H1>(mesh, P, NC);
  fill_solution_field(h_u, mesh);
  BasisFunction<Family::H1> phi(P, NC);

  info.degree = P;
  info.num_elements = h_domain.mesh[geom].shape[0];
  info.num_qpts = total(h_domain.num_qpts);
  info.num_unknowns = h_u.size();
  const uint32_t nelem = uint32_t(info.num_elements);
  const uint32_t nq = uint32_t(info.num_qpts);

  printf("# case: %s %s p%u: %llu elements, %llu qpts, %llu unknowns\n",
         info.physics.c_str(), info.geom.c_str(), P,
         (unsigned long long)info.num_elements, (unsigned long long)info.num_qpts,
         (unsigned long long)info.num_unknowns);

  //////////////////////////////////////////////////////////////////////////////
  // library reference + timings at several thread counts
  //////////////////////////////////////////////////////////////////////////////

  nd::array<double, 3, cpu_mem> du_q = evaluate(grad(h_u), h_domain);
  auto dflux = forall(jac_model, du_q);

  sparse_matrix<> K = blank_sparse_matrix(grad(phi), grad(phi), h_domain);
  auto fill = integrate(dot(grad(phi), dflux, grad(phi)), h_domain);

  for (uint32_t nthreads : {1u, 8u, 32u}) {
    threadpool::set_num_threads(nthreads);
    double t = bench([&]() { fill(K); }, warmup, reps);
    char name[32];
    snprintf(name, sizeof(name), "lib:%ut", nthreads);
    report_time(info, "cpu_spmat", name, "total", t);
  }
  threadpool::set_num_threads(1);
  fill(K);
  nd::array<double, 1, cpu_mem> K_ref = K.values;

  auto verify = [&](const char * label) {
    report_verify(info, "cpu_spmat", label, rel_l2_diff(&K_ref(0), &K.values(0), K.nnz), 1.0e-10);
  };

  //////////////////////////////////////////////////////////////////////////////
  // inputs for the prototype kernels
  //////////////////////////////////////////////////////////////////////////////

  // flat qdata (nq, DSZ), same values the library consumed
  nd::array<double, 2, cpu_mem> qm({nq, DSZ});
  threadpool::set_num_threads(32);
  threadpool::block_parallel_for(nq, [&](uint32_t q0, uint32_t q1) {
    for (uint32_t i = q0; i < q1; i++) {
      grad_t<NC> g;
      memcpy(&g, &du_q(i, 0, 0), sizeof(g));
      jac_t<NC> D = jac_model(g);
      memcpy(&qm(i, 0), &D, sizeof(D));
    }
  });
  threadpool::set_num_threads(1);

  nd::array<double, 3, cpu_mem> G, GX;
  nd::array<double, 1, cpu_mem> W, W_unused;
  build_qtables<geom>(h_domain, P, G, W);
  build_qtables<geom>(h_domain, 1, GX, W_unused);

  // element dof ids and precomputed CSR block positions
  auto conn = h_domain.mesh[geom];
  FiniteElement<geom, Family::H1> el{P};
  FiniteElement<geom, Family::H1> X_el{1};
  nd::array<uint32_t, 2, cpu_mem> ids({nelem, NN});
  nd::array<uint32_t, 2, cpu_mem> X_ids({nelem, 4u});
  for (uint32_t e = 0; e < nelem; e++) {
    el.indices(h_u.offsets, &conn(e, 0), &ids(e, 0));
    X_el.indices(h_domain.mesh.X.offsets, &conn(e, 0), &X_ids(e, 0));
  }

  femto::timer stopwatch;
  stopwatch.start();
  nd::array<uint16_t, 2, cpu_mem> pair_k({nelem, NN * NN});
  {
    const int * rp = &K.row_ptr(0);
    const int * ci = &K.col_ind(0);
    for (uint32_t e = 0; e < nelem; e++) {
      for (uint32_t I = 0; I < NN; I++) {
        uint32_t r = ids(e, I) * NC;
        const int * begin = ci + rp[r];
        const int * end = ci + rp[r + 1];
        for (uint32_t J = 0; J < NN; J++) {
          int c = int(ids(e, J) * NC);
          uint32_t k = uint32_t(std::lower_bound(begin, end, c) - begin) / NC;
          FEMTO_ASSERT(k < 65536, "block index exceeds uint16");
          pair_k(e, I * NN + J) = uint16_t(k);
        }
      }
    }
  }
  stopwatch.stop();
  printf("# position table build: %.3f s\n", stopwatch.elapsed());

  //////////////////////////////////////////////////////////////////////////////
  // prototype stiffness kernels
  //////////////////////////////////////////////////////////////////////////////

  auto zero_values = [&]() { for (size_t i = 0; i < K.nnz; i++) { K.values(i) = 0.0; } };

  auto run_proto = [&](const char * name, auto && range_fn, uint32_t nthreads) {
    std::vector<std::mutex> mutexes(nthreads > 1 ? 1024 : 0);
    std::mutex * mtx = nthreads > 1 ? mutexes.data() : nullptr;
    double t = bench([&]() {
      zero_values();
      if (nthreads > 1) {
        threadpool::set_num_threads(nthreads);
        threadpool::block_parallel_for(nelem, [&](uint32_t e0, uint32_t e1) { range_fn(e0, e1, mtx); });
        threadpool::set_num_threads(1);
      } else {
        range_fn(0, nelem, nullptr);
      }
    }, warmup, reps);
    report_time(info, "cpu_spmat", name, "total", t);
    verify(name);
  };

  auto args = [&](auto fn, uint32_t e0, uint32_t e1, std::mutex * mtx) {
    fn(e0, e1, &K.values(0), &K.row_ptr(0), &pair_k(0, 0), &ids(0, 0), &X_ids(0, 0),
       h_domain.mesh.X.data.data(), &qm(0, 0), &G(0, 0, 0), &GX(0, 0, 0), &W(0), mtx);
  };

  #define SPMAT_PROTO(name, ns, collapse, nthreads) \
    run_proto(name, [&](uint32_t e0, uint32_t e1, std::mutex * mtx) { \
      args(ns::spmat_range<NC, NN, NQ, collapse>, e0, e1, mtx); \
    }, nthreads)

  SPMAT_PROTO("cpu:fixed", plain, false, 1);
  SPMAT_PROTO("cpu:fixed_avx2", avx2, false, 1);
  if constexpr (P == 1) {
    SPMAT_PROTO("cpu:affine", plain, true, 1);
    SPMAT_PROTO("cpu:affine_avx2", avx2, true, 1);
    SPMAT_PROTO("cpu:affine_avx2_8t", avx2, true, 8);
    SPMAT_PROTO("cpu:affine_avx2_32t", avx2, true, 32);
  }
  SPMAT_PROTO("cpu:fixed_avx2_8t", avx2, false, 8);
  SPMAT_PROTO("cpu:fixed_avx2_32t", avx2, false, 32);

  #undef SPMAT_PROTO

  //////////////////////////////////////////////////////////////////////////////
  // residual
  //////////////////////////////////////////////////////////////////////////////

  {
    Residual<Family::H1> r(FunctionSpace{Family::H1, P, NC}, mesh);

    for (uint32_t nthreads : {1u, 32u}) {
      threadpool::set_num_threads(nthreads);
      double t = bench([&]() { r = integrate(dot(du_q, grad(phi)), h_domain); }, warmup, reps);
      char name[32];
      snprintf(name, sizeof(name), "res:lib_%ut", nthreads);
      report_time(info, "cpu_spmat", name, "total", t);
    }
    threadpool::set_num_threads(1);
    r = integrate(dot(du_q, grad(phi)), h_domain);
    nd::array<double, 2, cpu_mem> r_ref = r.data;

    auto run_res = [&](const char * name, auto && range_fn) {
      double t = bench([&]() {
        for (size_t i = 0; i < r.data.size(); i++) { r.data(i / NC, i % NC) = 0.0; }
        range_fn(0, nelem);
      }, warmup, reps);
      report_time(info, "cpu_spmat", name, "total", t);
      report_verify(info, "cpu_spmat", name,
                    rel_l2_diff(&r_ref(0, 0), &r.data(0, 0), r_ref.size()), 1.0e-10);
    };

    #define RES_PROTO(name, ns, collapse) \
      run_res(name, [&](uint32_t e0, uint32_t e1) { \
        ns::residual_range<NC, NN, NQ, collapse>(e0, e1, &r.data(0, 0), &ids(0, 0), &X_ids(0, 0), \
          h_domain.mesh.X.data.data(), &du_q(0, 0, 0), &G(0, 0, 0), &GX(0, 0, 0), &W(0)); \
      })

    RES_PROTO("res:fixed", plain, false);
    RES_PROTO("res:fixed_avx2", avx2, false);
    if constexpr (P == 1) {
      RES_PROTO("res:affine", plain, true);
      RES_PROTO("res:affine_avx2", avx2, true);
    }

    #undef RES_PROTO
  }
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char * argv[]) {
  int size = (argc > 1) ? atoi(argv[1]) : 1;
  int warmup = (argc > 2) ? atoi(argv[2]) : 1;
  int reps = (argc > 3) ? atoi(argv[3]) : 3;

  uint32_t tet_n[2] = {10, 40};
  uint32_t tet2_n[2] = {8, 24};
  int s = (size == 0) ? 0 : 1;

  {
    Mesh<> mesh = tet_cuboid(tet_n[s]);
    run_case<Geometry::Tetrahedron, 1, 1>(CaseInfo{"cpu", "poisson", "tet", "p1"}, mesh,
                                          PoissonJacobianModel{kappa0}, warmup, reps);
    run_case<Geometry::Tetrahedron, 3, 1>(CaseInfo{"cpu", "elasticity", "tet", "p1"}, mesh,
                                          NeoHookeanJacobianModel{lambda0, mu0}, warmup, reps);
  }
  {
    Mesh<> mesh = tet_cuboid(tet2_n[s]);
    run_case<Geometry::Tetrahedron, 1, 2>(CaseInfo{"cpu", "poisson", "tet", "p2"}, mesh,
                                          PoissonJacobianModel{kappa0}, warmup, reps);
    run_case<Geometry::Tetrahedron, 3, 2>(CaseInfo{"cpu", "elasticity", "tet", "p2"}, mesh,
                                          NeoHookeanJacobianModel{lambda0, mu0}, warmup, reps);
  }

  return 0;
}
