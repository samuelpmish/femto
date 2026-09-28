#ifdef FEMTO_ENABLE_PASTIX

// PaStiX backends (https://gitlab.inria.fr/solverstack/pastix): multithreaded
// and, unlike MKL, buildable from source -- webassembly builds included. Note
// that PaStiX pivots statically, so LU/LDLT solves are followed by a few
// steps of iterative refinement to restore direct-solver accuracy.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

#include <pastix.h>
#include <spm.h>

#include <Eigen/OrderingMethods>
#include <Eigen/SparseCholesky>

#include "sparse_factorization_impl.hpp"

namespace femto {

namespace {

// parallel factorization only pays off with enough work per thread: one
// thread per ~1M nonzeros, up to the core count. Wasm builds additionally
// cap the team (sync through Atomics.wait is expensive, and wide teams stop
// paying off) and leave one core free for the browser's own threads.
int num_solver_threads(std::size_t nnz) {
#ifdef FEMTO_SINGLE_THREADED
  (void)nnz;
  return 1;
#else
  std::uint64_t cap = std::thread::hardware_concurrency();
#ifdef __EMSCRIPTEN__
  cap = std::min<std::uint64_t>(8, std::max<std::uint64_t>(1, cap - 1));
#endif
  return int(std::clamp<std::uint64_t>(nnz / 1000000, 1, cap));
#endif
}

struct pastix_impl final : public sparse_factorization::impl {
  SparseFactorizationBackend which;
  pastix_factotype_t facto;
  pastix_data_t * data = nullptr;
  pastix_int_t iparm[IPARM_SIZE];
  double dparm[DPARM_SIZE];
  spmatrix_t spm;

  // the spm arrays are views of these (kept alive for refactorization)
  std::vector<pastix_int_t> colptr;
  std::vector<pastix_int_t> rowind;
  std::vector<double> values;

  // pastixInit happens on the first factorize(), once the matrix is known:
  // the worker team is fixed at init, and its size is chosen from the
  // matrix size (threaded wasm builds rely on the preallocated worker pool
  // being large enough for these solver threads, see cmake/wasm_module.cmake)
  pastix_impl(SparseFactorizationBackend which_, pastix_factotype_t facto_)
    : which(which_), facto(facto_) {}

  ~pastix_impl() override {
    if (data) { pastixFinalize(&data); }
  }

  void factorize(const eigen_sparse_matrix & A, bool analyze) override {
    // the symmetric variants pass just the lower triangle, as spm expects
    bool symmetric = (which != SparseFactorizationBackend::PASTIX_LU);

    if (analyze) {
      int n = int(A.cols());
      colptr.assign(std::size_t(n) + 1, 0);
      rowind.clear();
      values.clear();
      for (int j = 0; j < n; j++) {
        for (eigen_sparse_matrix::InnerIterator it(A, j); it; ++it) {
          if (symmetric && it.row() < j) { continue; }
          rowind.push_back(it.row());
          values.push_back(it.value());
        }
        colptr[std::size_t(j) + 1] = pastix_int_t(rowind.size());
      }

      spmInit(&spm);
      spm.mtxtype = symmetric ? SpmSymmetric : SpmGeneral;
      spm.flttype = SpmDouble;
      spm.fmttype = SpmCSC;
      spm.baseval = 0;
      spm.n = n;
      spm.nnz = pastix_int_t(rowind.size());
      spm.dof = 1;
      spm.colptr = colptr.data();
      spm.rowptr = rowind.data();
      spm.values = values.data();
      spmUpdateComputedFields(&spm);

      pastixInitParam(iparm, dparm);
      iparm[IPARM_VERBOSE] = PastixVerboseNot;
      iparm[IPARM_FACTORIZATION] = facto;
      // refinement is a correction step, not an iterative solver: the default
      // budget (250) is burned in full on every solve of an ill-conditioned
      // system, where the true residual stalls above the target tolerance
      iparm[IPARM_ITERMAX] = 10;
      // note: set explicitly, since PaStiX's own detection reports 1 core
      // when built without hwloc
      iparm[IPARM_THREAD_NBR] = num_solver_threads(std::size_t(A.nonZeros()));
      pastixInit(&data, MPI_COMM_WORLD, iparm, dparm);

      if (pastix_task_analyze(data, &spm) != PASTIX_SUCCESS) {
        std::cout << "error: sparse factorization (analysis) failed" << std::endl;
        exit(1);
      }
    } else {
      // same sparsity pattern: refresh the values in place
      std::size_t k = 0;
      for (int j = 0; j < int(A.cols()); j++) {
        for (eigen_sparse_matrix::InnerIterator it(A, j); it; ++it) {
          if (symmetric && it.row() < j) { continue; }
          values[k++] = it.value();
        }
      }
    }

    if (pastix_task_numfact(data, &spm) != PASTIX_SUCCESS) {
      std::cout << "error: sparse factorization failed" << std::endl;
      exit(1);
    }
  }

  void solve(Eigen::Map<const Eigen::VectorXd> b, Eigen::Map<Eigen::VectorXd> x) const override {
    pastix_int_t n = spm.nexp;
    x = b;  // solved in place
    pastix_task_solve(data, n, 1, x.data(), n);
    // Cholesky of an SPD matrix pivots nothing, so its solves are already
    // backward stable; LU and LDLT pivot statically, and a few refinement
    // steps recover the accuracy that can cost
    if (which != SparseFactorizationBackend::PASTIX_LLT) {
      std::vector<double> rhs(b.data(), b.data() + n);
      pastix_task_refine(data, n, 1, rhs.data(), n, x.data(), n);
    }
  }

  SparseFactorizationBackend backend() const override { return which; }
};

}  // namespace

std::unique_ptr<sparse_factorization::impl> make_pastix_impl(SparseFactorizationBackend which) {
  using enum SparseFactorizationBackend;
  switch (which) {
    case PASTIX_LU:   return std::make_unique<pastix_impl>(PASTIX_LU, PastixFactLU);
    case PASTIX_LLT:  return std::make_unique<pastix_impl>(PASTIX_LLT, PastixFactLLT);
    case PASTIX_LDLT: return std::make_unique<pastix_impl>(PASTIX_LDLT, PastixFactLDLT);
    default:
      return nullptr;  // unreachable, make_impl only forwards PaStiX backends
  }
}

////////////////////////////////////////////////////////////////////////////////
// benchmark (see sparse_direct.hpp)

namespace {

// lower triangle of A as the CSC that spm expects (the matrix stays alive in the vectors)
struct SpmMatrix {
  std::vector<pastix_int_t> colptr, rowind;
  std::vector<double> values;
  spmatrix_t spm;
  SpmMatrix(const eigen_sparse_matrix & A) {
    int n = int(A.cols());
    colptr.assign(std::size_t(n) + 1, 0);
    for (int j = 0; j < n; j++) {
      for (eigen_sparse_matrix::InnerIterator it(A, j); it; ++it) {
        if (it.row() < j) { continue; }
        rowind.push_back(it.row());
        values.push_back(it.value());
      }
      colptr[std::size_t(j) + 1] = pastix_int_t(rowind.size());
    }
    spmInit(&spm);
    spm.mtxtype = SpmSymmetric;
    spm.flttype = SpmDouble;
    spm.fmttype = SpmCSC;
    spm.baseval = 0;
    spm.n = n;
    spm.nnz = pastix_int_t(rowind.size());
    spm.dof = 1;
    spm.colptr = colptr.data();
    spm.rowptr = rowind.data();
    spm.values = values.data();
    spmUpdateComputedFields(&spm);
  }
};

using permutation = Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int>;

// Scotch's nested-dissection ordering of A's graph, computed by PaStiX's
// ordering step and copied out: old node i -> new position perm.indices()[i]
permutation scotch_ordering(const eigen_sparse_matrix & A) {
  int n = int(A.cols());
  permutation perm(n);
  SpmMatrix m(A);
  pastix_int_t iparm[IPARM_SIZE];
  double dparm[DPARM_SIZE];
  pastixInitParam(iparm, dparm);
  iparm[IPARM_VERBOSE] = PastixVerboseNot;
  iparm[IPARM_THREAD_NBR] = 1;
  pastix_data_t * data = nullptr;
  pastixInit(&data, MPI_COMM_WORLD, iparm, dparm);
  pastix_order_t order;
  pastixOrderAlloc(&order, n, 0);
  pastix_subtask_order(data, &m.spm, &order);
  for (int i = 0; i < n; i++) { perm.indices()[i] = int(order.permtab[i]); }
  pastixOrderExit(&order);
  pastixFinalize(&data);
  return perm;
}

permutation ordering_permutation(const eigen_sparse_matrix & A, Ordering ordering) {
  if (ordering == Ordering::NESTED_DISSECTION) { return scotch_ordering(A); }
  permutation perm(int(A.cols()));
  perm.setIdentity();
  if (ordering == Ordering::AMD) {
    Eigen::AMDOrdering<int> amd;
    amd(A, perm);
  }
  return perm;
}

// Eigen's simplicial LDLT on P A P^T with P from Scotch: Eigen's plain
// triangular sweeps beat PaStiX's supernodal ones on the small matrices the
// browser examples solve (see benchmark_* below), and
// nested dissection gives them the least fill
struct eigen_nested_dissection_impl final : public sparse_factorization::impl {
  permutation perm;
  Eigen::SimplicialLDLT<eigen_sparse_matrix, Eigen::Lower, Eigen::NaturalOrdering<int>> solver;

  void factorize(const eigen_sparse_matrix & A, bool analyze) override {
    if (analyze) { perm = scotch_ordering(A); }
    eigen_sparse_matrix Ap = perm * A * perm.transpose();
    if (analyze) { solver.analyzePattern(Ap); }
    solver.factorize(Ap);
    if (solver.info() != Eigen::Success) {
      std::cout << "error: sparse factorization failed" << std::endl;
      exit(1);
    }
  }

  void solve(Eigen::Map<const Eigen::VectorXd> b, Eigen::Map<Eigen::VectorXd> x) const override {
    x = perm.inverse() * solver.solve(perm * b);
  }

  SparseFactorizationBackend backend() const override { return SparseFactorizationBackend::EIGEN_SIMPLICIAL_LDLT_NESTED_DISSECTION; }
};

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

std::unique_ptr<sparse_factorization::impl> make_eigen_nested_dissection_impl() {
  return std::make_unique<eigen_nested_dissection_impl>();
}

SolverTiming benchmark_eigen_llt(const sparse_matrix<> & A, Ordering ordering, int reps) {
  eigen_sparse_matrix eA = to_eigen(A);
  permutation perm = ordering_permutation(eA, ordering);
  eigen_sparse_matrix Ap = perm * eA * perm.transpose();  // P A P^T
  Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(eA.cols(), 0.0, 1.0), x;
  SolverTiming t;
  auto t0 = std::chrono::steady_clock::now();
  Eigen::SimplicialLLT<eigen_sparse_matrix, Eigen::Lower, Eigen::NaturalOrdering<int>> llt(Ap);
  t.factor_ms = ms_since(t0);
  x = llt.solve(b);
  t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; r++) { x = llt.solve(b); }
  t.solve_ms = ms_since(t0) / reps;
  t.nnz_factor = long(llt.matrixL().nestedExpression().nonZeros());
  return t;
}

SolverTiming benchmark_pastix_llt(const sparse_matrix<> & A, Ordering ordering, int threads, int reps) {
  eigen_sparse_matrix eA = to_eigen(A);
  int n = int(eA.cols());
  SpmMatrix m(eA);
  pastix_int_t iparm[IPARM_SIZE];
  double dparm[DPARM_SIZE];
  pastixInitParam(iparm, dparm);
  iparm[IPARM_VERBOSE] = PastixVerboseNot;
  iparm[IPARM_FACTORIZATION] = PastixFactLLT;
  iparm[IPARM_THREAD_NBR] = threads;
  iparm[IPARM_ORDERING] = (ordering == Ordering::NESTED_DISSECTION) ? PastixOrderScotch : PastixOrderPersonal;
  pastix_data_t * data = nullptr;
  pastixInit(&data, MPI_COMM_WORLD, iparm, dparm);

  SolverTiming t;
  auto t0 = std::chrono::steady_clock::now();
  if (ordering == Ordering::NESTED_DISSECTION) {
    pastix_task_analyze(data, &m.spm);
  } else {
    permutation perm = ordering_permutation(eA, ordering);
    pastix_order_t order;
    pastixOrderAlloc(&order, n, 0);
    for (int i = 0; i < n; i++) { order.permtab[i] = perm.indices()[i]; order.peritab[perm.indices()[i]] = i; }
    pastix_subtask_order(data, &m.spm, &order);
    pastix_subtask_symbfact(data);
    pastix_subtask_reordering(data);
    pastix_subtask_blend(data);
    pastixOrderExit(&order);
  }
  pastix_task_numfact(data, &m.spm);
  t.factor_ms = ms_since(t0);
  t.nnz_factor = long(iparm[IPARM_NNZEROS]);

  std::vector<double> b(n, 0.0), x(n, 0.0);
  for (int i = 0; i < n; i++) { b[std::size_t(i)] = double(i) / n; }
  x = b;
  pastix_task_solve(data, n, 1, x.data(), n);
  t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; r++) {
    x = b;
    pastix_task_solve(data, n, 1, x.data(), n);
  }
  t.solve_ms = ms_since(t0) / reps;
  pastixFinalize(&data);
  return t;
}

}  // namespace femto

#endif
