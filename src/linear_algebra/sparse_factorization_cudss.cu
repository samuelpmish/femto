// NVIDIA cuDSS backend: factorizes and solves on the GPU. This is the
// backend for device-memory matrices (inv(sparse_matrix<gpu>)), and host
// matrices can use it too, in which case the matrix is copied to the device
// once and each solve moves its right-hand side / solution across.
//
// cuDSS reads the matrix through a descriptor pointing at CSR arrays that
// must stay valid for the lifetime of the factorization (the solve phase
// can still touch them, e.g. for iterative refinement), so the impl keeps
// its own device copy of the matrix.

#include <cudss.h>
#include <cuda_runtime.h>

#include "sparse_factorization_impl.hpp"

namespace femto {

namespace {

void check(cudssStatus_t status, const char * what) {
  if (status != CUDSS_STATUS_SUCCESS) {
    std::cout << "error: " << what << " failed with cuDSS status " << int(status) << std::endl;
    exit(1);
  }
}

void check(cudaError_t status, const char * what) {
  if (status != cudaSuccess) {
    std::cout << "error: " << what << " failed: " << cudaGetErrorString(status) << std::endl;
    exit(1);
  }
}

struct cudss_impl final : public sparse_factorization::impl {

  SparseFactorizationBackend which;
  cudssHandle_t handle = nullptr;
  cudssConfig_t config = nullptr;
  cudssData_t data = nullptr;
  cudssMatrix_t A_desc = nullptr;
  sparse_matrix<memory::space::gpu> A;  // the arrays A_desc points into
  uint32_t n = 0;

  explicit cudss_impl(SparseFactorizationBackend which_) : which(which_) {
    check(cudssCreate(&handle), "cudssCreate");
    check(cudssConfigCreate(&config), "cudssConfigCreate");
    check(cudssDataCreate(handle, &data), "cudssDataCreate");
  }

  ~cudss_impl() override {
    if (A_desc) { cudssMatrixDestroy(A_desc); }
    if (data) { cudssDataDestroy(handle, data); }
    if (config) { cudssConfigDestroy(config); }
    if (handle) { cudssDestroy(handle); }
  }

  cudssMatrixType_t matrix_type() const {
    using enum SparseFactorizationBackend;
    if (which == CUDSS_LLT) { return CUDSS_MTYPE_SPD; }
    if (which == CUDSS_LDLT) { return CUDSS_MTYPE_SYMMETRIC; }
    return CUDSS_MTYPE_GENERAL;
  }

  // (re)point the descriptor at the current device copy of the matrix
  void describe() {
    if (A_desc) { cudssMatrixDestroy(A_desc); A_desc = nullptr; }
    check(cudssMatrixCreateCsr(&A_desc, int64_t(A.nrows), int64_t(A.ncols), int64_t(A.nnz),
                               A.row_ptr.data(), nullptr, A.col_ind.data(), A.values.data(),
                               CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F,
                               matrix_type(), CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO),
          "cudssMatrixCreateCsr");
  }

  void run(cudssPhase_t phase, cudssMatrix_t x, cudssMatrix_t b, const char * what) {
    check(cudssExecute(handle, phase, config, data, A_desc, x, b), what);
    check(cudaDeviceSynchronize(), what);
  }

  void factorize_gpu(const sparse_matrix<memory::space::gpu> & A_in, bool analyze) override {
    A = A_in;
    n = uint32_t(A.nrows);
    describe();
    // the analysis and factorization phases take no right-hand side, but the
    // API wants descriptors: pass placeholders of the right shape
    cudssMatrix_t x = nullptr, b = nullptr;
    check(cudssMatrixCreateDn(&x, int64_t(n), 1, int64_t(n), nullptr, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR), "cudssMatrixCreateDn");
    check(cudssMatrixCreateDn(&b, int64_t(n), 1, int64_t(n), nullptr, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR), "cudssMatrixCreateDn");
    if (analyze) { run(CUDSS_PHASE_ANALYSIS, x, b, "cuDSS analysis"); }
    run(CUDSS_PHASE_FACTORIZATION, x, b, "cuDSS factorization");
    cudssMatrixDestroy(x);
    cudssMatrixDestroy(b);
  }

  void solve_gpu(const double * b_ptr, double * x_ptr, uint32_t size) const override {
    if (size != n) { backend_error("sparse factorization solve: right-hand side has the wrong size"); }
    cudssMatrix_t x = nullptr, b = nullptr;
    check(cudssMatrixCreateDn(&x, int64_t(n), 1, int64_t(n), x_ptr, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR), "cudssMatrixCreateDn");
    check(cudssMatrixCreateDn(&b, int64_t(n), 1, int64_t(n), const_cast<double *>(b_ptr), CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR), "cudssMatrixCreateDn");
    check(cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, A_desc, x, b), "cuDSS solve");
    check(cudaDeviceSynchronize(), "cuDSS solve");
    cudssMatrixDestroy(x);
    cudssMatrixDestroy(b);
  }

  // host matrices / vectors go through the device
  void factorize(const sparse_matrix<> & A_host, bool analyze) override {
    factorize_gpu(sparse_matrix<memory::space::gpu>(A_host), analyze);
  }

  void factorize(const eigen_sparse_matrix &, bool) override {
    backend_error("cuDSS backend: internal error, Eigen matrices are not accepted");
  }

  void solve(Eigen::Map<const Eigen::VectorXd> b, Eigen::Map<Eigen::VectorXd> x) const override {
    gpu_vector b_gpu(uint32_t(b.size())), x_gpu(uint32_t(x.size()));
    memory::memcpy<double, memory::space::gpu, memory::space::cpu>(b_gpu.ptr, b.data(), uint32_t(b.size()));
    solve_gpu(b_gpu.ptr, x_gpu.ptr, uint32_t(b.size()));
    memory::memcpy<double, memory::space::cpu, memory::space::gpu>(x.data(), x_gpu.ptr, uint32_t(x.size()));
  }

  SparseFactorizationBackend backend() const override { return which; }

};

}  // namespace

std::unique_ptr<sparse_factorization::impl> make_cudss_impl(SparseFactorizationBackend which) {
  return std::make_unique<cudss_impl>(which);
}

////////////////////////////////////////////////////////////////////////////////
// device-memory entry points

SparseFactorizationBackend select_backend(const sparse_matrix<memory::space::gpu> & A) {
  using enum SparseFactorizationBackend;
  bool symmetric = (A.symmetry != Symmetry::Unsymmetric);
  bool spd = symmetric && (A.definiteness == Definiteness::PositiveDefinite);
  if (spd) { return CUDSS_LLT; }
  if (symmetric) { return CUDSS_LDLT; }
  return CUDSS_LU;
}

sparse_factorization inv(const sparse_matrix<memory::space::gpu> & A, SparseFactorizationBackend backend) {
  if (A.nrows != A.ncols) { backend_error("requesting factorization of rectangular matrix"); }
  sparse_factorization invA;
  invA.pimpl = make_cudss_impl(backend);
  invA.pimpl->factorize_gpu(A, true);
  return invA;
}

sparse_factorization inv(const sparse_matrix<memory::space::gpu> & A) {
  return inv(A, select_backend(A));
}

void sparse_factorization::update(const sparse_matrix<memory::space::gpu> & A) {
  pimpl->factorize_gpu(A, false);
}

gpu_vector dot(const sparse_factorization & invA, const gpu_vector & b) {
  gpu_vector x(b.size());
  invA.pimpl->solve_gpu(b.ptr, x.ptr, b.size());
  return x;
}

}  // namespace femto
