#include <cuda_runtime.h>
#include <cudss.h>
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define CUDSS_TEST_NVTX_ENABLED 1
#else
#define CUDSS_TEST_NVTX_ENABLED 0
#endif

// Standalone build example:
//   nvcc -std=c++17 cuDSS_test.cu -o cuDSS_test -lcudss

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Triplet {
  int row;
  int col;
  double value;
};

struct CsrMatrix {
  int64_t nrows = 0;
  int64_t ncols = 0;
  int64_t nnz = 0;
  std::vector<int> row_offsets;
  std::vector<int> columns;
  std::vector<double> values;
};

const char * cudss_status_name(cudssStatus_t status) {
  switch (status) {
    case CUDSS_STATUS_SUCCESS: return "CUDSS_STATUS_SUCCESS";
    case CUDSS_STATUS_NOT_INITIALIZED: return "CUDSS_STATUS_NOT_INITIALIZED";
    case CUDSS_STATUS_ALLOC_FAILED: return "CUDSS_STATUS_ALLOC_FAILED";
    case CUDSS_STATUS_INVALID_VALUE: return "CUDSS_STATUS_INVALID_VALUE";
    case CUDSS_STATUS_NOT_SUPPORTED: return "CUDSS_STATUS_NOT_SUPPORTED";
    case CUDSS_STATUS_EXECUTION_FAILED: return "CUDSS_STATUS_EXECUTION_FAILED";
    case CUDSS_STATUS_INTERNAL_ERROR: return "CUDSS_STATUS_INTERNAL_ERROR";
  }
  return "CUDSS_STATUS_UNKNOWN";
}

void check_cuda(cudaError_t status, const char * expr, const char * file, int line) {
  if (status != cudaSuccess) {
    std::ostringstream out;
    out << file << ":" << line << ": CUDA call failed: " << expr
        << " returned " << cudaGetErrorString(status);
    throw std::runtime_error(out.str());
  }
}

void check_cudss(cudssStatus_t status, const char * expr, const char * file, int line) {
  if (status != CUDSS_STATUS_SUCCESS) {
    std::ostringstream out;
    out << file << ":" << line << ": cuDSS call failed: " << expr
        << " returned " << cudss_status_name(status) << " (" << static_cast<int>(status) << ")";
    throw std::runtime_error(out.str());
  }
}

#define CHECK_CUDA(expr) check_cuda((expr), #expr, __FILE__, __LINE__)
#define CHECK_CUDSS(expr) check_cudss((expr), #expr, __FILE__, __LINE__)

#if defined(CUDSS_VERSION) && CUDSS_VERSION >= 800
#define CUDSS_TEST_HAS_080_API 1
#else
#define CUDSS_TEST_HAS_080_API 0
#endif

struct NvtxRange {
  explicit NvtxRange(const char * name) {
#if CUDSS_TEST_NVTX_ENABLED
    nvtxRangePushA(name);
#else
    (void)name;
#endif
  }

  ~NvtxRange() {
#if CUDSS_TEST_NVTX_ENABLED
    nvtxRangePop();
#endif
  }

  NvtxRange(const NvtxRange &) = delete;
  NvtxRange & operator=(const NvtxRange &) = delete;
};

std::string lowercase(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

bool is_comment_or_blank(const std::string & line) {
  for (char c : line) {
    if (std::isspace(static_cast<unsigned char>(c))) {
      continue;
    }
    return c == '%';
  }
  return true;
}

void add_entry(std::vector<Triplet> & triplets, int64_t row, int64_t col, double value) {
  if (row < 1 || col < 1) {
    throw std::runtime_error("Matrix Market indices must be one-based positive integers");
  }
  if (row > std::numeric_limits<int>::max() || col > std::numeric_limits<int>::max()) {
    throw std::runtime_error("matrix indices exceed 32-bit cuDSS index limits used by this test");
  }
  triplets.push_back({static_cast<int>(row - 1), static_cast<int>(col - 1), value});
}

CsrMatrix triplets_to_csr(int64_t nrows, int64_t ncols, std::vector<Triplet> triplets) {
  if (nrows <= 0 || ncols <= 0) {
    throw std::runtime_error("matrix dimensions must be positive");
  }
  if (nrows > std::numeric_limits<int>::max() || ncols > std::numeric_limits<int>::max()) {
    throw std::runtime_error("matrix dimensions exceed 32-bit cuDSS index limits used by this test");
  }

  std::sort(triplets.begin(), triplets.end(), [](const Triplet & a, const Triplet & b) {
    if (a.row != b.row) return a.row < b.row;
    return a.col < b.col;
  });

  CsrMatrix A;
  A.nrows = nrows;
  A.ncols = ncols;
  A.row_offsets.assign(static_cast<size_t>(nrows) + 1, 0);

  int current_row = 0;
  for (size_t i = 0; i < triplets.size();) {
    const int row = triplets[i].row;
    const int col = triplets[i].col;
    double value = 0.0;

    if (row < 0 || row >= nrows || col < 0 || col >= ncols) {
      throw std::runtime_error("Matrix Market entry is outside declared dimensions");
    }

    while (i < triplets.size() && triplets[i].row == row && triplets[i].col == col) {
      value += triplets[i].value;
      i++;
    }

    while (current_row <= row) {
      A.row_offsets[static_cast<size_t>(current_row)] = static_cast<int>(A.columns.size());
      current_row++;
    }

    if (value != 0.0) {
      A.columns.push_back(col);
      A.values.push_back(value);
    }
  }

  while (current_row <= nrows) {
    A.row_offsets[static_cast<size_t>(current_row)] = static_cast<int>(A.columns.size());
    current_row++;
  }

  A.nnz = static_cast<int64_t>(A.values.size());
  if (A.nnz > std::numeric_limits<int>::max()) {
    throw std::runtime_error("matrix nnz exceeds 32-bit cuDSS index limits used by this test");
  }
  return A;
}

CsrMatrix read_matrix_market(const std::string & filename) {
  std::ifstream input(filename);
  if (!input) {
    throw std::runtime_error("could not open Matrix Market file: " + filename);
  }

  std::string banner;
  std::string object;
  std::string format;
  std::string field;
  std::string symmetry;
  input >> banner >> object >> format >> field >> symmetry;

  if (banner != "%%MatrixMarket") {
    throw std::runtime_error("invalid Matrix Market banner");
  }

  object = lowercase(object);
  format = lowercase(format);
  field = lowercase(field);
  symmetry = lowercase(symmetry);

  if (object != "matrix") {
    throw std::runtime_error("only Matrix Market matrix objects are supported");
  }
  if (format != "coordinate") {
    throw std::runtime_error("only Matrix Market coordinate format is supported");
  }
  if (field != "real" && field != "integer" && field != "pattern") {
    throw std::runtime_error("only real, integer, and pattern Matrix Market fields are supported");
  }
  if (symmetry != "general" && symmetry != "symmetric" && symmetry != "hermitian" && symmetry != "skew-symmetric") {
    throw std::runtime_error("unsupported Matrix Market symmetry: " + symmetry);
  }

  std::string line;
  std::getline(input, line);
  do {
    if (!std::getline(input, line)) {
      throw std::runtime_error("missing Matrix Market size line");
    }
  } while (is_comment_or_blank(line));

  int64_t nrows = 0;
  int64_t ncols = 0;
  int64_t entries = 0;
  {
    std::istringstream size_line(line);
    if (!(size_line >> nrows >> ncols >> entries)) {
      throw std::runtime_error("invalid Matrix Market size line");
    }
  }

  if (nrows != ncols) {
    throw std::runtime_error("cuDSS solve test expects a square matrix");
  }

  std::vector<Triplet> triplets;
  triplets.reserve(static_cast<size_t>(entries) * (symmetry == "general" ? 1 : 2));

  for (int64_t entry = 0; entry < entries;) {
    if (!std::getline(input, line)) {
      throw std::runtime_error("Matrix Market file ended before all entries were read");
    }
    if (is_comment_or_blank(line)) {
      continue;
    }

    std::istringstream entry_line(line);
    int64_t row = 0;
    int64_t col = 0;
    double value = 1.0;
    if (!(entry_line >> row >> col)) {
      throw std::runtime_error("invalid Matrix Market entry");
    }
    if (field != "pattern" && !(entry_line >> value)) {
      throw std::runtime_error("missing Matrix Market entry value");
    }

    add_entry(triplets, row, col, value);
    if (row != col) {
      if (symmetry == "symmetric" || symmetry == "hermitian") {
        add_entry(triplets, col, row, value);
      } else if (symmetry == "skew-symmetric") {
        add_entry(triplets, col, row, -value);
      }
    }

    entry++;
  }

  return triplets_to_csr(nrows, ncols, std::move(triplets));
}

double time_phase(cudaStream_t stream, const char * label, const std::function<void()> & phase) {
  CHECK_CUDA(cudaStreamSynchronize(stream));
  NvtxRange range(label);
  auto start = std::chrono::steady_clock::now();
  phase();
  CHECK_CUDA(cudaStreamSynchronize(stream));
  auto end = std::chrono::steady_clock::now();
  const double milliseconds = std::chrono::duration<double, std::milli>(end - start).count();
  std::cout << label << ": " << std::fixed << std::setprecision(3) << milliseconds << " ms\n";
  return milliseconds;
}

void create_csr_matrix(cudssMatrix_t * matrix, const CsrMatrix & A, const int * row_offsets, const int * columns, const double * values) {
#if CUDSS_TEST_HAS_080_API
  CHECK_CUDSS(cudssMatrixCreateCsr(
    matrix,
    A.nrows,
    A.ncols,
    A.nnz,
    row_offsets,
    nullptr,
    columns,
    values,
    CUDSS_R_32I,
    CUDSS_R_32I,
    CUDSS_R_64F,
    CUDSS_MTYPE_GENERAL,
    CUDSS_MVIEW_FULL,
    CUDSS_BASE_ZERO));
#else
  CHECK_CUDSS(cudssMatrixCreateCsr(
    matrix,
    A.nrows,
    A.ncols,
    A.nnz,
    const_cast<int *>(row_offsets),
    nullptr,
    const_cast<int *>(columns),
    const_cast<double *>(values),
    CUDA_R_32I,
    CUDA_R_64F,
    CUDSS_MTYPE_GENERAL,
    CUDSS_MVIEW_FULL,
    CUDSS_BASE_ZERO));
#endif
}

void create_dense_matrix(cudssMatrix_t * matrix, int64_t nrows, int64_t ncols, int64_t ld, const double * values) {
#if CUDSS_TEST_HAS_080_API
  CHECK_CUDSS(cudssMatrixCreateDn(
    matrix, nrows, ncols, ld, values, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
#else
  CHECK_CUDSS(cudssMatrixCreateDn(
    matrix, nrows, ncols, ld, const_cast<double *>(values), CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));
#endif
}

double residual_relative_norm(const CsrMatrix & A, const std::vector<double> & x, const std::vector<double> & b) {
  double r2 = 0.0;
  double b2 = 0.0;
  for (int64_t row = 0; row < A.nrows; row++) {
    double Ax = 0.0;
    for (int p = A.row_offsets[static_cast<size_t>(row)]; p < A.row_offsets[static_cast<size_t>(row + 1)]; p++) {
      Ax += A.values[static_cast<size_t>(p)] * x[static_cast<size_t>(A.columns[static_cast<size_t>(p)])];
    }
    const double r = Ax - b[static_cast<size_t>(row)];
    r2 += r * r;
    b2 += b[static_cast<size_t>(row)] * b[static_cast<size_t>(row)];
  }
  return std::sqrt(r2) / std::max(std::sqrt(b2), std::numeric_limits<double>::min());
}

template <typename T>
void cuda_malloc_copy(T ** device_ptr, const std::vector<T> & host_values) {
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(device_ptr), host_values.size() * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(*device_ptr, host_values.data(), host_values.size() * sizeof(T), cudaMemcpyHostToDevice));
}

} // namespace

int main(int argc, char ** argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " matrix.mtx\n";
    return 1;
  }

  int * d_row_offsets = nullptr;
  int * d_columns = nullptr;
  double * d_values = nullptr;
  double * d_x = nullptr;
  double * d_b = nullptr;
  cudaStream_t stream = nullptr;
  cudssHandle_t handle = nullptr;
  cudssConfig_t solver_config = nullptr;
  cudssData_t solver_data = nullptr;
  cudssMatrix_t A_matrix = nullptr;
  cudssMatrix_t x_matrix = nullptr;
  cudssMatrix_t b_matrix = nullptr;

  try {
    CsrMatrix A = read_matrix_market(argv[1]);
    const int64_t n = A.nrows;
    const int nrhs = 1;

    std::vector<double> b(static_cast<size_t>(n), 1.0);
    std::vector<double> x(static_cast<size_t>(n), 0.0);

    std::cout << "Matrix: " << argv[1] << "\n";
    std::cout << "Rows: " << A.nrows << ", cols: " << A.ncols << ", nnz: " << A.nnz << "\n";

    cuda_malloc_copy(&d_row_offsets, A.row_offsets);
    cuda_malloc_copy(&d_columns, A.columns);
    cuda_malloc_copy(&d_values, A.values);
    cuda_malloc_copy(&d_b, b);
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_x), x.size() * sizeof(double)));
    CHECK_CUDA(cudaMemset(d_x, 0, x.size() * sizeof(double)));

    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDSS(cudssCreate(&handle));
    CHECK_CUDSS(cudssSetStream(handle, stream));
    CHECK_CUDSS(cudssConfigCreate(&solver_config));
    CHECK_CUDSS(cudssDataCreate(handle, &solver_data));

    create_csr_matrix(&A_matrix, A, d_row_offsets, d_columns, d_values);
    create_dense_matrix(&x_matrix, n, nrhs, n, d_x);
    create_dense_matrix(&b_matrix, n, nrhs, n, d_b);

    time_phase(stream, "Reordering", [&]() {
      CHECK_CUDSS(cudssExecute(
        handle, CUDSS_PHASE_REORDERING, solver_config, solver_data, A_matrix, x_matrix, b_matrix));
    });

    time_phase(stream, "Symbolic factorization", [&]() {
      CHECK_CUDSS(cudssExecute(
        handle, CUDSS_PHASE_SYMBOLIC_FACTORIZATION, solver_config, solver_data, A_matrix, x_matrix, b_matrix));
    });

    time_phase(stream, "Numerical factorization", [&]() {
      CHECK_CUDSS(cudssExecute(
        handle, CUDSS_PHASE_FACTORIZATION, solver_config, solver_data, A_matrix, x_matrix, b_matrix));
    });

    time_phase(stream, "Solve", [&]() {
      CHECK_CUDSS(cudssExecute(
        handle, CUDSS_PHASE_SOLVE, solver_config, solver_data, A_matrix, x_matrix, b_matrix));
    });

    CHECK_CUDA(cudaMemcpy(x.data(), d_x, x.size() * sizeof(double), cudaMemcpyDeviceToHost));
    std::cout << "Relative residual ||Ax-b||/||b||: "
              << std::scientific << residual_relative_norm(A, x, b) << "\n";

    CHECK_CUDSS(cudssMatrixDestroy(A_matrix));
    A_matrix = nullptr;
    CHECK_CUDSS(cudssMatrixDestroy(x_matrix));
    x_matrix = nullptr;
    CHECK_CUDSS(cudssMatrixDestroy(b_matrix));
    b_matrix = nullptr;
    CHECK_CUDSS(cudssDataDestroy(handle, solver_data));
    solver_data = nullptr;
    CHECK_CUDSS(cudssConfigDestroy(solver_config));
    solver_config = nullptr;
    CHECK_CUDSS(cudssDestroy(handle));
    handle = nullptr;
    CHECK_CUDA(cudaStreamDestroy(stream));
    stream = nullptr;
    CHECK_CUDA(cudaFree(d_row_offsets));
    d_row_offsets = nullptr;
    CHECK_CUDA(cudaFree(d_columns));
    d_columns = nullptr;
    CHECK_CUDA(cudaFree(d_values));
    d_values = nullptr;
    CHECK_CUDA(cudaFree(d_x));
    d_x = nullptr;
    CHECK_CUDA(cudaFree(d_b));
    d_b = nullptr;

    return 0;
  } catch (const std::exception & e) {
    std::cerr << "cuDSS_test failed: " << e.what() << "\n";
  }

  if (A_matrix) cudssMatrixDestroy(A_matrix);
  if (x_matrix) cudssMatrixDestroy(x_matrix);
  if (b_matrix) cudssMatrixDestroy(b_matrix);
  if (solver_data && handle) cudssDataDestroy(handle, solver_data);
  if (solver_config) cudssConfigDestroy(solver_config);
  if (handle) cudssDestroy(handle);
  if (stream) cudaStreamDestroy(stream);
  if (d_row_offsets) cudaFree(d_row_offsets);
  if (d_columns) cudaFree(d_columns);
  if (d_values) cudaFree(d_values);
  if (d_x) cudaFree(d_x);
  if (d_b) cudaFree(d_b);
  return 1;
}
