#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>

#include "linear_algebra/sparse_direct.hpp"
#include "linear_algebra/sparse_matrix.hpp"
#include "linear_algebra/vector.hpp"

using triplet = femto::triplet;

constexpr double tolerance = 1.0e-14;

// clang-format off
TEST(VectorTests, BasicArithmetic) {
  femto::vector u(10);
  femto::vector v(10);
  femto::vector w(10);

  for (int i = 0; i < 10; i++) {
    u[i] = 1.0;
    v[i] = 2.0;
    w[i] = 3.0;
  }

  EXPECT_LT(norm(u + v - w), tolerance);

  femto::vector output = u + v; 
  EXPECT_LT(norm(output - w), tolerance);
  EXPECT_LT(norm(u - w / 3.0), tolerance);
  EXPECT_LT(norm(3 * u - w), tolerance);
  EXPECT_NEAR(dot(u, w) - 30, 0.0, tolerance);
  EXPECT_NEAR(dot(femto::ones(10), w) - 30, 0.0, tolerance);
  EXPECT_EQ(u.size(), 10);
}

// clang-format off
TEST(VectorTests, slices) {
  femto::vector u(10);
  for (int i = 0; i < 10; i++) {
    u[i] = i;
  }

  femto::vector output = u[{1,2,3}];
  EXPECT_EQ(output.size(), 3);
  EXPECT_EQ(output[0], 1);
  EXPECT_EQ(output[1], 2);
  EXPECT_EQ(output[2], 3);

  // slice-vector assignment
  u[{4,5,6}] = femto::vector({1.0, 7.0, 3.0});
  EXPECT_EQ(u[4], 1);
  EXPECT_EQ(u[5], 7);
  EXPECT_EQ(u[6], 3);

  // slice-slice assignment
  u[{4,5,6}] = u[{7,8,9}];
  EXPECT_EQ(u[4], 7);
  EXPECT_EQ(u[5], 8);
  EXPECT_EQ(u[6], 9);
}

// clang-format off
TEST(VectorTests, LinearCombinations) {
  femto::vector u(10);
  femto::vector v(10);
  for (int i = 0; i < 10; i++) {
    u[i] = 1.0;
    v[i] = i;
  }

  femto::vector w(3);
  for (int i = 0; i < 3; i++) {
    w[i] = i*i;
  }

  femto::vector z1 = u + v;
  EXPECT_NEAR(z1[0], 1.0, tolerance);
  EXPECT_NEAR(z1[1], 2.0, tolerance);
  EXPECT_NEAR(z1[2], 3.0, tolerance);

  femto::vector z2 = u - v;
  EXPECT_NEAR(z2[0],  1.0, tolerance);
  EXPECT_NEAR(z2[1],  0.0, tolerance);
  EXPECT_NEAR(z2[2], -1.0, tolerance);

  femto::vector z3 = u - 2 * v;
  EXPECT_NEAR(z3[0],  1.0, tolerance);
  EXPECT_NEAR(z3[1], -1.0, tolerance);
  EXPECT_NEAR(z3[2], -3.0, tolerance);

  femto::vector z4 = -(u - 2 * v);
  EXPECT_NEAR(z4[0], -1.0, tolerance);
  EXPECT_NEAR(z4[1], +1.0, tolerance);
  EXPECT_NEAR(z4[2], +3.0, tolerance);

  femto::vector z5 = v[{1,2,3}] - w;
  EXPECT_NEAR(z5[0],  1.0, tolerance);
  EXPECT_NEAR(z5[1],  1.0, tolerance);
  EXPECT_NEAR(z5[2], -1.0, tolerance);

  femto::vector z6 = (v[{1,2,3}] - w) + u[{4,5,6}];
  EXPECT_NEAR(z6[0],  2.0, tolerance);
  EXPECT_NEAR(z6[1],  2.0, tolerance);
  EXPECT_NEAR(z6[2],  0.0, tolerance);

  femto::vector z7 = (v[{1,2,3}] - w) + (u[{4,5,6}] / 2);
  EXPECT_NEAR(z7[0],  1.5, tolerance);
  EXPECT_NEAR(z7[1],  1.5, tolerance);
  EXPECT_NEAR(z7[2], -0.5, tolerance);

  femto::vector z8 = v[{1,2,3}] - w + u[{4,5,6}] * 0.5;
  EXPECT_NEAR(z8[0],  1.5, tolerance);
  EXPECT_NEAR(z8[1],  1.5, tolerance);
  EXPECT_NEAR(z8[2], -0.5, tolerance);

  femto::vector z9 = w - v[{1,2,3}];
  EXPECT_NEAR(z9[0], -1.0, tolerance);
  EXPECT_NEAR(z9[1], -1.0, tolerance);
  EXPECT_NEAR(z9[2], +1.0, tolerance);
}

TEST(SparseMatrixTests, TripletConstructionTrivial) {
  std::vector< triplet > entries{{0, 0, 1.0}};
  auto A = femto::sparse_matrix<>::from_triplets(entries, 1, 1); 
  EXPECT_EQ(A.nrows, 1);
  EXPECT_EQ(A.ncols, 1);
  EXPECT_EQ(A.nnz, 1);
}

TEST(SparseMatrixTests, TripletConstructionEmptyRows) {
  std::vector< triplet > entries{{0, 0, 1.0}, {0, 1, 1.0}, {0, 0, 1.0}, {3, 4, 1.0}};
  auto A = femto::sparse_matrix<>::from_triplets(entries, 4, 5); 
  EXPECT_EQ(A.nrows, 4);
  EXPECT_EQ(A.ncols, 5);
  EXPECT_EQ(A.nnz, 3);
}

TEST(SparseMatrixTests, MatvecTrivial) {
  std::vector< triplet > entries{{0, 0, 1.0}};
  auto A = femto::sparse_matrix<>::from_triplets(entries, 1, 1); 
  femto::vector x({2.0});
  femto::vector b = dot(A, x);
  EXPECT_EQ(b[0], 2.0);
}

TEST(SparseMatrixTests, Matvec) {
  std::vector< triplet > entries{{0, 0, 1.0}, {0, 1, 1.0}, {2, 0, 2.0}, {2, 0, 1.0}};
  auto A = femto::sparse_matrix<>::from_triplets(entries, 3, 2); 
  femto::vector x({{1.0, 2.0}});
  femto::vector b = dot(A, x);
  femto::vector expected({{3.0, 0.0, 3.0}});
  EXPECT_LT(norm(b - expected), tolerance);
}

TEST(SparseMatrixTests, ImportMatrixMarket) {
  auto A = femto::sparse_matrix<>::from_matrix_market(FEMTO_DATA_DIR"matrices/arc130.mtx"); 
  EXPECT_EQ(A.nrows, 130);
  EXPECT_EQ(A.ncols, 130);
  EXPECT_EQ(A.nnz, 1282);
  EXPECT_EQ(129, A.col_ind[A.nnz - 1]);
  EXPECT_NEAR(1.025157410651445, A.values[A.nnz - 1], tolerance);
}

TEST(SparseMatrixTests, ExportMatrixMarket) {
  std::vector< triplet > entries{{0, 0, 1.0}, {0, 1, 1.240935092309530250}, {2, 0, 2.0}, {2, 0, 1.0}};
  auto A = femto::sparse_matrix<>::from_triplets(entries); 
  femto::export_matrix_market(A, "test.mtx");

  auto B = femto::sparse_matrix<>::from_matrix_market("test.mtx"); 

  EXPECT_EQ(A.nrows, B.nrows); 
  EXPECT_EQ(A.ncols, B.ncols); 
  EXPECT_EQ(A.nnz, B.nnz); 
  for (size_t i = 0; i < A.nnz; i++) {
    EXPECT_EQ(A.col_ind[i], B.col_ind[i]); 
    EXPECT_NEAR(A.values[i], B.values[i], tolerance); 
  }
  std::remove("test.mtx");
}

TEST(SparseMatrixTests, Inv2x2) {
  std::vector< triplet > entries{
    {0, 0, 2.0}, 
    {1, 0, 1.0}, {1, 1, 2.0}
  };

  auto A = femto::sparse_matrix<>::from_triplets(entries, 2, 2); 
  EXPECT_EQ(A.nrows, 2);
  EXPECT_EQ(A.ncols, 2);
  EXPECT_EQ(A.nnz, 3);

  femto::vector b(2);
  b[0] = 1;
  b[1] = 2;

  femto::vector x = dot(inv(A), b);

  EXPECT_NEAR(x[0], 0.5, tolerance);
  EXPECT_NEAR(x[1], 0.75, tolerance);

}

TEST(SparseMatrixTests, Inv3x3) {
  std::vector< triplet > entries{
    {0, 0, 3.0},
    {1, 0, 1.0}, {1, 1, 2.0},
    {2, 0, 0.5}, {2, 1, 1.0}, {2, 2, 2.0}
  };
  auto A = femto::sparse_matrix<>::from_triplets(entries, 3, 3); 
  EXPECT_EQ(A.nrows, 3);
  EXPECT_EQ(A.ncols, 3);
  EXPECT_EQ(A.nnz, 6);

  auto invA = inv(A);

  femto::vector b(3);
  b[0] = 1;
  b[1] = 2;
  b[2] = 3;

  femto::vector x = dot(invA, b);

  EXPECT_NEAR(x[0], 1.0 / 3.0, tolerance);
  EXPECT_NEAR(x[1], 5.0 / 6.0, tolerance);
  EXPECT_NEAR(x[2], 1.0, tolerance);

}

TEST(SparseMatrixTests, inv_tridiagonal) {
  std::size_t n = 1000;

  std::vector< triplet > entries(3 * n - 2);
  int count = 0;
  for (int i = 0; i < n; i++) {
    if (0 <= i-1) entries[count++] = {i, i-1, -1.0};
    if (  true  ) entries[count++] = {i, i,    2.0};
    if (i+1 <  n) entries[count++] = {i, i+1, -1.0};
  }

  auto A = femto::sparse_matrix<>::from_triplets(entries, n, n); 
  EXPECT_EQ(A.nrows, n);
  EXPECT_EQ(A.ncols, n);
  EXPECT_EQ(A.nnz, 3 * n - 2);

  auto invA = inv(A);

  femto::vector b(n);
  for (int i = 0; i < n; i++) {
    b[i] = i;
  }

  femto::vector x = dot(invA, b);

}

TEST(SparseMatrixTests, set_slice_values) {
  std::size_t n = 5;

  std::vector< triplet > entries(3 * n - 2);
  int count = 0;
  for (int i = 0; i < n; i++) {
    if (0 <= i-1) entries[count++] = {i, i-1, -1.0};
    if (  true  ) entries[count++] = {i, i,    2.0};
    if (i+1 <  n) entries[count++] = {i, i+1, -1.0};
  }

  femto::sparse_matrix A = femto::sparse_matrix<>::from_triplets(entries, n, n); 

  std::vector<int> ids = {2, 3};
  A(ids, ids) = [](int i, int j){ return i == j; };

  EXPECT_EQ(A.values[ 6], 1);
  EXPECT_EQ(A.values[ 7], 0);

  EXPECT_EQ(A.values[ 8], 0);
  EXPECT_EQ(A.values[ 9], 1);
}

TEST(SparseMatrixTests, create_sparse_matrix_from_slice) {
  std::size_t n = 5;

  std::vector< triplet > entries(3 * n - 2);
  int count = 0;
  for (int i = 0; i < n; i++) {
    if (0 <= i-1) entries[count++] = {i, i-1, -1.0};
    if (  true  ) entries[count++] = {i, i,    2.0};
    if (i+1 <  n) entries[count++] = {i, i+1, -1.0};
  }

  femto::sparse_matrix A = femto::sparse_matrix<>::from_triplets(entries, n, n); 

  /*
    [ +2   -1 [         ]    ]
    [ -1   +2 [ -1      ]    ]
    [      -1 [ +2   -1 ]    ]
    [         [ -1   +2 ] -1 ]
    [         [      -1 ] +2 ]
  */ 

  std::vector<int> columns = {2, 3};
  femto::sparse_matrix B = A({}, columns);
  EXPECT_EQ(B.nrows, 5);
  EXPECT_EQ(B.ncols, 2);
  EXPECT_EQ(B.nnz, 6);

  EXPECT_EQ(B.row_ptr[0], 0);

  EXPECT_EQ(B.row_ptr[1], 0);
  EXPECT_EQ(B.col_ind[0], 0); EXPECT_EQ(B.values[0], -1);

  EXPECT_EQ(B.row_ptr[2], 1);
  EXPECT_EQ(B.col_ind[1], 0); EXPECT_EQ(B.values[1], +2);
  EXPECT_EQ(B.col_ind[2], 1); EXPECT_EQ(B.values[2], -1);

  EXPECT_EQ(B.row_ptr[3], 3);
  EXPECT_EQ(B.col_ind[3], 0); EXPECT_EQ(B.values[3], -1);
  EXPECT_EQ(B.col_ind[4], 1); EXPECT_EQ(B.values[4], +2);

  EXPECT_EQ(B.row_ptr[4], 5);
  EXPECT_EQ(B.col_ind[5], 1); EXPECT_EQ(B.values[5], -1);

  EXPECT_EQ(B.row_ptr[5], 6);

}

TEST(SparseMatrixTests, combined_slicing) {

  std::size_t n = 5;

  std::vector< triplet > entries(3 * n - 2);
  int count = 0;
  for (int i = 0; i < n; i++) {
    if (0 <= i-1) entries[count++] = {i, i-1, -1.0};
    if (  true  ) entries[count++] = {i, i,    2.0};
    if (i+1 <  n) entries[count++] = {i, i+1, -1.0};
  }

  femto::sparse_matrix A = femto::sparse_matrix<>::from_triplets(entries, n, n); 

  femto::vector x(n);
  for (int i = 0; i < n; i++) {
    x[i] = i;
  }

  femto::vector y = dot(A({}, {1,2,3}), x[{0, 2, 4}]);

  EXPECT_EQ(y.size(), n);
  EXPECT_EQ(y[0],  0);
  EXPECT_EQ(y[1], -2);
  EXPECT_EQ(y[2],  0);
  EXPECT_EQ(y[3],  6);
  EXPECT_EQ(y[4], -4);

}

#if 0
TEST(SparseMatrixDeathTests, ImportMatrixMarketFileNonexistent) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH({
    auto A = femto::sparse_matrix<>::from_matrix_market("this_file_does_not_exist.mtx");
  }, "not found");
}

TEST(SparseMatrixDeathTests, ImportMatrixMarketFileWrongObject) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  std::ofstream outfile("vector.mtx");
  outfile << "%%MatrixMarket vector coordinate real general" << std::endl;
  outfile.close();
  EXPECT_DEATH({
    auto A = femto::sparse_matrix<>::from_matrix_market("vector.mtx");
  }, "object type, expected \"matrix\"");
  std::remove("vector.mtx");
}

TEST(SparseMatrixDeathTests, ImportMatrixMarketFileWrongFormat) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  std::ofstream outfile("wrong_format.mtx");
  outfile << "%%MatrixMarket matrix array real general" << std::endl;
  outfile.close();
  EXPECT_DEATH({
    auto A = femto::sparse_matrix<>::from_matrix_market("wrong_format.mtx");
  }, "only coordinate type is currently supported");
  std::remove("wrong_format.mtx");
}

TEST(SparseMatrixDeathTests, ImportMatrixMarketFileWrongField) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  std::ofstream outfile("wrong_field.mtx");
  outfile << "%%MatrixMarket matrix coordinate complex general" << std::endl;
  outfile.close();
  EXPECT_DEATH({
    auto A = femto::sparse_matrix<>::from_matrix_market("wrong_field.mtx");
  }, "only real-valued matrices are currently supported");
  std::remove("wrong_field.mtx");
}

TEST(SparseMatrixDeathTests, ImportMatrixMarketFileWrongSymmetry) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  std::ofstream outfile("symmetric_matrix.mtx");
  outfile << "%%MatrixMarket matrix coordinate real symmetric" << std::endl;
  outfile.close();
  EXPECT_DEATH({
    auto A = femto::sparse_matrix<>::from_matrix_market("symmetric_matrix.mtx");
  }, "only general sparse matrices");
  std::remove("symmetric_matrix.mtx");
}

TEST(SparseMatrixDeathTests, ImportMatrixMarketFileBadDimensions) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  std::ofstream outfile("bad_dimensions.mtx");
  outfile << "%%MatrixMarket matrix coordinate real general" << std::endl;
  outfile << "4 0 12" << std::endl;
  outfile.close();
  EXPECT_DEATH({
    auto A = femto::sparse_matrix<>::from_matrix_market("bad_dimensions.mtx");
  }, "invalid matrix dimensions");
  std::remove("bad_dimensions.mtx");
}

TEST(SparseMatrixDeathTests, ImportMatrixMarketFileBadValues) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  std::ofstream outfile("bad_values.mtx");
  outfile << "%%MatrixMarket matrix coordinate real general" << std::endl;
  outfile << "5 5 10" << std::endl;
  outfile << "0 0 3.14" << std::endl;
  outfile.close();
  EXPECT_DEATH({
    auto A = femto::sparse_matrix<>::from_matrix_market("bad_values.mtx");
  }, "invalid row/column value");
  std::remove("bad_values.mtx");
}
#endif
