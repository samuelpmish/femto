#include "linear_algebra/sparse_matrix.hpp"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <algorithm>

#include "femto/assert.hpp"
#include "femto/threadpool.hpp"

#include "misc/timer.hpp"

namespace femto {

template <>
sparse_matrix<> sparse_matrix<>::with_sparsity(sparsity_pattern<> pattern) {
  sparse_matrix<> A;
  static_cast<sparsity_pattern<> &>(A) = pattern;
  A.values.resize(A.nnz);
  return A;
}

template <>
sparse_matrix<> sparse_matrix<>::with_dimensions(const size_t rows, const size_t cols, const size_t nonzeros) {
  sparse_matrix<> A;
  A.nrows = rows;
  A.ncols = cols;
  A.nnz = nonzeros;
  A.row_ptr.resize(rows + 1);
  A.col_ind.resize(nonzeros);
  A.values.resize(nonzeros);
  return A;
}

template <>
sparse_matrix<> sparse_matrix<>::from_triplets(std::vector<triplet>& triplets, const size_t rows, const size_t cols) {

  constexpr auto same_row_and_column = [](const triplet & x, const triplet & y){
    return ((std::get<0>(x) == std::get<0>(y)) && (std::get<1>(x) == std::get<1>(y)));
  };

  sparse_matrix<> A;

  if (triplets.size() > 0) {

    std::sort(triplets.begin(), triplets.end());

    A.nnz = 0;
    A.nrows = (rows) ? rows : std::get<ROW>(triplets.back()) + 1;

    A.row_ptr.resize(A.nrows + 1);
    A.row_ptr[0] = 0;

    int max_col = std::get<COL>(triplets[0]);

    triplet sum = triplets[0];
    for (size_t i = 1; i < triplets.size(); i++) {
      triplet t = triplets[i];
      max_col = std::max(max_col, std::get<COL>(t));
      if (same_row_and_column(sum, t)) {
        std::get<2>(sum) += std::get<2>(t);
      } else {
        triplets[A.nnz++] = sum;
        for (int r = std::get<0>(sum); r < std::get<0>(t); r++) {
          A.row_ptr[r+1] = A.nnz;
        }
        sum = t;
      }
    }
    triplets[A.nnz++] = sum;
    for (int r = std::get<0>(sum); r < A.nrows; r++) {
      A.row_ptr[r+1] = A.nnz;
    }

    A.col_ind.resize(A.nnz);
    A.values.resize(A.nnz);
  
    for (size_t i = 0; i < A.nnz; i++) {
      A.col_ind[i] = std::get<1>(triplets[i]);
      A.values[i] = std::get<2>(triplets[i]);
    }

    A.ncols = (cols) ? cols : max_col + 1;

  }

  return A;
}

template <>
sparse_matrix<> sparse_matrix<>::from_matrix_market(std::string filename) {

  int line_number = 0;

  std::ifstream infile(filename);
  if (!infile) {
    std::cerr << "error: file \"" << filename << "\" not found" << std::endl;
    exit(1);
  }

  std::string preamble, object, format, field, symmetry;
  infile >> preamble;
  if (preamble != "%%MatrixMarket") {
    std::cerr << "error: invalid file format" << std::endl;
    exit(1);
  }

  infile >> object;
  if (object != "matrix") {
    std::cerr << "error: object type, expected \"matrix\", but got " << object << std::endl;
    exit(1);
  }

  infile >> format;
  if (format != "coordinate") {
    std::cerr << "error: only coordinate type is currently supported" << std::endl;
    exit(1);
  }

  infile >> field;
  if (field != "real") {
    std::cerr << "error: only real-valued matrices are currently supported" << std::endl;
    exit(1);
  }

  infile >> symmetry;
  if (symmetry != "general") {
    std::cerr << "error: only general sparse matrices are currently supported" << std::endl;
    exit(1);
  }
  std::string line;
  std::getline(infile, line);
  line_number++;

  size_t nrows = 0;
  size_t ncols = 0;
  size_t nnz = 0;
  std::vector < triplet > triplets;

  while (std::getline(infile, line))
  {
    line_number++;

    if (line[0] == '%') continue;

    std::istringstream iss(line);
    if (nnz == 0) {
      iss >> nrows >> ncols >> nnz;
      if (nrows == 0 || ncols == 0 || nnz == 0) {
        std::cerr << "error: invalid matrix dimensions in matrix market file" << std::endl;
        exit(1);
      }
      triplets.reserve(nnz);
      continue;
    } else {
      int row, col;
      double value;
      iss >> row >> col >> value;
      if (row == 0 || col == 0) {
        std::cerr << "error: invalid row/column value on line " << line_number << std::endl;
        exit(1);
      }
      // offset row/col by 1, since matrix market uses 1-based indexing
      triplets.push_back({row - 1, col - 1, value});
    }

  }

  return sparse_matrix<>::from_triplets(triplets, nrows, ncols);
}

template <>
sparse_matrix<> import_matrix_market<memory::space::cpu>(std::string filename) {
  return sparse_matrix<>::from_matrix_market(filename);
}

template <>
void export_matrix_market<memory::space::cpu>(const sparse_matrix<> & A, std::string filename) {

  std::ofstream outfile(filename);
  
  // for now, we're just assuming all matrices are real, general
  outfile << "%%MatrixMarket matrix coordinate real general" << std::endl;

  outfile << A.nrows << " " << A.ncols << " " << A.nnz << std::endl;

  outfile << std::setprecision(16);

  for (int r = 0; r < A.nrows; r++) {
    for (int i = A.row_ptr[r]; i < A.row_ptr[r + 1]; i++) {
      // offset rows/columns by 1, since matrix market uses 1-based indexing
      outfile << r+1 << " " << A.col_ind[i]+1 << " " << A.values[i] << '\n';
    }
  }

  outfile.close();

}



nd::array<double, 1, memory::space::cpu> diagonal(const sparse_matrix<> & A) {
  FEMTO_ASSERT(A.nrows == A.ncols, "diagonal(...) expects a square matrix");

  nd::array<double, 1, memory::space::cpu> D(stack::array<uint32_t, 1>{uint32_t(A.nrows)});

  for (int i = 0; i < A.nrows; i++) {
    for (int k = A.row_ptr[i]; k < A.row_ptr[i+1]; k++) {
      int j = A.col_ind[k];
      if (i == j) {
        D[i] = A.values[k];
        break;
      }
    }
  }

  return D;
}

template <>
void sparse_matrix_slice<>::operator=(std::function< double(int, int) > func) {

  int nrows = (rows.size() == 0) ? spmat->nrows : rows.size();

  threadpool::parallel_for(nrows, [&](uint32_t i) {
    int row = (rows.size() == 0) ? i : rows[i];

    int j = 0;
    for (int p = spmat->row_ptr[row]; p < spmat->row_ptr[row+1]; p++) {
      int col = spmat->col_ind[p];

      if (cols.size() != 0) {
        while (j < cols.size() && cols[j] < col) { j++; };
        if (j < cols.size() && col == cols[j]) {
          spmat->values[p] = func(row, col);
        };
      } else {
        spmat->values[p] = func(row, col);
      }
    }
  });
}

template <>
void sparse_matrix<>::operator=(const sparse_matrix_slice<> & slice) {

  nrows = (slice.rows.size() == 0) ? slice.spmat->nrows : slice.rows.size();
  ncols = (slice.cols.size() == 0) ? slice.spmat->ncols : slice.cols.size();

  row_ptr.resize(nrows+1); 
  nd::zero(row_ptr);   // the counts below accumulate into it

  // first, count the nonzeros and set up row_ptr
  int slice_nrows = (slice.rows.size() == 0) ? slice.spmat->nrows : slice.rows.size();
  threadpool::parallel_for(slice_nrows, [&](uint32_t i) {
    int row = (slice.rows.size() == 0) ? i : slice.rows[i];

    int j = 0;
    for (int p = slice.spmat->row_ptr[row]; p < slice.spmat->row_ptr[row+1]; p++) {
      int col = slice.spmat->col_ind[p];
      if (slice.cols.size() != 0) {
        while (j < slice.cols.size() && slice.cols[j] < col) { j++; };
        if (j < slice.cols.size() && col == slice.cols[j]) {
          row_ptr[i+1]++;
        };
      } else {
        row_ptr[i+1]++;
      }
    }
  });

  // then, scan the entries of row_ptr
  row_ptr[0] = 0;
  for (int row = 1; row <= nrows; row++) {
    row_ptr[row] += row_ptr[row-1];
  }
  nnz = row_ptr[nrows];

  col_ind.resize(nnz);
  values.resize(nnz);

  // go through once more, this time filling in the col_ind and values arrays
  threadpool::parallel_for(slice_nrows, [&](uint32_t i) {
    int row = (slice.rows.size() == 0) ? i : slice.rows[i];
    int nz = row_ptr[i];   // the slice's row, not the source's

    int j = 0;
    for (int p = slice.spmat->row_ptr[row]; p < slice.spmat->row_ptr[row+1]; p++) {
      int col = slice.spmat->col_ind[p];
      if (slice.cols.size() != 0) {
        while (j < slice.cols.size() && slice.cols[j] < col) { j++; };
        if (j < slice.cols.size() && col == slice.cols[j]) {
          col_ind[nz] = j;
          values[nz] = slice.spmat->values[p];
          nz++;
        };
      } else {
        col_ind[nz] = j;
        values[nz] = slice.spmat->values[p];
        nz++;
      }
    }
  });

}

template <>
sparse_matrix<>::sparse_matrix(const sparse_matrix_slice<> & slice) {
  *this = slice;
}


sparse_matrix<> triu(const sparse_matrix<> & A) {

  sparse_matrix<> output;
  output.nrows = A.nrows;
  output.ncols = A.ncols;
  output.nnz = 0;

  output.row_ptr.resize(A.nrows + 1);

  for (int row = 0; row < A.nrows; row++) {
    for (int nz = A.row_ptr[row]; nz < A.row_ptr[row+1]; nz++) {
      int col = A.col_ind[nz];
      if (row <= col) { 
        output.row_ptr[row+1]++;
        output.nnz++; 
      }
    }
  }

  output.row_ptr[0] = 0;
  for (int row = 1; row <= A.nrows; row++) {
    output.row_ptr[row] += output.row_ptr[row-1];
  }

  output.col_ind.resize(output.nnz);
  output.values.resize(output.nnz);

  int k = 0;
  for (int row = 0; row < A.nrows; row++) {
    for (int nz = A.row_ptr[row]; nz < A.row_ptr[row+1]; nz++) {
      int col = A.col_ind[nz];
      if (row <= col) {
        output.values[k] = A.values[nz];
        output.col_ind[k] = col;
        k++; 
      }
    }
  }

  return output;

}

sparse_matrix<> tril(const sparse_matrix<> & A) {

  sparse_matrix<> output;
  output.nrows = A.nrows;
  output.ncols = A.ncols;
  output.nnz = 0;

  output.row_ptr.resize(A.nrows + 1);
  zero(output.row_ptr);

  for (int row = 0; row < A.nrows; row++) {
    for (int nz = A.row_ptr[row]; nz < A.row_ptr[row+1]; nz++) {
      int col = A.col_ind[nz];
      if (row >= col) { 
        output.row_ptr[row+1]++;
        output.nnz++; 
      }
    }
  }

  output.row_ptr[0] = 0;
  for (int row = 1; row <= A.nrows; row++) {
    output.row_ptr[row] += output.row_ptr[row-1];
  }

  output.col_ind.resize(output.nnz);
  output.values.resize(output.nnz);

  int k = 0;
  for (int row = 0; row < A.nrows; row++) {
    for (int nz = A.row_ptr[row]; nz < A.row_ptr[row+1]; nz++) {
      int col = A.col_ind[nz];
      if (row >= col) {
        output.values[k] = A.values[nz];
        output.col_ind[k] = col;
        k++; 
      }
    }
  }

  return output;

}

}
