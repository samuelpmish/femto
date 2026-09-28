// CPU prototype kernels for sparse matrix / residual integration on
// linear-geometry tets, in the style of the FFCx-generated DOLFINx kernels:
// compile-time sizes, restrict pointers, a stack element matrix filled in one
// pass over the qdata, and a single scatter per element.
//
// Included twice by spmat_cpu_perf.cpp under different PROTO_NS namespaces,
// the second time inside `#pragma GCC target("avx2,fma")`, to separate the
// effect of the refactor from the effect of the instruction set.
//
// A_e layout is [I*NC+i][j][J] so the innermost trial-node loop is unit stride.

namespace PROTO_NS {

// dX/dxi is constant on a linear tet: compute it once per element
template < uint32_t NNX >
inline double element_jacobian(const uint32_t * __restrict X_ids,
                               const double * __restrict X,
                               const double * __restrict GX,   // (NNX, 3), any qpt row
                               double Ji[3][3]) {
  double J[3][3] = {};
  for (uint32_t i = 0; i < NNX; i++) {
    const double * x = X + X_ids[i] * 3;
    for (uint32_t d = 0; d < 3; d++) {
      for (uint32_t c = 0; c < 3; c++) { J[d][c] += x[d] * GX[i * 3 + c]; }
    }
  }
  double detJ = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1])
              - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0])
              + J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
  double inv_det = 1.0 / detJ;
  Ji[0][0] = (J[1][1] * J[2][2] - J[1][2] * J[2][1]) * inv_det;
  Ji[0][1] = (J[0][2] * J[2][1] - J[0][1] * J[2][2]) * inv_det;
  Ji[0][2] = (J[0][1] * J[1][2] - J[0][2] * J[1][1]) * inv_det;
  Ji[1][0] = (J[1][2] * J[2][0] - J[1][0] * J[2][2]) * inv_det;
  Ji[1][1] = (J[0][0] * J[2][2] - J[0][2] * J[2][0]) * inv_det;
  Ji[1][2] = (J[0][2] * J[1][0] - J[0][0] * J[1][2]) * inv_det;
  Ji[2][0] = (J[1][0] * J[2][1] - J[1][1] * J[2][0]) * inv_det;
  Ji[2][1] = (J[0][1] * J[2][0] - J[0][0] * J[2][1]) * inv_det;
  Ji[2][2] = (J[0][0] * J[1][1] - J[0][1] * J[1][0]) * inv_det;
  return detJ;
}

// stiffness over the element range [e0, e1).  collapse folds the quadrature
// loop into the qdata (valid when the FE gradients are constant, i.e. P == 1):
// A_e = detJ * dN_I . (sum_q w_q C_q) . dN_J
template < uint32_t NC, uint32_t NN, uint32_t NQ, bool collapse >
void spmat_range(uint32_t e0, uint32_t e1,
                 double * __restrict values,
                 const int * __restrict row_ptr,
                 const uint16_t * __restrict pair_k,    // (nelem, NN*NN)
                 const uint32_t * __restrict ids,       // (nelem, NN)
                 const uint32_t * __restrict X_ids,     // (nelem, 4)
                 const double * __restrict X,           // (nnodes, 3)
                 const double * __restrict qdata,       // (nq, DSZ)
                 const double * __restrict G,           // (NQ, NN, 3) reference gradients
                 const double * __restrict GX,          // (4, 3)
                 const double * __restrict W,           // (NQ)
                 std::mutex * mutexes) {
  constexpr uint32_t DSZ = NC * 3 * NC * 3;
  static_assert(!collapse || NN == 4, "collapse requires constant basis gradients (p1)");

  for (uint32_t e = e0; e < e1; e++) {
    double Ji[3][3];
    double detJ = element_jacobian<4>(X_ids + e * 4, X, GX, Ji);

    // physical gradients, SoA over nodes: g[q][m][J]
    double g[collapse ? 1 : NQ][3][NN];
    constexpr uint32_t NGQ = collapse ? 1 : NQ;
    for (uint32_t q = 0; q < NGQ; q++) {
      for (uint32_t I = 0; I < NN; I++) {
        const double * gr = G + (q * NN + I) * 3;
        for (uint32_t m = 0; m < 3; m++) {
          g[q][m][I] = gr[0] * Ji[0][m] + gr[1] * Ji[1][m] + gr[2] * Ji[2][m];
        }
      }
    }

    double A[NN * NC][NC][NN] = {};

    if constexpr (collapse) {
      double Chat[DSZ];
      for (uint32_t c = 0; c < DSZ; c++) {
        double s = 0.0;
        for (uint32_t q = 0; q < NQ; q++) { s += W[q] * qdata[(e * NQ + q) * DSZ + c]; }
        Chat[c] = s * detJ;
      }
      for (uint32_t I = 0; I < NN; I++) {
        for (uint32_t i = 0; i < NC; i++) {
          for (uint32_t k = 0; k < 3; k++) {
            double gIk = g[0][k][I];
            for (uint32_t j = 0; j < NC; j++) {
              for (uint32_t m = 0; m < 3; m++) {
                double coef = gIk * Chat[((i * 3 + k) * NC + j) * 3 + m];
                for (uint32_t J = 0; J < NN; J++) {
                  A[I * NC + i][j][J] += coef * g[0][m][J];
                }
              }
            }
          }
        }
      }
    } else {
      for (uint32_t q = 0; q < NQ; q++) {
        const double * C = qdata + (e * NQ + q) * DSZ;
        double wJ = W[q] * detJ;
        for (uint32_t I = 0; I < NN; I++) {
          for (uint32_t i = 0; i < NC; i++) {
            for (uint32_t k = 0; k < 3; k++) {
              double c1 = wJ * g[q][k][I];
              for (uint32_t j = 0; j < NC; j++) {
                for (uint32_t m = 0; m < 3; m++) {
                  double coef = c1 * C[((i * 3 + k) * NC + j) * 3 + m];
                  for (uint32_t J = 0; J < NN; J++) {
                    A[I * NC + i][j][J] += coef * g[q][m][J];
                  }
                }
              }
            }
          }
        }
      }
    }

    // one scatter per element via the precomputed block-column indices
    const uint16_t * pk = pair_k + e * NN * NN;
    for (uint32_t I = 0; I < NN; I++) {
      uint32_t row_node = ids[e * NN + I];
      for (uint32_t i = 0; i < NC; i++) {
        int base = row_ptr[row_node * NC + i];
        if (mutexes) { mutexes[row_node % 1024].lock(); }
        for (uint32_t J = 0; J < NN; J++) {
          int pos = base + int(pk[I * NN + J]) * NC;
          for (uint32_t j = 0; j < NC; j++) {
            values[pos + j] += A[I * NC + i][j][J];
          }
        }
        if (mutexes) { mutexes[row_node % 1024].unlock(); }
      }
    }
  }
}

// residual over the element range [e0, e1) (serial scatter: += with no lock)
template < uint32_t NC, uint32_t NN, uint32_t NQ, bool collapse >
void residual_range(uint32_t e0, uint32_t e1,
                    double * __restrict r,              // (nnodes, NC)
                    const uint32_t * __restrict ids,
                    const uint32_t * __restrict X_ids,
                    const double * __restrict X,
                    const double * __restrict f,        // (nq, NC*3)
                    const double * __restrict G,
                    const double * __restrict GX,
                    const double * __restrict W) {
  static_assert(!collapse || NN == 4, "collapse requires constant basis gradients (p1)");

  for (uint32_t e = e0; e < e1; e++) {
    double Ji[3][3];
    double detJ = element_jacobian<4>(X_ids + e * 4, X, GX, Ji);

    double r_e[NN][NC] = {};

    if constexpr (collapse) {
      double fh[NC][3] = {};
      for (uint32_t q = 0; q < NQ; q++) {
        const double * fq = f + (e * NQ + q) * NC * 3;
        for (uint32_t c = 0; c < NC; c++) {
          for (uint32_t d = 0; d < 3; d++) { fh[c][d] += W[q] * fq[c * 3 + d]; }
        }
      }
      for (uint32_t I = 0; I < NN; I++) {
        const double * gr = G + I * 3;
        double dN[3];
        for (uint32_t m = 0; m < 3; m++) {
          dN[m] = gr[0] * Ji[0][m] + gr[1] * Ji[1][m] + gr[2] * Ji[2][m];
        }
        for (uint32_t c = 0; c < NC; c++) {
          r_e[I][c] = detJ * (dN[0] * fh[c][0] + dN[1] * fh[c][1] + dN[2] * fh[c][2]);
        }
      }
    } else {
      for (uint32_t q = 0; q < NQ; q++) {
        const double * fq = f + (e * NQ + q) * NC * 3;
        double wJ = W[q] * detJ;
        for (uint32_t I = 0; I < NN; I++) {
          const double * gr = G + (q * NN + I) * 3;
          double dN[3];
          for (uint32_t m = 0; m < 3; m++) {
            dN[m] = gr[0] * Ji[0][m] + gr[1] * Ji[1][m] + gr[2] * Ji[2][m];
          }
          for (uint32_t c = 0; c < NC; c++) {
            r_e[I][c] += wJ * (dN[0] * fq[c * 3 + 0] + dN[1] * fq[c * 3 + 1] + dN[2] * fq[c * 3 + 2]);
          }
        }
      }
    }

    for (uint32_t I = 0; I < NN; I++) {
      double * out = r + ids[e * NN + I] * NC;
      for (uint32_t c = 0; c < NC; c++) { out[c] += r_e[I][c]; }
    }
  }
}

} // namespace PROTO_NS
