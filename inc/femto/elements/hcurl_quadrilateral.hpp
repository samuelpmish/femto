#pragma once

#include "fm/types/vec.hpp"

namespace femto {

using namespace fm;

// clang-format off
template <>
struct FiniteElement<Geometry::Quadrilateral, Family::Hcurl> {

  using value_type = vec2;
  using derivative_type = vec1;

  using source_type = vec2;
  using flux_type = vec1;

  static constexpr int dim = 2;

  __host__ __device__ uint32_t num_nodes() const { return 2 * p * (p + 1); }

  void nodes(nd::view<double,2> xi) const {
    if (p == 1) {
      xi(0, 0) = 0.5; xi(0, 1) = 0;
      xi(1, 0) = 0.5; xi(1, 1) = 1.0;
      xi(2, 0) = 0; xi(2, 1) = 0.5;
      xi(3, 0) = 1.0; xi(3, 1) = 0.5;
    }
    if (p == 2) {
      xi(0, 0) = 0.21132486540518711775; xi(0, 1) = 0;
      xi(1, 0) = 0.78867513459481288225; xi(1, 1) = 0;
      xi(2, 0) = 0.21132486540518711775; xi(2, 1) = 0.5;
      xi(3, 0) = 0.78867513459481288225; xi(3, 1) = 0.5;
      xi(4, 0) = 0.21132486540518711775; xi(4, 1) = 1.0;
      xi(5, 0) = 0.78867513459481288225; xi(5, 1) = 1.0;
      xi(6, 0) = 0; xi(6, 1) = 0.21132486540518711775;
      xi(7, 0) = 0; xi(7, 1) = 0.78867513459481288225;
      xi(8, 0) = 0.5; xi(8, 1) = 0.21132486540518711775;
      xi(9, 0) = 0.5; xi(9, 1) = 0.78867513459481288225;
      xi(10, 0) = 1.0; xi(10, 1) = 0.21132486540518711775;
      xi(11, 0) = 1.0; xi(11, 1) = 0.78867513459481288225;
    }
    if (p == 3) {
      xi(0, 0) = 0.11270166537925831148; xi(0, 1) = 0;
      xi(1, 0) = 0.5; xi(1, 1) = 0;
      xi(2, 0) = 0.88729833462074168852; xi(2, 1) = 0;
      xi(3, 0) = 0.11270166537925831148; xi(3, 1) = 0.27639320225002103036;
      xi(4, 0) = 0.5; xi(4, 1) = 0.27639320225002103036;
      xi(5, 0) = 0.88729833462074168852; xi(5, 1) = 0.27639320225002103036;
      xi(6, 0) = 0.11270166537925831148; xi(6, 1) = 0.72360679774997896964;
      xi(7, 0) = 0.5; xi(7, 1) = 0.72360679774997896964;
      xi(8, 0) = 0.88729833462074168852; xi(8, 1) = 0.72360679774997896964;
      xi(9, 0) = 0.11270166537925831148; xi(9, 1) = 1.0;
      xi(10, 0) = 0.5; xi(10, 1) = 1.0;
      xi(11, 0) = 0.88729833462074168852; xi(11, 1) = 1.0;
      xi(12, 0) = 0; xi(12, 1) = 0.11270166537925831148;
      xi(13, 0) = 0; xi(13, 1) = 0.5;
      xi(14, 0) = 0; xi(14, 1) = 0.88729833462074168852;
      xi(15, 0) = 0.27639320225002103036; xi(15, 1) = 0.11270166537925831148;
      xi(16, 0) = 0.27639320225002103036; xi(16, 1) = 0.5;
      xi(17, 0) = 0.27639320225002103036; xi(17, 1) = 0.88729833462074168852;
      xi(18, 0) = 0.72360679774997896964; xi(18, 1) = 0.11270166537925831148;
      xi(19, 0) = 0.72360679774997896964; xi(19, 1) = 0.5;
      xi(20, 0) = 0.72360679774997896964; xi(20, 1) = 0.88729833462074168852;
      xi(21, 0) = 1.0; xi(21, 1) = 0.11270166537925831148;
      xi(22, 0) = 1.0; xi(22, 1) = 0.5;
      xi(23, 0) = 1.0; xi(23, 1) = 0.88729833462074168852;
    }
  }

  void directions(nd::view<double,2> d) const {
    int i = 0;
    for (int k = 0; k < (p * (p + 1)); k++) { d(i, 0) =  1.0; d(i++, 1) =  0.0; }
    for (int k = 0; k < (p * (p + 1)); k++) { d(i, 0) =  0.0; d(i++, 1) =  1.0; }
  }

  __host__ __device__ uint32_t num_interior_nodes() const { return (p > 1) ? 2 * p * (p - 1) : 0; }

  void interior_nodes(nd::view<double,2> xi) const {
    if (p == 2) {
      xi(0, 0) = 0.21132486540518711775; xi(0, 1) = 0.5;
      xi(1, 0) = 0.78867513459481288225; xi(1, 1) = 0.5;
      xi(2, 0) = 0.5; xi(2, 1) = 0.21132486540518711775;
      xi(3, 0) = 0.5; xi(3, 1) = 0.78867513459481288225;
    }
    if (p == 3) {
      xi(0, 0) = 0.11270166537925831148; xi(0, 1) = 0.27639320225002103036;
      xi(1, 0) = 0.5; xi(1, 1) = 0.27639320225002103036;
      xi(2, 0) = 0.88729833462074168852; xi(2, 1) = 0.27639320225002103036;
      xi(3, 0) = 0.11270166537925831148; xi(3, 1) = 0.72360679774997896964;
      xi(4, 0) = 0.5; xi(4, 1) = 0.72360679774997896964;
      xi(5, 0) = 0.88729833462074168852; xi(5, 1) = 0.72360679774997896964;
      xi(6, 0) = 0.27639320225002103036; xi(6, 1) = 0.11270166537925831148;
      xi(7, 0) = 0.27639320225002103036; xi(7, 1) = 0.5;
      xi(8, 0) = 0.27639320225002103036; xi(8, 1) = 0.88729833462074168852;
      xi(9, 0) = 0.72360679774997896964; xi(9, 1) = 0.11270166537925831148;
      xi(10, 0) = 0.72360679774997896964; xi(10, 1) = 0.5;
      xi(11, 0) = 0.72360679774997896964; xi(11, 1) = 0.88729833462074168852;
    }
  }

  void interior_directions(nd::view<double,2> d) const {
    if (p > 1) {
      int i = 0;
      for (int k = 0; k < (p * (p - 1)); k++) { d(i, 0) =  1.0; d(i++, 1) =  0.0; }
      for (int k = 0; k < (p * (p - 1)); k++) { d(i, 0) =  0.0; d(i++, 1) =  1.0; }
    }
  }

  __host__ __device__ void indices(const GeometryInfo & offsets, const Connection * quad, uint32_t * indices) const {
    
    const Connection * edge = quad + Quadrilateral::edge_offset;
    const Connection cell = *(quad + Quadrilateral::cell_offset);

    if (p == 1) {

      // o-----→-----o         o-----1-----o
      // |           |         |           |
      // |           |         |           |
      // ↑           ↑         2           3
      // |           |         |           |
      // |           |         |           |
      // o-----→-----o         o-----0-----o
      indices[0] = offsets.edge + edge[0].index;
      indices[1] = offsets.edge + edge[2].index;
      indices[2] = offsets.edge + edge[3].index;
      indices[3] = offsets.edge + edge[1].index;
      return;

    }

    if (p == 2) {

      // o---→---→---o         o---4---5---o
      // |           |         |           |
      // ↑     ↑     ↑         7     9    11
      // |   →   →   |         |   2   3   |
      // ↑     ↑     ↑         6     8    10
      // |           |         |           |
      // o---→---→---o         o---0---1---o
      indices[0] = offsets.edge + 2 * edge[0].index + 0;
      indices[1] = offsets.edge + 2 * edge[0].index + 1;
      indices[2] = offsets.quad + 4 * cell.index + 0;
      indices[3] = offsets.quad + 4 * cell.index + 1;
      indices[4] = offsets.edge + 2 * edge[2].index + 1;
      indices[5] = offsets.edge + 2 * edge[2].index + 0;

      indices[ 6] = offsets.edge + 2 * edge[3].index + 1;
      indices[ 7] = offsets.edge + 2 * edge[3].index + 0;
      indices[ 8] = offsets.quad + 4 * cell.index + 2;
      indices[ 9] = offsets.quad + 4 * cell.index + 3;
      indices[10] = offsets.edge + 2 * edge[1].index + 0;
      indices[11] = offsets.edge + 2 * edge[1].index + 1;

      if (flip(edge[0])) { fm::swap(indices[ 0], indices[ 1]); }
      if (flip(edge[1])) { fm::swap(indices[10], indices[11]); }
      if (flip(edge[2])) { fm::swap(indices[ 4], indices[ 5]); }
      if (flip(edge[3])) { fm::swap(indices[ 6], indices[ 7]); }

    }

    if (p == 3) {

      // o--→--→--→--o         o--9-10-11--o
      // ↑   ↑   ↑   ↑         14  17 20  23
      // |  →  →  →  |         |  6  7  8  |
      // ↑   ↑   ↑   ↑         13  16 19  22
      // |  →  →  →  |         |  3  4  5  |
      // ↑   ↑   ↑   ↑         12  15 18  21
      // o--→--→--→--o         o--0--1--2--o
      indices[ 0] = offsets.edge + 3 * edge[0].index + 0;
      indices[ 1] = offsets.edge + 3 * edge[0].index + 1;
      indices[ 2] = offsets.edge + 3 * edge[0].index + 2;
      indices[ 3] = offsets.quad + 12 * cell.index + 0;
      indices[ 4] = offsets.quad + 12 * cell.index + 1;
      indices[ 5] = offsets.quad + 12 * cell.index + 2;
      indices[ 6] = offsets.quad + 12 * cell.index + 3;
      indices[ 7] = offsets.quad + 12 * cell.index + 4;
      indices[ 8] = offsets.quad + 12 * cell.index + 5;
      indices[ 9] = offsets.edge + 3 * edge[2].index + 2;
      indices[10] = offsets.edge + 3 * edge[2].index + 1;
      indices[11] = offsets.edge + 3 * edge[2].index + 0;

      indices[12] = offsets.edge + 3 * edge[3].index + 2;
      indices[13] = offsets.edge + 3 * edge[3].index + 1;
      indices[14] = offsets.edge + 3 * edge[3].index + 0;
      indices[15] = offsets.quad + 12 * cell.index +  6;
      indices[16] = offsets.quad + 12 * cell.index +  7;
      indices[17] = offsets.quad + 12 * cell.index +  8;
      indices[18] = offsets.quad + 12 * cell.index +  9;
      indices[19] = offsets.quad + 12 * cell.index + 10;
      indices[20] = offsets.quad + 12 * cell.index + 11;
      indices[21] = offsets.edge + 3 * edge[1].index + 0;
      indices[22] = offsets.edge + 3 * edge[1].index + 1;
      indices[23] = offsets.edge + 3 * edge[1].index + 2;

      if (flip(edge[0])) { fm::swap(indices[ 0], indices[ 2]); }
      if (flip(edge[1])) { fm::swap(indices[21], indices[23]); }
      if (flip(edge[2])) { fm::swap(indices[ 9], indices[11]); }
      if (flip(edge[3])) { fm::swap(indices[12], indices[14]); }

    }

  }

  template < typename T >
  __host__ __device__ void reorient(const TransformationType type, const Connection * quad, T * values) const {

    const Connection * edge = quad + Quadrilateral::edge_offset;

    if (p == 1) {
      if (edge[0].sign() == Sign::Negative) { values[0] *= -1; }
      if (edge[1].sign() == Sign::Negative) { values[3] *= -1; }
      if (edge[2].sign() == Sign::Positive) { values[1] *= -1; }
      if (edge[3].sign() == Sign::Positive) { values[2] *= -1; }
      return;
    }

    if (p == 2) {
      if (edge[0].sign() == Sign::Negative) { values[ 0] *= -1; values[ 1] *= -1; }
      if (edge[1].sign() == Sign::Negative) { values[10] *= -1; values[11] *= -1; }
      if (edge[2].sign() == Sign::Positive) { values[ 4] *= -1; values[ 5] *= -1; }
      if (edge[3].sign() == Sign::Positive) { values[ 6] *= -1; values[ 7] *= -1; }
      return;
    }

    if (p == 3) {
      if (edge[0].sign() == Sign::Negative) { values[ 0] *= -1; values[ 1] *= -1; values[ 2] *= -1;}
      if (edge[1].sign() == Sign::Negative) { values[21] *= -1; values[22] *= -1; values[23] *= -1;}
      if (edge[2].sign() == Sign::Positive) { values[ 9] *= -1; values[10] *= -1; values[11] *= -1;}
      if (edge[3].sign() == Sign::Positive) { values[12] *= -1; values[13] *= -1; values[14] *= -1;}
      return;
    }

  }

  __host__ __device__ void reorient(const TransformationType type, const Connection * quad, int8_t * transformation) {

    const Connection * edge = quad + Quadrilateral::edge_offset;

    uint32_t nnodes = num_nodes();
    for (int k = 0; k < nnodes; k++) {
      transformation[k] = 0;
    }

    if (p == 1) {
      if (edge[0].sign() == Sign::Negative) { transformation[0] = -1; }
      if (edge[1].sign() == Sign::Negative) { transformation[3] = -1; }
      if (edge[2].sign() == Sign::Positive) { transformation[1] = -1; }
      if (edge[3].sign() == Sign::Positive) { transformation[2] = -1; }
      return;
    }

    if (p == 2) {
      if (edge[0].sign() == Sign::Negative) { transformation[ 0] = -1; transformation[ 1] = -1; }
      if (edge[1].sign() == Sign::Negative) { transformation[10] = -1; transformation[11] = -1; }
      if (edge[2].sign() == Sign::Positive) { transformation[ 4] = -1; transformation[ 5] = -1; }
      if (edge[3].sign() == Sign::Positive) { transformation[ 6] = -1; transformation[ 7] = -1; }
      return;
    }

    if (p == 3) {
      if (edge[0].sign() == Sign::Negative) { transformation[ 0] = -1; transformation[ 1] = -1; transformation[ 2] = -1;}
      if (edge[1].sign() == Sign::Negative) { transformation[21] = -1; transformation[22] = -1; transformation[23] = -1;}
      if (edge[2].sign() == Sign::Positive) { transformation[ 9] = -1; transformation[10] = -1; transformation[11] = -1;}
      if (edge[3].sign() == Sign::Positive) { transformation[12] = -1; transformation[13] = -1; transformation[14] = -1;}
      return;
    }

  }

  __host__ __device__ constexpr vec2 shape_function(vec2 xi, uint32_t i) const {
    if (p == 1) {
      if (i == 0) { return vec2{1 - xi[1],0}; }
      if (i == 1) { return vec2{xi[1],0}; }
      if (i == 2) { return vec2{0,1 - xi[0]}; }
      if (i == 3) { return vec2{0,xi[0]}; }
    }
    if (p == 2) {
      if (i == 0) { return vec2{-1.7320508075688772935274463415*(-0.7886751345948128822545743902 + xi[0])*(-1 + xi[1])*(-1 + 2*xi[1]),0}; }
      if (i == 1) { return vec2{(-0.3660254037844386467637231708 + 1.7320508075688772935274463415*xi[0])*(-1 + xi[1])*(-1 + 2*xi[1]),0}; }
      if (i == 2) { return vec2{6.928203230275509174109785366*(-0.7886751345948128822545743902 + xi[0])*(-1. + xi[1])*xi[1],0}; }
      if (i == 3) { return vec2{-6.92820323027550917410978537*(-0.21132486540518711774542561 + 1.*xi[0])*(-1. + xi[1])*xi[1],0}; }
      if (i == 4) { return vec2{-1.7320508075688772935274463415*(-0.7886751345948128822545743902 + xi[0])*xi[1]*(-1 + 2*xi[1]),0}; }
      if (i == 5) { return vec2{(-0.3660254037844386467637231708 + 1.7320508075688772935274463415*xi[0])*xi[1]*(-1 + 2*xi[1]),0}; }
      if (i == 6) { return vec2{0,-1.7320508075688772935274463415*(-1 + xi[0])*(-1 + 2*xi[0])*(-0.7886751345948128822545743902 + xi[1])}; }
      if (i == 7) { return vec2{0,(-1 + xi[0])*(-1 + 2*xi[0])*(-0.3660254037844386467637231708 + 1.7320508075688772935274463415*xi[1])}; }
      if (i == 8) { return vec2{0,6.928203230275509174109785366*(-1. + xi[0])*xi[0]*(-0.7886751345948128822545743902 + xi[1])}; }
      if (i == 9) { return vec2{0,-6.92820323027550917410978537*(-1. + xi[0])*xi[0]*(-0.21132486540518711774542561 + 1.*xi[1])}; }
      if (i == 10) { return vec2{0,-1.7320508075688772935274463415*xi[0]*(-1 + 2*xi[0])*(-0.7886751345948128822545743902 + xi[1])}; }
      if (i == 11) { return vec2{0,xi[0]*(-1 + 2*xi[0])*(-0.3660254037844386467637231708 + 1.7320508075688772935274463415*xi[1])}; }
    }
    if (p == 3) {
      if (i == 0) { return vec2{(1.4788305577012361475298776 + xi[0]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[0]))*(1 + xi[1]*(-6. + (10. - 5.*xi[1])*xi[1])),0}; }
      if (i == 1) { return vec2{-6.66666666666666666666666667*(-0.88729833462074168851792654 + xi[0])*(-0.11270166537925831148207346 + xi[0])*(1 + xi[1]*(-6. + (10. - 5.*xi[1])*xi[1])),0}; }
      if (i == 2) { return vec2{(0.1878361089654305191367891 + xi[0]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[0]))*(1 + xi[1]*(-6. + (10. - 5.*xi[1])*xi[1])),0}; }
      if (i == 3) { return vec2{11.180339887498948*(1.4788305577012361475298776 + xi[0]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[0]))*(-1. + xi[1])*xi[1]*(-0.723606797749979 + 1.*xi[1]),0}; }
      if (i == 4) { return vec2{-74.53559924999299*(-0.88729833462074168851792654 + xi[0])*(-0.11270166537925831148207346 + xi[0])*(-1. + xi[1])*xi[1]*(-0.723606797749979 + 1.*xi[1]),0}; }
      if (i == 5) { return vec2{11.180339887498948*(0.1878361089654305191367891 + xi[0]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[0]))*(-1. + xi[1])*xi[1]*(-0.723606797749979 + 1.*xi[1]),0}; }
      if (i == 6) { return vec2{-11.180339887498948482*(1.4788305577012361475298776 + xi[0]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[0]))*(-1. + xi[1])*(-0.2763932022500210304 + xi[1])*xi[1],0}; }
      if (i == 7) { return vec2{74.53559924999298988*(-0.88729833462074168851792654 + xi[0])*(-0.11270166537925831148207346 + xi[0])*(-1. + xi[1])*(-0.2763932022500210304 + xi[1])*xi[1],0}; }
      if (i == 8) { return vec2{-11.180339887498948482*(0.1878361089654305191367891 + xi[0]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[0]))*(-1. + xi[1])*(-0.2763932022500210304 + xi[1])*xi[1],0}; }
      if (i == 9) { return vec2{(1.4788305577012361475298776 + xi[0]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[0]))*xi[1]*(1. + xi[1]*(-5. + 5.*xi[1])),0}; }
      if (i == 10) { return vec2{-6.66666666666666666666666667*(-0.88729833462074168851792654 + xi[0])*(-0.11270166537925831148207346 + xi[0])*xi[1]*(1. + xi[1]*(-5. + 5.*xi[1])),0}; }
      if (i == 11) { return vec2{(0.1878361089654305191367891 + xi[0]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[0]))*xi[1]*(1. + xi[1]*(-5. + 5.*xi[1])),0}; }
      if (i == 12) { return vec2{0,(1 + xi[0]*(-6. + (10. - 5.*xi[0])*xi[0]))*(1.4788305577012361475298776 + xi[1]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[1]))}; }
      if (i == 13) { return vec2{0,-6.66666666666666666666666667*(1 + xi[0]*(-6. + (10. - 5.*xi[0])*xi[0]))*(-0.88729833462074168851792654 + xi[1])*(-0.11270166537925831148207346 + xi[1])}; }
      if (i == 14) { return vec2{0,(1 + xi[0]*(-6. + (10. - 5.*xi[0])*xi[0]))*(0.1878361089654305191367891 + xi[1]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[1]))}; }
      if (i == 15) { return vec2{0,11.180339887498948*(-1. + xi[0])*xi[0]*(-0.723606797749979 + 1.*xi[0])*(1.4788305577012361475298776 + xi[1]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[1]))}; }
      if (i == 16) { return vec2{0,-74.53559924999299*(-1. + xi[0])*xi[0]*(-0.723606797749979 + 1.*xi[0])*(-0.88729833462074168851792654 + xi[1])*(-0.11270166537925831148207346 + xi[1])}; }
      if (i == 17) { return vec2{0,11.180339887498948*(-1. + xi[0])*xi[0]*(-0.723606797749979 + 1.*xi[0])*(0.1878361089654305191367891 + xi[1]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[1]))}; }
      if (i == 18) { return vec2{0,-11.180339887498948482*(-1. + xi[0])*(-0.2763932022500210304 + xi[0])*xi[0]*(1.4788305577012361475298776 + xi[1]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[1]))}; }
      if (i == 19) { return vec2{0,74.53559924999298988*(-1. + xi[0])*(-0.2763932022500210304 + xi[0])*xi[0]*(-0.88729833462074168851792654 + xi[1])*(-0.11270166537925831148207346 + xi[1])}; }
      if (i == 20) { return vec2{0,-11.180339887498948482*(-1. + xi[0])*(-0.2763932022500210304 + xi[0])*xi[0]*(0.1878361089654305191367891 + xi[1]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[1]))}; }
      if (i == 21) { return vec2{0,xi[0]*(1. + xi[0]*(-5. + 5.*xi[0]))*(1.4788305577012361475298776 + xi[1]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[1]))}; }
      if (i == 22) { return vec2{0,-6.66666666666666666666666667*xi[0]*(1. + xi[0]*(-5. + 5.*xi[0]))*(-0.88729833462074168851792654 + xi[1])*(-0.11270166537925831148207346 + xi[1])}; }
      if (i == 23) { return vec2{0,xi[0]*(1. + xi[0]*(-5. + 5.*xi[0]))*(0.1878361089654305191367891 + xi[1]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[1]))}; }
    }

    return {};
  }

  __host__ __device__ constexpr vec2 reoriented_shape_function(vec2 xi, uint32_t i, int8_t transformation) const {
    if (transformation == -1) {
      return -shape_function(xi, i);
    } else {
      return  shape_function(xi, i);
    }
  }

  __host__ __device__ vec<1> shape_function_curl(vec2 xi, uint32_t i) const {
    // expressions generated symbolically by mathematica
    if (p == 1) {
      if (i == 0) { return 1; }
      if (i == 1) { return -1; }
      if (i == 2) { return -1; }
      if (i == 3) { return 1; }
    }
    if (p == 2) {
      if (i == 0) { return 3.464101615137754587054892683*(-0.7886751345948128822545743902 + xi[0])*(-1 + xi[1]) + 1.7320508075688772935274463415*(-0.7886751345948128822545743902 + xi[0])*(-1 + 2*xi[1]); }
      if (i == 1) { return -2*(-0.3660254037844386467637231708 + 1.7320508075688772935274463415*xi[0])*(-1 + xi[1]) - (-0.3660254037844386467637231708 + 1.7320508075688772935274463415*xi[0])*(-1 + 2*xi[1]); }
      if (i == 2) { return -6.928203230275509174109785366*(-0.7886751345948128822545743902 + xi[0])*(-1. + xi[1]) - 6.928203230275509174109785366*(-0.7886751345948128822545743902 + xi[0])*xi[1]; }
      if (i == 3) { return 6.92820323027550917410978537*(-0.21132486540518711774542561 + 1.*xi[0])*(-1. + xi[1]) + 6.92820323027550917410978537*(-0.21132486540518711774542561 + 1.*xi[0])*xi[1]; }
      if (i == 4) { return 3.464101615137754587054892683*(-0.7886751345948128822545743902 + xi[0])*xi[1] + 1.7320508075688772935274463415*(-0.7886751345948128822545743902 + xi[0])*(-1 + 2*xi[1]); }
      if (i == 5) { return -2*(-0.3660254037844386467637231708 + 1.7320508075688772935274463415*xi[0])*xi[1] - (-0.3660254037844386467637231708 + 1.7320508075688772935274463415*xi[0])*(-1 + 2*xi[1]); }
      if (i == 6) { return -3.464101615137754587054892683*(-1 + xi[0])*(-0.7886751345948128822545743902 + xi[1]) - 1.7320508075688772935274463415*(-1 + 2*xi[0])*(-0.7886751345948128822545743902 + xi[1]); }
      if (i == 7) { return 2*(-1 + xi[0])*(-0.3660254037844386467637231708 + 1.7320508075688772935274463415*xi[1]) + (-1 + 2*xi[0])*(-0.3660254037844386467637231708 + 1.7320508075688772935274463415*xi[1]); }
      if (i == 8) { return 6.928203230275509174109785366*(-1. + xi[0])*(-0.7886751345948128822545743902 + xi[1]) + 6.928203230275509174109785366*xi[0]*(-0.7886751345948128822545743902 + xi[1]); }
      if (i == 9) { return -6.92820323027550917410978537*(-1. + xi[0])*(-0.21132486540518711774542561 + 1.*xi[1]) - 6.92820323027550917410978537*xi[0]*(-0.21132486540518711774542561 + 1.*xi[1]); }
      if (i == 10) { return -3.464101615137754587054892683*xi[0]*(-0.7886751345948128822545743902 + xi[1]) - 1.7320508075688772935274463415*(-1 + 2*xi[0])*(-0.7886751345948128822545743902 + xi[1]); }
      if (i == 11) { return 2*xi[0]*(-0.3660254037844386467637231708 + 1.7320508075688772935274463415*xi[1]) + (-1 + 2*xi[0])*(-0.3660254037844386467637231708 + 1.7320508075688772935274463415*xi[1]); }
    }
    if (p == 3) {
      if (i == 0) { return -((1.4788305577012361475298776 + xi[0]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[0]))*(-6. + (10. - 10.*xi[1])*xi[1] + (10. - 5.*xi[1])*xi[1])); }
      if (i == 1) { return 6.66666666666666666666666667*(-0.88729833462074168851792654 + xi[0])*(-0.11270166537925831148207346 + xi[0])*(-6. + (10. - 10.*xi[1])*xi[1] + (10. - 5.*xi[1])*xi[1]); }
      if (i == 2) { return -((0.1878361089654305191367891 + xi[0]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[0]))*(-6. + (10. - 10.*xi[1])*xi[1] + (10. - 5.*xi[1])*xi[1])); }
      if (i == 3) { return -11.180339887498948*(1.4788305577012361475298776 + xi[0]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[0]))*(-1. + xi[1])*xi[1] - 11.180339887498948*(1.4788305577012361475298776 + xi[0]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[0]))*(-1. + xi[1])*(-0.723606797749979 + 1.*xi[1]) - 11.180339887498948*(1.4788305577012361475298776 + xi[0]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[0]))*xi[1]*(-0.723606797749979 + 1.*xi[1]); }
      if (i == 4) { return 74.53559924999299*(-0.88729833462074168851792654 + xi[0])*(-0.11270166537925831148207346 + xi[0])*(-1. + xi[1])*xi[1] + 74.53559924999299*(-0.88729833462074168851792654 + xi[0])*(-0.11270166537925831148207346 + xi[0])*(-1. + xi[1])*(-0.723606797749979 + 1.*xi[1]) + 74.53559924999299*(-0.88729833462074168851792654 + xi[0])*(-0.11270166537925831148207346 + xi[0])*xi[1]*(-0.723606797749979 + 1.*xi[1]); }
      if (i == 5) { return -11.180339887498948*(0.1878361089654305191367891 + xi[0]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[0]))*(-1. + xi[1])*xi[1] - 11.180339887498948*(0.1878361089654305191367891 + xi[0]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[0]))*(-1. + xi[1])*(-0.723606797749979 + 1.*xi[1]) - 11.180339887498948*(0.1878361089654305191367891 + xi[0]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[0]))*xi[1]*(-0.723606797749979 + 1.*xi[1]); }
      if (i == 6) { return 11.180339887498948482*(1.4788305577012361475298776 + xi[0]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[0]))*(-1. + xi[1])*(-0.2763932022500210304 + xi[1]) + 11.180339887498948482*(1.4788305577012361475298776 + xi[0]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[0]))*(-1. + xi[1])*xi[1] + 11.180339887498948482*(1.4788305577012361475298776 + xi[0]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[0]))*(-0.2763932022500210304 + xi[1])*xi[1]; }
      if (i == 7) { return -74.53559924999298988*(-0.88729833462074168851792654 + xi[0])*(-0.11270166537925831148207346 + xi[0])*(-1. + xi[1])*(-0.2763932022500210304 + xi[1]) - 74.53559924999298988*(-0.88729833462074168851792654 + xi[0])*(-0.11270166537925831148207346 + xi[0])*(-1. + xi[1])*xi[1] - 74.53559924999298988*(-0.88729833462074168851792654 + xi[0])*(-0.11270166537925831148207346 + xi[0])*(-0.2763932022500210304 + xi[1])*xi[1]; }
      if (i == 8) { return 11.180339887498948482*(0.1878361089654305191367891 + xi[0]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[0]))*(-1. + xi[1])*(-0.2763932022500210304 + xi[1]) + 11.180339887498948482*(0.1878361089654305191367891 + xi[0]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[0]))*(-1. + xi[1])*xi[1] + 11.180339887498948482*(0.1878361089654305191367891 + xi[0]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[0]))*(-0.2763932022500210304 + xi[1])*xi[1]; }
      if (i == 9) { return -((1.4788305577012361475298776 + xi[0]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[0]))*xi[1]*(-5. + 10.*xi[1])) - (1.4788305577012361475298776 + xi[0]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[0]))*(1. + xi[1]*(-5. + 5.*xi[1])); }
      if (i == 10) { return 6.66666666666666666666666667*(-0.88729833462074168851792654 + xi[0])*(-0.11270166537925831148207346 + xi[0])*xi[1]*(-5. + 10.*xi[1]) + 6.66666666666666666666666667*(-0.88729833462074168851792654 + xi[0])*(-0.11270166537925831148207346 + xi[0])*(1. + xi[1]*(-5. + 5.*xi[1])); }
      if (i == 11) { return -((0.1878361089654305191367891 + xi[0]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[0]))*xi[1]*(-5. + 10.*xi[1])) - (0.1878361089654305191367891 + xi[0]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[0]))*(1. + xi[1]*(-5. + 5.*xi[1])); }
      if (i == 12) { return (-6. + (10. - 10.*xi[0])*xi[0] + (10. - 5.*xi[0])*xi[0])*(1.4788305577012361475298776 + xi[1]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[1])); }
      if (i == 13) { return -6.66666666666666666666666667*(-6. + (10. - 10.*xi[0])*xi[0] + (10. - 5.*xi[0])*xi[0])*(-0.88729833462074168851792654 + xi[1])*(-0.11270166537925831148207346 + xi[1]); }
      if (i == 14) { return (-6. + (10. - 10.*xi[0])*xi[0] + (10. - 5.*xi[0])*xi[0])*(0.1878361089654305191367891 + xi[1]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[1])); }
      if (i == 15) { return 11.180339887498948*(-1. + xi[0])*xi[0]*(1.4788305577012361475298776 + xi[1]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[1])) + 11.180339887498948*(-1. + xi[0])*(-0.723606797749979 + 1.*xi[0])*(1.4788305577012361475298776 + xi[1]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[1])) + 11.180339887498948*xi[0]*(-0.723606797749979 + 1.*xi[0])*(1.4788305577012361475298776 + xi[1]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[1])); }
      if (i == 16) { return -74.53559924999299*(-1. + xi[0])*xi[0]*(-0.88729833462074168851792654 + xi[1])*(-0.11270166537925831148207346 + xi[1]) - 74.53559924999299*(-1. + xi[0])*(-0.723606797749979 + 1.*xi[0])*(-0.88729833462074168851792654 + xi[1])*(-0.11270166537925831148207346 + xi[1]) - 74.53559924999299*xi[0]*(-0.723606797749979 + 1.*xi[0])*(-0.88729833462074168851792654 + xi[1])*(-0.11270166537925831148207346 + xi[1]); }
      if (i == 17) { return 11.180339887498948*(-1. + xi[0])*xi[0]*(0.1878361089654305191367891 + xi[1]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[1])) + 11.180339887498948*(-1. + xi[0])*(-0.723606797749979 + 1.*xi[0])*(0.1878361089654305191367891 + xi[1]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[1])) + 11.180339887498948*xi[0]*(-0.723606797749979 + 1.*xi[0])*(0.1878361089654305191367891 + xi[1]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[1])); }
      if (i == 18) { return -11.180339887498948482*(-1. + xi[0])*(-0.2763932022500210304 + xi[0])*(1.4788305577012361475298776 + xi[1]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[1])) - 11.180339887498948482*(-1. + xi[0])*xi[0]*(1.4788305577012361475298776 + xi[1]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[1])) - 11.180339887498948482*(-0.2763932022500210304 + xi[0])*xi[0]*(1.4788305577012361475298776 + xi[1]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[1])); }
      if (i == 19) { return 74.53559924999298988*(-1. + xi[0])*(-0.2763932022500210304 + xi[0])*(-0.88729833462074168851792654 + xi[1])*(-0.11270166537925831148207346 + xi[1]) + 74.53559924999298988*(-1. + xi[0])*xi[0]*(-0.88729833462074168851792654 + xi[1])*(-0.11270166537925831148207346 + xi[1]) + 74.53559924999298988*(-0.2763932022500210304 + xi[0])*xi[0]*(-0.88729833462074168851792654 + xi[1])*(-0.11270166537925831148207346 + xi[1]); }
      if (i == 20) { return -11.180339887498948482*(-1. + xi[0])*(-0.2763932022500210304 + xi[0])*(0.1878361089654305191367891 + xi[1]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[1])) - 11.180339887498948482*(-1. + xi[0])*xi[0]*(0.1878361089654305191367891 + xi[1]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[1])) - 11.180339887498948482*(-0.2763932022500210304 + xi[0])*xi[0]*(0.1878361089654305191367891 + xi[1]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[1])); }
      if (i == 21) { return xi[0]*(-5. + 10.*xi[0])*(1.4788305577012361475298776 + xi[1]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[1])) + (1. + xi[0]*(-5. + 5.*xi[0]))*(1.4788305577012361475298776 + xi[1]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[1])); }
      if (i == 22) { return -6.66666666666666666666666667*xi[0]*(-5. + 10.*xi[0])*(-0.88729833462074168851792654 + xi[1])*(-0.11270166537925831148207346 + xi[1]) - 6.66666666666666666666666667*(1. + xi[0]*(-5. + 5.*xi[0]))*(-0.88729833462074168851792654 + xi[1])*(-0.11270166537925831148207346 + xi[1]); }
      if (i == 23) { return xi[0]*(-5. + 10.*xi[0])*(0.1878361089654305191367891 + xi[1]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[1])) + (1. + xi[0]*(-5. + 5.*xi[0]))*(0.1878361089654305191367891 + xi[1]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[1])); }
    }

    return {};
  }

  __host__ __device__ vec1 reoriented_shape_function_curl(vec2 xi, uint32_t i, int8_t transformation) const {
    if (transformation == -1) {
      return -shape_function_curl(xi, i);
    } else {
      return  shape_function_curl(xi, i);
    }
  }

  vec<1> shape_function_derivative(vec2 xi, uint32_t i) const {
    return shape_function_curl(xi, i);
  }

  vec2 interpolate(vec2 xi, const double * values) const {
    vec2 interpolated_value{};
    for (int i = 0; i < num_nodes(); i++) {
      interpolated_value += values[i] * shape_function(xi, i);
    }
    return interpolated_value;
  }

  vec<1> curl(vec2 xi, const double * values) const {
    double interpolated_curl = 0.0;
    for (int i = 0; i < num_nodes(); i++) {
      interpolated_curl += values[i] * shape_function_curl(xi, i);
    }
    return interpolated_curl;
  }
       
  // TODO: this is set to nonzero for the batched interpolation, 
  //       even though batched curl doesn't use the buffer
  __host__ __device__ uint32_t batch_interpolation_scratch_space(nd::view<const double,2> xi) const {
    uint32_t q = xi.shape[0];
    return q * (p + 1);
  }

  nd::array< double, 1, memory::space::cpu > evaluate_shape_functions(nd::view<const double,2> xi) const {
    uint32_t q = xi.shape[0];
    uint32_t num_entries = 0;
    num_entries += q * p;       // B1
    num_entries += q * (p + 1); // B2
    nd::array<double, 1, memory::space::cpu> buffer({num_entries});
    for (int i = 0; i < q; i++) {
      GaussLegendreInterpolation(xi(i, 0), p, &buffer(p*i));
      GaussLobattoInterpolation(xi(i, 0), p+1, &buffer((p+1)*i + (p*q)));
    }
    return buffer;
  }

  void interpolate(nd::view<value_type> values_q, nd::view<const double> values_e, nd::view<double> shape_fn, double * buffer) const {
    uint32_t n = p + 1;
    uint32_t q = sqrt(values_q.shape[0]);

    // 1D shape function evaluations
    nd::view<double, 2> B1(shape_fn.data(), {q, p}); // legendre shape functions
    nd::view<double, 2> B2(B1.end(),        {q, n}); // lobatto  shape functions

    nd::view<double, 2> A1(buffer, {q, n}); // storage for intermediates

    nd::view<const double, 2> ue(values_e.data(), {n, p});
    nd::view<double, 2> uq((double*)values_q.data(), {q, q}, {2*q, 2});
    
    _contract(A1, B1, ue); //  A1(qx, iy) = sum_{ix} B1(qx, ix) * ue(iy, ix)  
    _contract(uq, B2, A1); //  uq(qy, qx) = sum_{iy} B2(qy, iy) * A1(qx, iy)

    ue = nd::view< const double, 2 >(values_e.data() + (n * p), {n, p});

    // note: column-major strides here, since quadrature points are still 
    // enumerated lexicographically as {y, x} but y-component nodes are {x, y}
    uq = nd::view<double, 2>(((double*)values_q.data())+1, {q, q}, {2, 2*q});
    
    _contract(A1, B1, ue); //  A1(qx, iy) = sum_{ix} B1(qx, ix) * ue(iy, ix)  
    _contract(uq, B2, A1); //  uq(qy, qx) = sum_{iy} B2(qy, iy) * A1(qx, iy)
  }

  // ---------------------------------------------------------------------------
  // batched interpolation / integration, exploiting the tensor product
  // structure of the shape functions: only the 1D evaluations are tabulated.
  //
  // two 1D families are involved ("open" = gauss-legendre interpolants, p per
  // point; "closed" = gauss-lobatto interpolants, p+1 per point), assigned per
  // axis according to the dof ordering established in nodes():
  //
  //   x-directed:   open(x) closed(y),  dof slice shape {n(y), p(x)}
  //   y-directed: closed(x)   open(y),  dof slice shape {n(x), p(y)}
  //
  // table layouts (q = # of 1D quadrature points, n = p + 1):
  //   curls:           [B1 {q,p} open, B2 {q,n} closed, G2 {q,n} closed derivs, B1m {q,p} = -B1]
  //   weighted values: [B1t {p,q}, B2t {n,q}]                      (transposed, x weight)
  //   weighted curls:  [B1t {p,q}, B2t {n,q}, G2t {n,q}, B1mt {p,q}]

  nd::array< double, 1, memory::space::cpu > evaluate_shape_function_curls(nd::view<const double,2> xi) const {
    uint32_t n = p + 1;
    uint32_t q = xi.shape[0];
    nd::array<double, 1, memory::space::cpu> buffer({2 * q * (p + n)});
    for (uint32_t i = 0; i < q; i++) {
      GaussLegendreInterpolation(xi(i, 0), p, &buffer(p * i));
      GaussLobattoInterpolation(xi(i, 0), n, &buffer(q * p + n * i));
      GaussLobattoInterpolationDerivative(xi(i, 0), n, &buffer(q * (p + n) + n * i));
      for (uint32_t j = 0; j < p; j++) {
        buffer(q * (p + 2 * n) + p * i + j) = -buffer(p * i + j);
      }
    }
    return buffer;
  }

  void curl(nd::view<derivative_type> values_q, nd::view<const double, 1> values_e, nd::view<double> shape_fns, double * buffer) const {
    uint32_t n = p + 1;
    uint32_t q = 0; while (q * q < values_q.shape[0]) q++;

    nd::view<const double, 2> B1(&shape_fns[0], {q, p});
    nd::view<const double, 2> G2(&shape_fns[q * (p + n)], {q, n});
    nd::view<const double, 2> B1m(&shape_fns[q * (p + 2 * n)], {q, p});

    nd::view<const double, 2> ue_x(&values_e[0], {n, p});
    nd::view<const double, 2> ue_y(&values_e[0] + n * p, {n, p});

    double * cq = (double *)&values_q[0];
    nd::view<double, 2> out(cq, {q, q});           // (qy, qx)
    nd::view<double, 2> outT(cq, {q, q}, {1, q});  // (qx, qy)

    // curl(u) = dx(u_y) - dy(u_x)
    tp_apply(ue_y, B1, G2, outT, buffer, false);
    tp_apply(ue_x, B1m, G2, out, buffer, true);
  }

  nd::array< double, 1, memory::space::cpu > evaluate_weighted_shape_functions(nd::view<const double, 2> xi, nd::view<const double, 1> weights) const {
    uint32_t n = p + 1;
    uint32_t q = xi.shape[0];
    nd::array<double, 1, memory::space::cpu> buffer({q * (p + n)});
    double tmp[4];
    for (uint32_t i = 0; i < q; i++) {
      GaussLegendreInterpolation(xi(i, 0), p, tmp);
      for (uint32_t j = 0; j < p; j++) { buffer(j * q + i) = tmp[j] * weights(i); }
      GaussLobattoInterpolation(xi(i, 0), n, tmp);
      for (uint32_t j = 0; j < n; j++) { buffer(p * q + j * q + i) = tmp[j] * weights(i); }
    }
    return buffer;
  }

  void integrate_source(nd::view<double> residual_e, nd::view<const source_type> source_q, nd::view<double> shape_fn, double * buffer) const {
    uint32_t n = p + 1;
    uint32_t q = 0; while (q * q < source_q.shape[0]) q++;

    nd::view<const double, 2> B1t(&shape_fn[0], {p, q});
    nd::view<const double, 2> B2t(&shape_fn[p * q], {n, q});

    uint32_t s = source_q.stride[0] * 2;
    nd::view<const double, 2> f_x(nullptr, {q, q}, {s * q, s});   // (qy, qx)
    nd::view<const double, 2> f_yT(nullptr, {q, q}, {s, s * q});  // (qx, qy)
    f_x.values = &source_q(0)[0];
    f_yT.values = &source_q(0)[1];

    nd::view<double, 2> r_x(&residual_e[0], {n, p});
    nd::view<double, 2> r_y(&residual_e[0] + n * p, {n, p});

    tp_apply(f_x, B1t, B2t, r_x, buffer, false);
    tp_apply(f_yT, B1t, B2t, r_y, buffer, false);
  }

  nd::array< double, 1, memory::space::cpu > evaluate_weighted_shape_function_curls(nd::view<const double, 2> xi, nd::view<const double, 1> weights) const {
    uint32_t n = p + 1;
    uint32_t q = xi.shape[0];
    nd::array<double, 1, memory::space::cpu> buffer({2 * q * (p + n)});
    double tmp[4];
    for (uint32_t i = 0; i < q; i++) {
      GaussLegendreInterpolation(xi(i, 0), p, tmp);
      for (uint32_t j = 0; j < p; j++) {
        buffer(j * q + i) = tmp[j] * weights(i);
        buffer(q * (p + 2 * n) + j * q + i) = -tmp[j] * weights(i);
      }
      GaussLobattoInterpolation(xi(i, 0), n, tmp);
      for (uint32_t j = 0; j < n; j++) { buffer(p * q + j * q + i) = tmp[j] * weights(i); }
      GaussLobattoInterpolationDerivative(xi(i, 0), n, tmp);
      for (uint32_t j = 0; j < n; j++) { buffer(q * (p + n) + j * q + i) = tmp[j] * weights(i); }
    }
    return buffer;
  }

  void integrate_flux(nd::view<double> residual_e, nd::view<const flux_type> flux_q, nd::view<double> shape_fn_curl, double * buffer) const {
    uint32_t n = p + 1;
    uint32_t q = 0; while (q * q < flux_q.shape[0]) q++;

    nd::view<const double, 2> B1t(&shape_fn_curl[0], {p, q});
    nd::view<const double, 2> G2t(&shape_fn_curl[q * (p + n)], {n, q});
    nd::view<const double, 2> B1mt(&shape_fn_curl[q * (p + 2 * n)], {p, q});

    uint32_t s = flux_q.stride[0];
    nd::view<const double, 2> f(nullptr, {q, q}, {s * q, s});    // (qy, qx)
    nd::view<const double, 2> fT(nullptr, {q, q}, {s, s * q});   // (qx, qy)
    f.values = &flux_q(0)[0];
    fT.values = &flux_q(0)[0];

    nd::view<double, 2> r_x(&residual_e[0], {n, p});
    nd::view<double, 2> r_y(&residual_e[0] + n * p, {n, p});

    // curl(phi) for an x-directed dof is -dy(phi_x); for a y-directed one, +dx(phi_y)
    tp_apply(f, B1mt, G2t, r_x, buffer, false);
    tp_apply(fT, B1t, G2t, r_y, buffer, false);
  }

  #ifdef __CUDACC__
  __device__ void cuda_interpolate(nd::view<value_type> values_q, nd::view<double> values_e, nd::view<double> shape_fn, double * buffer) const {
    uint32_t n = p + 1;
    uint32_t q = 0;
    while (q * q < values_q.shape[0]) q++;

    nd::view<double, 2> B1(shape_fn.data(), {q, p});
    nd::view<double, 2> B2(B1.end(),        {q, n});

    threadid tid;
    tid.x = threadIdx.x % q;
    tid.y = threadIdx.x / q;
    tid.stride = q;

    nd::view<double, 2> ue_x(values_e.data(), {n, p});
    nd::view<double, 2> ue_y(values_e.data() + n * p, {n, p});

    double * uq = (double *)values_q.data();
    nd::view<double, 2> out((double *)uq, {q, q}, {2 * q, 2});      // x component, (qy, qx)
    nd::view<double, 2> outT((double *)uq + 1, {q, q}, {2, 2 * q}); // y component, (qx, qy)

    cuda_tp_apply(tid, ue_x, B1, B2, out, buffer);
    cuda_tp_apply(tid, ue_y, B1, B2, outT, buffer);
  }

  __device__ void cuda_curl(nd::view<derivative_type> values_q, nd::view<const double, 1> values_e, nd::view<double> shape_fns, double * buffer) const {
    uint32_t n = p + 1;
    uint32_t q = 0; while (q * q < values_q.shape[0]) q++;

    nd::view<double, 2> B1(&shape_fns[0], {q, p});
    nd::view<double, 2> G2(&shape_fns[q * (p + n)], {q, n});
    nd::view<double, 2> B1m(&shape_fns[q * (p + 2 * n)], {q, p});

    nd::view<double, 2> ue_x((double *)&values_e[0], {n, p});
    nd::view<double, 2> ue_y((double *)&values_e[0] + n * p, {n, p});

    double * cq = (double *)&values_q[0];
    nd::view<double, 2> out(cq, {q, q});
    nd::view<double, 2> outT(cq, {q, q}, {1, q});

    threadid tid;
    tid.x = threadIdx.x % q;
    tid.y = threadIdx.x / q;
    tid.stride = q;

    cuda_tp_apply(tid, ue_y, B1, G2, outT, buffer, false);
    cuda_tp_apply(tid, ue_x, B1m, G2, out, buffer, true);
  }

  __device__ void cuda_integrate_source(nd::view<double> residual_e, nd::view<const source_type> source_q, nd::view<double> shape_fn, double * buffer) const {
    uint32_t n = p + 1;
    uint32_t q = 0; while (q * q < source_q.shape[0]) q++;

    nd::view<double, 2> B1t(&shape_fn[0], {p, q});
    nd::view<double, 2> B2t(&shape_fn[p * q], {n, q});

    uint32_t s = source_q.stride[0] * 2;
    nd::view<double, 2> f_x(nullptr, {q, q}, {s * q, s});
    nd::view<double, 2> f_yT(nullptr, {q, q}, {s, s * q});
    f_x.values = (double *)&source_q(0)[0];
    f_yT.values = (double *)&source_q(0)[1];

    nd::view<double, 2> r_x(&residual_e[0], {n, p});
    nd::view<double, 2> r_y(&residual_e[0] + n * p, {n, p});

    threadid tid;
    tid.x = threadIdx.x % n;
    tid.y = threadIdx.x / n;
    tid.stride = n;

    cuda_tp_apply(tid, f_x, B1t, B2t, r_x, buffer, false);
    cuda_tp_apply(tid, f_yT, B1t, B2t, r_y, buffer, false);
  }

  __device__ void cuda_integrate_flux(nd::view<double> residual_e, nd::view<const flux_type> flux_q, nd::view<double> shape_fn_curl, double * buffer) const {
    uint32_t n = p + 1;
    uint32_t q = 0; while (q * q < flux_q.shape[0]) q++;

    nd::view<double, 2> B1t(&shape_fn_curl[0], {p, q});
    nd::view<double, 2> G2t(&shape_fn_curl[q * (p + n)], {n, q});
    nd::view<double, 2> B1mt(&shape_fn_curl[q * (p + 2 * n)], {p, q});

    uint32_t s = flux_q.stride[0];
    nd::view<double, 2> f(nullptr, {q, q}, {s * q, s});
    nd::view<double, 2> fT(nullptr, {q, q}, {s, s * q});
    f.values = (double *)&flux_q(0)[0];
    fT.values = (double *)&flux_q(0)[0];

    nd::view<double, 2> r_x(&residual_e[0], {n, p});
    nd::view<double, 2> r_y(&residual_e[0] + n * p, {n, p});

    threadid tid;
    tid.x = threadIdx.x % n;
    tid.y = threadIdx.x / n;
    tid.stride = n;

    cuda_tp_apply(tid, f, B1mt, G2t, r_x, buffer, false);
    cuda_tp_apply(tid, fT, B1t, G2t, r_y, buffer, false);
  }
  #endif

  uint32_t p;

};
// clang-format on

} // namespace femto
