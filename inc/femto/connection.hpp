#pragma once

#include <array>
#include <cinttypes>
#include <iostream>

#include "fm/macros.hpp"
#include "femto/geometry.hpp"

///////////////////////////
// orientation reference //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//   official   |                                                             //
//   ordering   |                                                             //
//              |           |                                                 //
//   a-----b    |  a-----b  |  b-----a                                        //
//              |           |                                                 //
//    sign      |     +     |     -                                           //
//  orientation |     0     |     0                                           //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//   official   |                                                             //
//   ordering   |                                                             //
//              |                              |                              //
//      c       |     c         a         b    |    b         c         a     //
//     / \      |    / \       / \       / \   |   / \       / \       / \    //
//    /   \     |   /   \     /   \     /   \  |  /   \     /   \     /   \   //
//   a-----b    |  a-----b   b-----c   c-----a | a-----c   b-----a   c-----b  // 
//              |                              |                              //
//    sign      |     +         +         +    |    -         -         -     //
//  orientation |     0         1         2    |    0         1         2     //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////
//                                                                            // 
//   official   |                                                             //
//   ordering   |                                                             //
//              |                              |                              //
//    d---c     |  d---c  a---d  b---a  c---b  |  b---c  c---d  d---a  a---b  //
//    |   |     |  |   |  |   |  |   |  |   |  |  |   |  |   |  |   |  |   |  //
//    a---b     |  a---b  b---c  c---d  d---a  |  a---d  b---a  c---b  d---c  //
//              |                              |                              //
//    sign      |    +      +      +      +    |    -      -      -      -    //
//  orientation |    0      1      2      3    |    0      1      2      3    //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// note: sign and orientation fields are only meaningful for Geometries that  //
//       are shared between elements (edges in 2D, edges/tris/quads in 3D)    //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

namespace femto {

enum class Sign : uint8_t {Positive, Negative};

inline Sign operator!(Sign s) {
  return (s == Sign::Positive) ? Sign::Negative : Sign::Positive;
}

struct Connection {

private:
  static constexpr uint32_t subindex_mask    = 0x0000'000F;
  static constexpr uint32_t orientation_mask = 0x0000'0030;
  static constexpr uint32_t sign_mask        = 0x0000'0040;
  static constexpr uint32_t geometry_mask    = 0x0000'0380;
  static constexpr uint32_t unspecified_mask = 0xFFFF'FB00;

  static constexpr uint32_t subindex_shift    =  0;
  static constexpr uint32_t orientation_shift =  4;
  static constexpr uint32_t sign_shift        =  6;
  static constexpr uint32_t geometry_shift    =  7;
  static constexpr uint32_t unspecified_shift = 10;

public:

  //  4-bits sub-index (for upward connectivity)
  //  2-bits orientation (only used for face elements in 3D)
  //  1-bit  sign
  //  3-bits geometry
  //  6-bits currently unspecified
  uint32_t metadata;

  uint32_t index;

  __host__ __device__ constexpr Connection() : index{}, metadata{} {}

  __host__ __device__ explicit constexpr Connection(uint32_t id) : index{id}, metadata{} {}
  
  template< typename T > 
  __host__ __device__ constexpr Connection(uint32_t id, 
             const std::array< T, 2 > & official_ordering,
             const std::array< T, 2 > & alternate_ordering) { 
    index = id;
    set_subindex(0);
    set_geometry(Geometry::Edge);
    set_sign((alternate_ordering[0] == official_ordering[0]) ? Sign::Positive : Sign::Negative);
    set_orientation(0);
  }

  template< typename T > 
  __host__ __device__ constexpr Connection(uint32_t id, 
             const std::array< T, 3 > & official_ordering,
             const std::array< T, 3 > & alternate_ordering) { 
    index = id;
    set_subindex(0);
    set_geometry(Geometry::Triangle);
    for (uint8_t i = 0; i < 3; i++) {
      if (alternate_ordering[0] == official_ordering[i]) {
        set_orientation(i);
        set_sign((alternate_ordering[1] == official_ordering[(i+1)%3]) ? Sign::Positive : Sign::Negative);
        break;
      }
    }
  }

  template< typename T > 
  __host__ __device__ constexpr Connection(uint32_t id, 
             const std::array< T, 4 > & official_ordering,
             const std::array< T, 4 > & alternate_ordering) { 
    index = id;
    set_subindex(0);
    set_geometry(Geometry::Quadrilateral);
    for (uint8_t i = 0; i < 4; i++) {
      if (alternate_ordering[0] == official_ordering[i]) {
        set_orientation(i);
        set_sign((alternate_ordering[1] == official_ordering[(i+1)%4]) ? Sign::Positive : Sign::Negative);
        break;
      }
    }
  }

  __host__ __device__ constexpr uint8_t subindex() const { 
    return (metadata & subindex_mask) >> subindex_shift; 
  }

  __host__ __device__ constexpr void set_subindex(uint8_t subindex) { 
    metadata = (metadata & ~subindex_mask) | (subindex_mask & (uint32_t(subindex) << subindex_shift)); 
  }

  __host__ __device__ constexpr uint8_t orientation() const { 
    return (metadata & orientation_mask) >> orientation_shift; 
  }

  __host__ __device__ constexpr void set_orientation(uint8_t o) { 
    metadata = (metadata & ~orientation_mask) | (orientation_mask & (uint32_t(o) << orientation_shift)); 
  }

  __host__ __device__ constexpr Sign sign() const { 
    return (metadata & sign_mask) ? Sign::Negative : Sign::Positive; 
  }

  __host__ __device__ void set_sign(Sign s) { 
    metadata = (metadata & ~sign_mask) | ((uint32_t(s == Sign::Negative) << sign_shift)); 
  }

  __host__ __device__ constexpr Geometry geometry() const { 
    return Geometry((metadata & geometry_mask) >> geometry_shift); 
  }

  __host__ __device__ constexpr void set_geometry(Geometry g) { 
    metadata = (metadata & ~geometry_mask) | ((uint32_t(g) << geometry_shift)); 
  }

  __host__ __device__ constexpr bool operator==(const Connection & other) const {
    return (index == other.index) && (metadata == other.metadata);
  }

};

inline std::ostream& operator<<(std::ostream & out, Connection c) {
  out << "{";
  out << "index: " << c.index << ", ";
  out << "subindex: " << int(c.subindex()) << ", ";
  out << "orientation: " << int(c.orientation()) << ", ";
  out << "sign: " << ((c.sign() == Sign::Positive) ? "+, " : "-, ");
  out << "geometry: " << to_string(c.geometry());
  out << "}";
  out << std::flush;
  return out;
}

__host__ __device__ constexpr bool flip(Connection c) { return (c.sign() == Sign::Negative); }

} // namespace femto