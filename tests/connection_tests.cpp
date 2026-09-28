#include "femto/connection.hpp"

#include "gtest/gtest.h"

#include <iostream>

using namespace femto;

TEST(connections, index) {
  Connection c{18};

  EXPECT_EQ(c.index, 18);
  EXPECT_EQ(c.subindex(), 0);
  EXPECT_EQ(c.orientation(), 0);
  EXPECT_EQ(c.sign(), Sign::Positive);
  EXPECT_EQ(c.geometry(), Geometry::Vertex);

  c.index = 29;
  EXPECT_EQ(c.index, 29);
  EXPECT_EQ(c.subindex(), 0);
  EXPECT_EQ(c.orientation(), 0);
  EXPECT_EQ(c.sign(), Sign::Positive);
  EXPECT_EQ(c.geometry(), Geometry::Vertex);
}

TEST(connections, subindex) {
  Connection c{18};

  EXPECT_EQ(c.index, 18);
  EXPECT_EQ(c.subindex(), 0);
  EXPECT_EQ(c.orientation(), 0);
  EXPECT_EQ(c.sign(), Sign::Positive);
  EXPECT_EQ(c.geometry(), Geometry::Vertex);

  c.set_subindex(13);
  EXPECT_EQ(c.index, 18);
  EXPECT_EQ(c.subindex(), 13);
  EXPECT_EQ(c.orientation(), 0);
  EXPECT_EQ(c.sign(), Sign::Positive);
  EXPECT_EQ(c.geometry(), Geometry::Vertex);

  // subindex field is 4 bits wide, so anything bigger is misrepresented 
  c.set_subindex(9 + (1 << 5));
  EXPECT_EQ(c.index, 18);
  EXPECT_EQ(c.subindex(), 9);
  EXPECT_EQ(c.orientation(), 0);
  EXPECT_EQ(c.sign(), Sign::Positive);
  EXPECT_EQ(c.geometry(), Geometry::Vertex);
}

TEST(connections, orientation) {
  Connection c{18};

  EXPECT_EQ(c.index, 18);
  EXPECT_EQ(c.subindex(), 0);
  EXPECT_EQ(c.orientation(), 0);
  EXPECT_EQ(c.sign(), Sign::Positive);
  EXPECT_EQ(c.geometry(), Geometry::Vertex);

  c.set_orientation(3);
  EXPECT_EQ(c.index, 18);
  EXPECT_EQ(c.subindex(), 0);
  EXPECT_EQ(c.orientation(), 3);
  EXPECT_EQ(c.sign(), Sign::Positive);
  EXPECT_EQ(c.geometry(), Geometry::Vertex);

  // orientation field is 2 bits wide, so anything bigger is misrepresented 
  c.set_orientation(3 + (1 << 5));
  EXPECT_EQ(c.index, 18);
  EXPECT_EQ(c.subindex(), 0);
  EXPECT_EQ(c.orientation(), 3);
  EXPECT_EQ(c.sign(), Sign::Positive);
  EXPECT_EQ(c.geometry(), Geometry::Vertex);
}

TEST(connections, sign) {
  Connection c{18};

  EXPECT_EQ(c.index, 18);
  EXPECT_EQ(c.subindex(), 0);
  EXPECT_EQ(c.orientation(), 0);
  EXPECT_EQ(c.sign(), Sign::Positive);
  EXPECT_EQ(c.geometry(), Geometry::Vertex);

  c.set_sign(Sign::Negative);
  EXPECT_EQ(c.index, 18);
  EXPECT_EQ(c.subindex(), 0);
  EXPECT_EQ(c.orientation(), 0);
  EXPECT_EQ(c.sign(), Sign::Negative);
  EXPECT_EQ(c.geometry(), Geometry::Vertex);
}

TEST(connections, geometry) {
  Connection c{18};

  EXPECT_EQ(c.index, 18);
  EXPECT_EQ(c.subindex(), 0);
  EXPECT_EQ(c.orientation(), 0);
  EXPECT_EQ(c.sign(), Sign::Positive);
  EXPECT_EQ(c.geometry(), Geometry::Vertex);

  c.set_geometry(Geometry::Edge);
  EXPECT_EQ(c.index, 18);
  EXPECT_EQ(c.subindex(), 0);
  EXPECT_EQ(c.orientation(), 0);
  EXPECT_EQ(c.sign(), Sign::Positive);
  EXPECT_EQ(c.geometry(), Geometry::Edge);

  c.set_geometry(Geometry::Hexahedron);
  EXPECT_EQ(c.index, 18);
  EXPECT_EQ(c.subindex(), 0);
  EXPECT_EQ(c.orientation(), 0);
  EXPECT_EQ(c.sign(), Sign::Positive);
  EXPECT_EQ(c.geometry(), Geometry::Hexahedron);
}

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//   official   |                                                             //
//   ordering   |                                                             //
//              |           |                                                 //
//   a-----b    |  a-----b  |  b-----a                                        //
//              |           |                                                 //
//    sign      |     +     |     -                                           //
//  orientation |     0     |     1                                           //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////
TEST(connections, determine_sign_and_orientation_edge) {
  std::array< uint64_t, 2 > ids = {17, 93};
  Connection c1(42, ids, std::array{ids[0], ids[1]});
  EXPECT_EQ(c1.index, 42);
  EXPECT_EQ(c1.subindex(), 0);
  EXPECT_EQ(c1.orientation(), 0);
  EXPECT_EQ(c1.sign(), Sign::Positive);
  EXPECT_EQ(c1.geometry(), Geometry::Edge);

  Connection c2(42, ids, std::array{ids[1], ids[0]});
  EXPECT_EQ(c2.index, 42);
  EXPECT_EQ(c2.subindex(), 0);
  EXPECT_EQ(c2.orientation(), 0);
  EXPECT_EQ(c2.sign(), Sign::Negative);
  EXPECT_EQ(c2.geometry(), Geometry::Edge);
}

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
TEST(connections, determine_sign_and_orientation_tri) {
  std::array< uint64_t, 3 > ids = {17, 93, 46};
  Connection c1(42, ids, std::array{ids[0], ids[1], ids[2]});
  EXPECT_EQ(c1.index, 42);
  EXPECT_EQ(c1.subindex(), 0);
  EXPECT_EQ(c1.orientation(), 0);
  EXPECT_EQ(c1.sign(), Sign::Positive);
  EXPECT_EQ(c1.geometry(), Geometry::Triangle);

  Connection c2(42, ids, std::array{ids[1], ids[2], ids[0]});
  EXPECT_EQ(c2.index, 42);
  EXPECT_EQ(c2.subindex(), 0);
  EXPECT_EQ(c2.orientation(), 1);
  EXPECT_EQ(c2.sign(), Sign::Positive);
  EXPECT_EQ(c2.geometry(), Geometry::Triangle);

  Connection c3(42, ids, std::array{ids[2], ids[0], ids[1]});
  EXPECT_EQ(c3.index, 42);
  EXPECT_EQ(c3.subindex(), 0);
  EXPECT_EQ(c3.orientation(), 2);
  EXPECT_EQ(c3.sign(), Sign::Positive);
  EXPECT_EQ(c3.geometry(), Geometry::Triangle);



  Connection c4(42, ids, std::array{ids[0], ids[2], ids[1]});
  EXPECT_EQ(c4.index, 42);
  EXPECT_EQ(c4.subindex(), 0);
  EXPECT_EQ(c4.orientation(), 0);
  EXPECT_EQ(c4.sign(), Sign::Negative);
  EXPECT_EQ(c4.geometry(), Geometry::Triangle);

  Connection c5(42, ids, std::array{ids[1], ids[0], ids[2]});
  EXPECT_EQ(c5.index, 42);
  EXPECT_EQ(c5.subindex(), 0);
  EXPECT_EQ(c5.orientation(), 1);
  EXPECT_EQ(c5.sign(), Sign::Negative);
  EXPECT_EQ(c5.geometry(), Geometry::Triangle);

  Connection c6(42, ids, std::array{ids[2], ids[1], ids[0]});
  EXPECT_EQ(c6.index, 42);
  EXPECT_EQ(c6.subindex(), 0);
  EXPECT_EQ(c6.orientation(), 2);
  EXPECT_EQ(c6.sign(), Sign::Negative);
  EXPECT_EQ(c6.geometry(), Geometry::Triangle);
}

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
TEST(connections, determine_sign_and_orientation_quad) {
  std::array< uint64_t, 4 > ids = {17, 93, 46, 28};
  Connection c1(42, ids, std::array{ids[0], ids[1], ids[2], ids[3]});
  EXPECT_EQ(c1.index, 42);
  EXPECT_EQ(c1.subindex(), 0);
  EXPECT_EQ(c1.orientation(), 0);
  EXPECT_EQ(c1.sign(), Sign::Positive);
  EXPECT_EQ(c1.geometry(), Geometry::Quadrilateral);

  Connection c2(42, ids, std::array{ids[1], ids[2], ids[3], ids[0]});
  EXPECT_EQ(c2.index, 42);
  EXPECT_EQ(c2.subindex(), 0);
  EXPECT_EQ(c2.orientation(), 1);
  EXPECT_EQ(c2.sign(), Sign::Positive);
  EXPECT_EQ(c2.geometry(), Geometry::Quadrilateral);

  Connection c3(42, ids, std::array{ids[2], ids[3], ids[0], ids[1]});
  EXPECT_EQ(c3.index, 42);
  EXPECT_EQ(c3.subindex(), 0);
  EXPECT_EQ(c3.orientation(), 2);
  EXPECT_EQ(c3.sign(), Sign::Positive);
  EXPECT_EQ(c3.geometry(), Geometry::Quadrilateral);

  Connection c4(42, ids, std::array{ids[3], ids[0], ids[1], ids[2]});
  EXPECT_EQ(c4.index, 42);
  EXPECT_EQ(c4.subindex(), 0);
  EXPECT_EQ(c4.orientation(), 3);
  EXPECT_EQ(c4.sign(), Sign::Positive);
  EXPECT_EQ(c4.geometry(), Geometry::Quadrilateral);



  Connection c5(42, ids, std::array{ids[0], ids[3], ids[2], ids[1]});
  EXPECT_EQ(c5.index, 42);
  EXPECT_EQ(c5.subindex(), 0);
  EXPECT_EQ(c5.orientation(), 0);
  EXPECT_EQ(c5.sign(), Sign::Negative);
  EXPECT_EQ(c5.geometry(), Geometry::Quadrilateral);

  Connection c6(42, ids, std::array{ids[1], ids[0], ids[3], ids[2]});
  EXPECT_EQ(c6.index, 42);
  EXPECT_EQ(c6.subindex(), 0);
  EXPECT_EQ(c6.orientation(), 1);
  EXPECT_EQ(c6.sign(), Sign::Negative);
  EXPECT_EQ(c6.geometry(), Geometry::Quadrilateral);

  Connection c7(42, ids, std::array{ids[2], ids[1], ids[0], ids[3]});
  EXPECT_EQ(c7.index, 42);
  EXPECT_EQ(c7.subindex(), 0);
  EXPECT_EQ(c7.orientation(), 2);
  EXPECT_EQ(c7.sign(), Sign::Negative);
  EXPECT_EQ(c7.geometry(), Geometry::Quadrilateral);

  Connection c8(42, ids, std::array{ids[3], ids[2], ids[1], ids[0]});
  EXPECT_EQ(c8.index, 42);
  EXPECT_EQ(c8.subindex(), 0);
  EXPECT_EQ(c8.orientation(), 3);
  EXPECT_EQ(c8.sign(), Sign::Negative);
  EXPECT_EQ(c8.geometry(), Geometry::Quadrilateral);
}