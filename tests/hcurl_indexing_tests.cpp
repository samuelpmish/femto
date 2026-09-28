#include <gtest/gtest.h>

#include <iostream>
#include <functional>

#include "femto/mesh.hpp"
#include "femto/field.hpp"

#include "femto/domain.hpp"

#include "misc/for_constexpr.hpp"
#include "forall.hpp"

#include "fm/types/matrix.hpp"
#include "fm/types/vec.hpp"

using namespace femto;

std::vector< Mesh<> > generate_permuted_meshes(Geometry geom) {

  if (geom == Geometry::Triangle) {
    constexpr uint32_t num_permutations = 3;
    uint32_t positive_permutations[num_permutations][3] = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}}; 
    uint32_t elements[2][3] = {{0, 1, 3}, {1, 2, 3}}; 
    double vertices[4][2] = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}};

    nd::array<double, 2, memory::space::cpu> nodes({4, 2});
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 2; j++) {
        nodes(i,j) = vertices[i][j];
      }
    }

    nd::array<uint32_t, 2, memory::space::cpu> tris({2, 3});
    nd::array<uint32_t, 2, memory::space::cpu> empty({0, 0});

    // the first element is fixed
    for (int j = 0; j < 3; j++) {
      tris(0,j) = elements[0][j];
    }

    std::vector< Mesh<> > meshes;

    for (int p = 0; p < num_permutations; p++) {
      // but the other element connectivity is permuted
      for (int j = 0; j < 3; j++) {
        tris(1,j) = elements[1][positive_permutations[p][j]];
      }

      meshes.push_back(Mesh<>::create_2D(nodes, 1, tris, empty));
    }

    return meshes;
  }

  if (geom == Geometry::Quadrilateral) {
    constexpr uint32_t num_permutations = 4;
    uint32_t positive_permutations[num_permutations][4] = {{0, 1, 2, 3}, {1, 2, 3, 0}, {2, 3, 0, 1}, {3, 0, 1, 2}}; 
    uint32_t elements[2][4] = {{0, 1, 4, 3}, {1, 2, 5, 4}}; 
    double vertices[6][2] = {{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {2.0, 1.0}};

    nd::array<double, 2, memory::space::cpu> nodes({6, 2});
    for (int i = 0; i < 6; i++) {
      for (int j = 0; j < 2; j++) {
        nodes(i,j) = vertices[i][j];
      }
    }

    nd::array<uint32_t, 2, memory::space::cpu> empty({0, 0});
    nd::array<uint32_t, 2, memory::space::cpu> quads({2, 4});

    // the first element is fixed
    for (int j = 0; j < 4; j++) {
      quads(0,j) = elements[0][j];
    }

    std::vector< Mesh<> > meshes;

    for (int p = 0; p < num_permutations; p++) {
      // but the other element connectivity is permuted
      for (int j = 0; j < 4; j++) {
        quads(1,j) = elements[1][positive_permutations[p][j]];
      }

      meshes.push_back(Mesh<>::create_2D(nodes, 1, empty, quads));
    }

    return meshes;
  }

  if (geom == Geometry::Tetrahedron) {
    constexpr uint32_t num_permutations = 12;
    uint32_t positive_permutations[num_permutations][4] = {
      {0, 1, 2, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {1, 0, 3, 2}, 
      {1, 2, 0, 3}, {1, 3, 2, 0}, {2, 0, 1, 3}, {2, 1, 3, 0}, 
      {2, 3, 0, 1}, {3, 0, 2, 1}, {3, 1, 0, 2}, {3, 2, 1, 0}
    }; 
    uint32_t elements[2][4] = {{0, 1, 2, 3}, {1, 2, 3, 4}}; 
    double vertices[5][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}};

    nd::array<double, 2, memory::space::cpu> nodes({5, 3});
    for (int i = 0; i < 5; i++) {
      for (int j = 0; j < 3; j++) {
        nodes(i,j) = vertices[i][j];
      }
    }

    nd::array<uint32_t, 2, memory::space::cpu> tets({2, 4});
    nd::array<uint32_t, 2, memory::space::cpu> empty({0, 0});

    // the first element is fixed
    for (int j = 0; j < 4; j++) {
      tets(0,j) = elements[0][j];
    }

    std::vector< Mesh<> > meshes;

    for (int p = 0; p < num_permutations; p++) {
      // but the other element connectivity is permuted
      for (int j = 0; j < 4; j++) {
        tets(1,j) = elements[1][positive_permutations[p][j]];
      }

      meshes.push_back(Mesh<>::create_3D(nodes, 1, tets, empty));
    }

    return meshes;
  }

  if (geom == Geometry::Hexahedron) {
    constexpr uint32_t num_permutations = 24;
    uint32_t positive_permutations[num_permutations][8] = {
      {0, 1, 2, 3, 4, 5, 6, 7}, {0, 3, 7, 4, 1, 2, 6, 5}, {0, 4, 5, 1, 3, 7, 6, 2}, 
      {1, 0, 4, 5, 2, 3, 7, 6}, {1, 2, 3, 0, 5, 6, 7, 4}, {1, 5, 6, 2, 0, 4, 7, 3}, 
      {2, 1, 5, 6, 3, 0, 4, 7}, {2, 3, 0, 1, 6, 7, 4, 5}, {2, 6, 7, 3, 1, 5, 4, 0}, 
      {3, 0, 1, 2, 7, 4, 5, 6}, {3, 2, 6, 7, 0, 1, 5, 4}, {3, 7, 4, 0, 2, 6, 5, 1}, 
      {4, 0, 3, 7, 5, 1, 2, 6}, {4, 5, 1, 0, 7, 6, 2, 3}, {4, 7, 6, 5, 0, 3, 2, 1}, 
      {5, 1, 0, 4, 6, 2, 3, 7}, {5, 4, 7, 6, 1, 0, 3, 2}, {5, 6, 2, 1, 4, 7, 3, 0}, 
      {6, 2, 1, 5, 7, 3, 0, 4}, {6, 5, 4, 7, 2, 1, 0, 3}, {6, 7, 3, 2, 5, 4, 0, 1}, 
      {7, 3, 2, 6, 4, 0, 1, 5}, {7, 4, 0, 3, 6, 5, 1, 2}, {7, 6, 5, 4, 3, 2, 1, 0}
    };

    double vertices[12][3] = {
      {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
      {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},
      {0, 0, 2}, {1, 0, 2}, {1, 1, 2}, {0, 1, 2}
    };

    uint32_t elements[2][8] = {
      {0, 1, 2, 3, 4, 5, 6, 7}, 
      {4, 5, 6, 7, 8, 9, 10, 11}
    }; 

    nd::array<double, 2, memory::space::cpu> nodes({12, 3});
    for (int i = 0; i < 12; i++) {
      for (int j = 0; j < 3; j++) {
        nodes(i,j) = vertices[i][j];
      }
    }

    nd::array<uint32_t, 2, memory::space::cpu> empty({0, 0});
    nd::array<uint32_t, 2, memory::space::cpu> hexes({2, 8});

    // the first element is fixed
    for (int j = 0; j < 8; j++) {
      hexes(0,j) = elements[0][j];
    }

    std::vector< Mesh<> > meshes;

    for (int p = 0; p < num_permutations; p++) {
      // but the other element connectivity is permuted
      for (int j = 0; j < 8; j++) {
        hexes(1,j) = elements[1][positive_permutations[p][j]];
      }

      meshes.push_back(Mesh<>::create_3D(nodes, 1, empty, hexes));
    }

    return meshes;
  }

  return {};

}

template < Geometry geom >
void HcurlIndexTest2D(uint32_t p) {

  for (auto mesh : generate_permuted_meshes(geom)) {

    FiniteElement<geom, Family::Hcurl> element{p};

    Field u = create_field<Family::Hcurl>(mesh, p);
    nd::array<double, 2, memory::space::cpu> x = nodes_for(u, mesh);
    nd::array<double, 2, memory::space::cpu> & v = mesh.X.data;

    nd::array<double, 2, memory::space::cpu> d = directions_for(u, mesh);
    nd::array<double, 1, memory::space::cpu> values({d.shape[0]});
    for (int i = 0; i < values.shape[0]; i++) {
      values[i] = dot(load<vec2>(d, i), vec2{1,2});
    }

    nd::array<uint32_t, 1, memory::space::cpu> ids{{element.num_nodes()}};
    nd::array<double, 1, memory::space::cpu> values_e{{element.num_nodes()}};
    nd::array<double, 2, memory::space::cpu> xi{{element.num_nodes(), 2}};
    nd::array<double, 2, memory::space::cpu> dir{{element.num_nodes(), 2}};
    element.nodes(xi);
    element.directions(dir);

    uint32_t a; // index of corner node along xi direction
    uint32_t b; // index of corner node along eta direction
    if (geom == Geometry::Triangle) {
      a = 1;
      b = 2;
    } else {
      a = 1;
      b = 3;
    }

    for (uint32_t i = 0; i < mesh[geom].shape[0]; i++) {

      element.indices(u.offsets, mesh[geom](i).data(), ids.data());

      for (uint32_t j = 0; j < element.num_nodes(); j++) {
        values_e[j] = values[ids[j]];
      }

      element.reorient(TransformationType::PhysicalToParent, mesh[geom](i).data(), values_e.data());

      //std::cout << i << ": " << std::endl;
      //for (uint32_t j = 0; j < mesh[geom].shape[1]; j++) {
      //  std::cout << "  " << mesh[geom](i, j) << std::endl;
      //}
      //std::cout << std::endl;

      vec2 o = load<vec2>(v, mesh[geom](i, 0).index);
      mat2 A = mat2{{
        load<vec2>(v, mesh[geom](i, a).index) - o, 
        load<vec2>(v, mesh[geom](i, b).index) - o
      }};

      for (uint32_t j = 0; j < element.num_nodes(); j++) {
        vec2 x_j = dot(load<vec2>(xi, j), A) + o;
        EXPECT_NEAR(norm(x_j - load<vec2>(x, ids[j])), 0.0, 1.0e-15);

        vec2 d_j = dot(load<vec2>(dir, j), A); 
        EXPECT_NEAR(dot(d_j, vec2{1,2}), values_e[j], 1.0e-15);
      }

      //element.reorient(TransformationType::ParentToPhysical, mesh[geom](i).data(), values_e.data());
      //for (int j = 0; j < element.num_nodes(); j++) {
      //  EXPECT_NEAR(values_e[j], values[ids[j]], 1.0e-15);
      //}

    }

  }

}

template < Geometry geom >
void HcurlIndexTest3D(uint32_t p) {

  for (auto mesh : generate_permuted_meshes(geom)) {

    FiniteElement<geom, Family::Hcurl> element{p};

    Field u = create_field<Family::Hcurl>(mesh, p);
    nd::array<double, 2, memory::space::cpu> x = nodes_for(u, mesh);
    nd::array<double, 2, memory::space::cpu> & v = mesh.X.data;

    nd::array<double, 2, memory::space::cpu> d = directions_for(u, mesh);
    nd::array<double, 1, memory::space::cpu> values({d.shape[0]});
    for (int i = 0; i < values.shape[0]; i++) {
      values[i] = dot(load<vec3>(d, i), vec3{1,2,3});
    }

    nd::array<uint32_t, 1, memory::space::cpu> ids{{element.num_nodes()}};
    nd::array<double, 1, memory::space::cpu> values_e{{element.num_nodes()}};
    nd::array<double, 2, memory::space::cpu> xi{{element.num_nodes(), 3}};
    nd::array<double, 2, memory::space::cpu> dir{{element.num_nodes(), 3}};
    element.nodes(xi);
    element.directions(dir);

    uint32_t a; // index of corner node along xi direction
    uint32_t b; // index of corner node along eta direction
    uint32_t c; // index of corner node along zeta direction
    if (geom == Geometry::Tetrahedron) {
      a = 1;
      b = 2;
      c = 3;
    } else {
      a = 1;
      b = 3;
      c = 4;
    }

    for (uint32_t i = 0; i < mesh[geom].shape[0]; i++) {

      element.indices(u.offsets, mesh[geom](i).data(), ids.data());

      for (uint32_t j = 0; j < element.num_nodes(); j++) {
        values_e[j] = values[ids[j]];
      }

      element.reorient(TransformationType::PhysicalToParent, mesh[geom](i).data(), values_e.data());

      //std::cout << i << ": " << std::endl;
      //for (uint32_t j = 0; j < mesh[geom].shape[1]; j++) {
      //  std::cout << "  " << mesh[geom](i, j) << std::endl;
      //}
      //std::cout << std::endl;

      vec3 o = load<vec3>(v, mesh[geom](i, 0).index);
      mat3 A = mat3{{
        load<vec3>(v, mesh[geom](i, a).index) - o, 
        load<vec3>(v, mesh[geom](i, b).index) - o,
        load<vec3>(v, mesh[geom](i, c).index) - o
      }};

      for (uint32_t j = 0; j < element.num_nodes(); j++) {
        vec3 xi_j = load<vec3>(xi, j);
        vec3 x_j = dot(xi_j, A) + o;
        EXPECT_NEAR(norm(x_j - load<vec3>(x, ids[j])), 0.0, 1.0e-15);

        vec3 d_j = load<vec3>(d, ids[j]); 
        vec3 dir_j = dot(load<vec3>(dir, j), A); 
        //std::cout << j << " " << xi_j << " " << dir_j << " " << d_j << std::endl;
        EXPECT_NEAR(dot(dir_j, vec3{1,2,3}), values_e[j], 1.0e-15);
      }

      //element.reorient(TransformationType::ParentToPhysical, mesh[geom](i).data(), values_e.data());
      //for (int j = 0; j < element.num_nodes(); j++) {
      //  EXPECT_NEAR(values_e[j], values[ids[j]], 1.0e-15);
      //}

    }

  }

}


TEST(permutation_test, hcurl_triangles_p1) { HcurlIndexTest2D<Geometry::Triangle>(1); }
TEST(permutation_test, hcurl_triangles_p2) { HcurlIndexTest2D<Geometry::Triangle>(2); }
TEST(permutation_test, hcurl_triangles_p3) { HcurlIndexTest2D<Geometry::Triangle>(3); }

TEST(permutation_test, hcurl_quadrilaterals_p1) { HcurlIndexTest2D<Geometry::Quadrilateral>(1); }
TEST(permutation_test, hcurl_quadrilaterals_p2) { HcurlIndexTest2D<Geometry::Quadrilateral>(2); }
TEST(permutation_test, hcurl_quadrilaterals_p3) { HcurlIndexTest2D<Geometry::Quadrilateral>(3); }
 
TEST(permutation_test, hcurl_tetrahedra_p1) { HcurlIndexTest3D<Geometry::Tetrahedron>(1); }
TEST(permutation_test, hcurl_tetrahedra_p2) { HcurlIndexTest3D<Geometry::Tetrahedron>(2); }
TEST(permutation_test, hcurl_tetrahedra_p3) { HcurlIndexTest3D<Geometry::Tetrahedron>(3); }

TEST(permutation_test, hcurl_hexahedra_p1) { HcurlIndexTest3D<Geometry::Hexahedron>(1); }
TEST(permutation_test, hcurl_hexahedra_p2) { HcurlIndexTest3D<Geometry::Hexahedron>(2); }
TEST(permutation_test, hcurl_hexahedra_p3) { HcurlIndexTest3D<Geometry::Hexahedron>(3); }
