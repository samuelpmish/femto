#include "gtest/gtest.h"

#include "femto/mesh/io.hpp"

#include <cmath>

using namespace io;

// both fixtures are the same 20-facet icosahedron surface, one written in each
// of the two encodings import_stl() has to recognize.  import_stl() does not
// weld vertices, so every facet contributes three nodes of its own.
constexpr size_t num_facets = 20;

TEST(stl, import_ascii) {
    Mesh mesh = import_stl(FEMTO_MESH_DIR"icosahedron_ascii.stl");
    EXPECT_EQ(mesh.elements.size(), num_facets);
    EXPECT_EQ(mesh.nodes.size(), 3 * num_facets);
    export_stl(mesh, "icosahedron_from_ascii.stl");
}

TEST(stl, import_binary) {
    Mesh mesh = import_stl(FEMTO_MESH_DIR"icosahedron_binary.stl");
    EXPECT_EQ(mesh.elements.size(), num_facets);
    EXPECT_EQ(mesh.nodes.size(), 3 * num_facets);
    export_stl(mesh, "icosahedron_from_binary.stl");
}

// the two encodings describe the same surface, so they have to agree to within
// the float32 precision the binary format stores its coordinates in
TEST(stl, encodings_agree) {
    Mesh from_ascii = import_stl(FEMTO_MESH_DIR"icosahedron_ascii.stl");
    Mesh from_binary = import_stl(FEMTO_MESH_DIR"icosahedron_binary.stl");

    ASSERT_EQ(from_ascii.nodes.size(), from_binary.nodes.size());
    for (size_t i = 0; i < from_ascii.nodes.size(); i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_NEAR(from_ascii.nodes[i][j], from_binary.nodes[i][j], 1.0e-6) << "node " << i;
        }
    }
}
