#include "gtest/gtest.h"

#include "femto/mesh/io.hpp"

#include "misc/timer.hpp"

#include "common.hpp"

using namespace io;

void export_gmsh_single_element(Element::Type type, std::string prefix) {
    export_gmsh_v22(single_element_mesh(type), prefix + "_txt.msh", FileEncoding::ASCII);
    export_gmsh_v22(single_element_mesh(type), prefix + "_bin.msh", FileEncoding::Binary);
}

TEST(gmsh, export_tests) {
    export_gmsh_single_element(Element::Type::Tri3, "tri3");
    export_gmsh_single_element(Element::Type::Line2, "line2");
    export_gmsh_single_element(Element::Type::Line3, "line3");
    export_gmsh_single_element(Element::Type::Tri3, "tri3");
    export_gmsh_single_element(Element::Type::Tri6, "tri6");
    export_gmsh_single_element(Element::Type::Quad4, "quad4");
    export_gmsh_single_element(Element::Type::Quad8, "quad8");
    export_gmsh_single_element(Element::Type::Quad9, "quad9");
    export_gmsh_single_element(Element::Type::Tet4, "tet4");
    export_gmsh_single_element(Element::Type::Tet10, "tet10");
    export_gmsh_single_element(Element::Type::Pyr5, "pyr5");
    export_gmsh_single_element(Element::Type::Pyr13, "pyr13");
    export_gmsh_single_element(Element::Type::Pyr14, "pyr14");
    export_gmsh_single_element(Element::Type::Prism6, "prism6");
    export_gmsh_single_element(Element::Type::Prism15, "prism15");
    export_gmsh_single_element(Element::Type::Prism18, "prism18");
    export_gmsh_single_element(Element::Type::Hex8, "hex8");
    export_gmsh_single_element(Element::Type::Hex20, "hex20");
    export_gmsh_single_element(Element::Type::Hex27, "hex27");
}

TEST(gmsh, big_import) {

    femto::timer stopwatch;
    stopwatch.start();
    Mesh mesh = import_gmsh_v22(FEMTO_MESH_DIR"octane_fine.msh");
    stopwatch.stop();
    std::cout << stopwatch.elapsed() * 1000.0 << "ms ";

    stopwatch.start();
    export_gmsh_v22(mesh, "tmp.msh", FileEncoding::ASCII);
    stopwatch.stop();
    std::cout << stopwatch.elapsed() * 1000.0 << "ms ";

    stopwatch.start();
    export_gmsh_v22(mesh, "tmp_bin.msh", FileEncoding::Binary);
    stopwatch.stop();
    std::cout << stopwatch.elapsed() * 1000.0 << "ms ";

    stopwatch.start();
    export_vtk(mesh, "tmp.vtk", FileEncoding::Binary);
    stopwatch.stop();
    std::cout << stopwatch.elapsed() * 1000.0 << "ms ";

    stopwatch.start();
    export_vtu(mesh, "tmp.vtu");
    stopwatch.stop();
    std::cout << stopwatch.elapsed() * 1000.0 << "ms ";

}