#include "gtest/gtest.h"

#include "femto/mesh/io.hpp"

#include "common.hpp"

using namespace io;

void export_vtk_single_element(Element::Type type, std::string prefix) {
    export_vtk(single_element_mesh(type), prefix + "_txt.vtk", FileEncoding::ASCII);
    export_vtk(single_element_mesh(type), prefix + "_bin.vtk", FileEncoding::Binary);
}

TEST(vtk, export_tests) {
    export_vtk_single_element(Element::Type::Tri3, "tri3");
    export_vtk_single_element(Element::Type::Line2, "line2");
    export_vtk_single_element(Element::Type::Line3, "line3");
    export_vtk_single_element(Element::Type::Tri3, "tri3");
    export_vtk_single_element(Element::Type::Tri6, "tri6");
    export_vtk_single_element(Element::Type::Quad4, "quad4");
    export_vtk_single_element(Element::Type::Quad8, "quad8");
    export_vtk_single_element(Element::Type::Quad9, "quad9");
    export_vtk_single_element(Element::Type::Tet4, "tet4");
    export_vtk_single_element(Element::Type::Tet10, "tet10");
    export_vtk_single_element(Element::Type::Pyr5, "pyr5");
    export_vtk_single_element(Element::Type::Pyr13, "pyr13");
    export_vtk_single_element(Element::Type::Prism6, "prism6");
    export_vtk_single_element(Element::Type::Prism15, "prism15");
    export_vtk_single_element(Element::Type::Prism18, "prism18");
    export_vtk_single_element(Element::Type::Hex8, "hex8");
    export_vtk_single_element(Element::Type::Hex20, "hex20");
    export_vtk_single_element(Element::Type::Hex27, "hex27");

    // 14-node pyramids seem to not be supported by vtk (?)
    // export_vtk_single_element(Element::Type::Pyr14, "pyr14");
}