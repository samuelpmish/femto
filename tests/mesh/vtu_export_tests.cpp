#include "gtest/gtest.h"

#include "femto/mesh/io.hpp"

#include "misc/timer.hpp"

#include "common.hpp"

using namespace io;

void export_vtu_single_element(Element::Type type, std::string prefix) {
    export_vtu(single_element_mesh(type), prefix + ".vtu");
}

TEST(vtu, export_tests) {
    export_vtu_single_element(Element::Type::Tri3, "tri3");
    export_vtu_single_element(Element::Type::Tri3, "tri3");
    export_vtu_single_element(Element::Type::Tri6, "tri6");
    export_vtu_single_element(Element::Type::Quad4, "quad4");
    export_vtu_single_element(Element::Type::Quad8, "quad8");
    export_vtu_single_element(Element::Type::Quad9, "quad9");
    export_vtu_single_element(Element::Type::Tet4, "tet4");
    export_vtu_single_element(Element::Type::Tet10, "tet10");
    export_vtu_single_element(Element::Type::Pyr5, "pyr5");
    export_vtu_single_element(Element::Type::Pyr13, "pyr13");
    export_vtu_single_element(Element::Type::Prism6, "prism6");
    export_vtu_single_element(Element::Type::Prism15, "prism15");
    export_vtu_single_element(Element::Type::Prism18, "prism18");
    export_vtu_single_element(Element::Type::Hex8, "hex8");
    export_vtu_single_element(Element::Type::Hex20, "hex20");
    export_vtu_single_element(Element::Type::Hex27, "hex27");

    // 14-node pyramids seem to not be supported by vtu (?)
    // export_vtu_single_element(Element::Type::Pyr14, "pyr14");
}

inline std::vector<int> range(int i, int j) {
    std::vector<int> values;
    for (; i < j; ++i) { values.push_back(i); }
    return values;
}

// Create a mesh large enough that more than one 4MB block is compressed. For simplicity, elements have no connectivity
// with other elements, so connectivity data and node location data have equal size for default int32 and float types.
TEST(vtu, multi_block_compression) {

    struct TestCase {
        Element::Type type;
        int num_elements; // choose such that connectivity and node location data are > 4MB
        std::string name;
    };
    std::vector<TestCase> TEST_SET { 
        {Element::Type::Tet4, 10000, "tet4_multiblock"},
        {Element::Type::Tet10, 125000, "tet10_multiblock"} 
    };

    for (const TestCase &tc : TEST_SET) {

        const int n = nodes_per_elem(tc.type);
        std::vector<fm::vec3> node_locations(n * tc.num_elements);
        std::vector<Element> element_definitions(tc.num_elements);

        for (int i = 0; i < tc.num_elements; ++i) {
            for (int j = 0; j < n; ++j) {
                node_locations[n*i+j] = tetrahedron_nodes[j]; // NOTE hard-coding to only tet elements
            }
            element_definitions[i] = Element{tc.type, range(n*i, n*(i+1))};
        }

        femto::timer stopwatch;
        stopwatch.start();
        export_vtu(Mesh{node_locations, element_definitions}, tc.name+".vtu");
        stopwatch.stop();
        std::cout << stopwatch.elapsed() << std::endl;
    }
}
