#include "gtest/gtest.h"

#include "femto/mesh/io.hpp"

#include <cmath>

using namespace io;

TEST(stl, tri3_test) {

    double s = sqrt(1.0 / 2.0);

    Mesh mesh;

    //   2
    //  5 4
    // 0 3 1
    mesh.nodes = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
        {  s,   s, 0.0},
        {0.0,   s,   s},
        {  s, 0.0,   s}
    };

    mesh.elements = {
        {Element::Type::Tri3, {0, 3, 5}},
        {Element::Type::Tri3, {3, 1, 4}},
        {Element::Type::Tri3, {3, 4, 5}},
        {Element::Type::Tri3, {5, 4, 2}}
    };

    export_stl(mesh, "sphere_octant_tri3.stl");

}

TEST(stl, tri6_test) {

    double s = sqrt(1.0 / 2.0);

    Mesh mesh;

    //      2
    //     5 4
    //    0 3 1
    mesh.nodes = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
        {  s,   s, 0.0},
        {0.0,   s,   s},
        {  s, 0.0,   s}
    };

    mesh.elements = {
        {Element::Type::Tri6, {0, 1, 2, 3, 4, 5}},
    };

    export_stl(mesh, "sphere_octant_tri6.stl");

}

TEST(stl, quad4_test) {

    Mesh mesh;
    mesh.nodes = {{0.0, 0.0, 0.3}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.8}, {0.0, 1.0, 0.1}};
    mesh.elements = {{Element::Type::Quad4, {0, 1, 2, 3}}};
    export_stl(mesh, "quad4.stl");

}

TEST(stl, quad8_test) {

    Mesh mesh;
    mesh.nodes = {
        {0.0, 0.0, 0.3}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.8}, {0.0, 1.0, 0.1},
        {0.5, 0.0, 0.1}, {1.0, 0.5, 0.3}, {0.4, 1.0, 0.5}, {0.0, 0.4, 0.1}
    };
    mesh.elements = {{Element::Type::Quad8, {0, 1, 2, 3, 4, 5, 6, 7}}};
    export_stl(mesh, "quad8.stl");

}

TEST(stl, quad9_test) {

    Mesh mesh;
    mesh.nodes = {
        {0.0, 0.0, 0.3}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.8}, {0.0, 1.0, 0.1},
        {0.5, 0.0, 0.1}, {1.0, 0.5, 0.3}, {0.4, 1.0, 0.5}, {0.0, 0.4, 0.1}, {0.5, 0.4, 0.3}
    };
    mesh.elements = {{Element::Type::Quad9, {0, 1, 2, 3, 4, 5, 6, 7, 8}}};
    export_stl(mesh, "quad9.stl");

}