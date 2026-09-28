#include <array>
#include <vector>
#include <random>
#include <iostream>

#include "Application.hpp"

#include "spheres.hpp"
#include "cylinders.hpp"
#include "triangles.hpp"

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_operation.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "femto/mesh.hpp"

#include "BVH.hpp"

//#include "gmsh_io.hpp"
//#include "interface.hpp"

#include <fstream>
#include <unordered_set>
#include <unordered_map>

using namespace femto;

using GSphere = Graphics::Sphere;
using GCylinder = Graphics::Cylinder;

inline float sign(float val) {
  return (0.0f < val) - (val < 0.0f);
}

inline float clamp(float val, float minval, float maxval) {
  return std::min(std::max(val, minval), maxval);
}

constexpr int show_angle_defect = 0;
constexpr int show_gauss_curv = 1;
constexpr int show_mean_curv = 2;
constexpr int show_dihedral_angle = 3;
constexpr int show_distance = 4;

constexpr double twopi = M_PI * 2.0;

double data_range[5][2] = {
  {-1, 1},
  {-25, 25},
  {-25, 25},
  {-2, 2},
  {0.0, 0.064}
};

namespace geometry {

float norm_squared(glm::vec3 x) { return dot(x,x); }

struct Triangle {

  float SDF(glm::vec3 p) const {
    glm::vec3 ba = vertices[1] - vertices[0]; 
    glm::vec3 cb = vertices[2] - vertices[1]; 
    glm::vec3 ac = vertices[0] - vertices[2]; 
    glm::vec3 pa = p - vertices[0];
    glm::vec3 pb = p - vertices[1];
    glm::vec3 pc = p - vertices[2];
    glm::vec3 n = cross(ba, ac);
  
    return sqrt(
      (sign(dot(cross(ba,n),pa)) +
       sign(dot(cross(cb,n),pb)) +
       sign(dot(cross(ac,n),pc))<2.0)
       ?
       std::min(std::min(
       norm_squared(ba*clamp(dot(ba,pa)/dot(ba,ba),0.0,1.0)-pa),
       norm_squared(cb*clamp(dot(cb,pb)/dot(cb,cb),0.0,1.0)-pb)),
       norm_squared(ac*clamp(dot(ac,pc)/dot(ac,ac),0.0,1.0)-pc))
       :
       (dot(n,pa)*dot(n,pa))/dot(n,n));
  }

  float area() {
    return 0.5f * glm::length(glm::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]));
  }

  glm::vec3 unit_normal() {
    return glm::normalize(glm::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]));
  }

  glm::vec3 interior_angles() {
    glm::vec3 e1 = glm::normalize(vertices[1] - vertices[0]);
    glm::vec3 e2 = glm::normalize(vertices[2] - vertices[1]);
    glm::vec3 e3 = glm::normalize(vertices[0] - vertices[2]);

    return {
      std::acos(-dot(e3, e1)),
      std::acos(-dot(e1, e2)),
      std::acos(-dot(e2, e3))
    };
  }

  glm::vec3 vertices[3];
};

using Tetrahedron = const std::array< glm::vec3, 4>;

float sqnorm(glm::vec3 x) { return dot(x, x); }

float volume(const Tetrahedron & tet) {
  auto e1 = tet[1] - tet[0];
  auto e2 = tet[2] - tet[0];
  auto e3 = tet[3] - tet[0];
  return dot(cross(e1, e2), e3) / 6.0;
}

float quality(const std::array< glm::vec3, 4> & tet) {
  float L_rms = sqrt((sqnorm(tet[1] - tet[0]) + 
                      sqnorm(tet[2] - tet[1]) + 
                      sqnorm(tet[0] - tet[2]) + 
                      sqnorm(tet[0] - tet[3]) + 
                      sqnorm(tet[1] - tet[3]) + 
                      sqnorm(tet[2] - tet[3])) / 6.0); 
  float V = volume(tet);
  return 6.0 * sqrt(2.0) *  V / (L_rms * L_rms * L_rms);
}

}

rgbcolor white{230, 230, 230, 255};
rgbcolor light_gray{170, 170, 170, 255};
rgbcolor dark_gray{70, 70, 70, 255};
rgbcolor blue{ 30,  30, 200, 255};
rgbcolor red{230,  30,  30, 255};
rgbcolor orange{240, 140,   0, 255};
rgbcolor yellow{240, 220,   0, 255};
rgbcolor purple{170,  20, 170, 255};

class MeshViewer : public Application {
 public:

  void precalculate_quantities() {

    int nvert = mesh.vert.shape[0];
    int nedge = mesh.edge.shape[0];

    int nbdrvert = bdr.vert.shape[0];
    int nbdredge = bdr.edge.shape[0];
    int nbdrtri = bdr.tri.shape[0];

    A.resize(nvert, 0.0);   // area (vertex)
    phi.resize(nedge, 0.0); // dihedral angle (edge)
    H.resize(nvert, 0.0);   // mean curvature (vertex)
    K.resize(nvert, 0.0);   // gaussian curvature (vertex)
    O.resize(nvert, twopi); // angle-defect (vertex)
    d.resize(nvert, 0);     // distance (vertex)
    n.resize(nvert, {});    // normals (vertex)

    auto & x = mesh.X.data;

    {
      glm::vec3 min{1e8, 1e8, 1e8}, max{-1e8, -1e8, -1e8};
      v.resize(nvert);
      for (int i = 0; i < mesh.vert.shape[0]; i++) {
        v[i] = glm::vec3{x(i, 0), x(i, 1), x(i, 2)} * 0.1f;

        min[0] = std::min(min[0], v[i][0]);
        min[1] = std::min(min[1], v[i][1]);
        min[2] = std::min(min[2], v[i][2]);

        max[0] = std::max(max[0], v[i][0]);
        max[1] = std::max(max[1], v[i][1]);
        max[2] = std::max(max[2], v[i][2]);
      }

      float scale = 0.5f * glm::length(max - min);
      glm::vec3 mid = 0.5f * (max + min);

      // rescale things to be roughly in the biunit cube
      for (int i = 0; i < mesh.vert.shape[0]; i++) {
        v[i] = (v[i] - mid) / scale;
      }
    }

    for (int i = 0; i < nbdrtri; i++) {
      int tri_id = bdr.tri[i];
      uint32_t v_ids[3] = {mesh.tri(tri_id, 0).index, mesh.tri(tri_id, 1).index, mesh.tri(tri_id, 2).index};

      geometry::Triangle tri{v[v_ids[0]], v[v_ids[1]], v[v_ids[2]]};

      float area = tri.area();
      glm::vec3 theta = tri.interior_angles();
      glm::vec3 unit_normal = tri.unit_normal();

      for (int j = 0; j < Triangle::num_vertices; j++) {
        O[v_ids[j]] -= theta[j];
        A[v_ids[j]] += area / 3.0;
        n[v_ids[j]] += theta[j] * unit_normal;
      }
    }

    for (int i = 0; i < nbdredge; i++) {
      int edge_id = bdr.edge[i];
      uint32_t v_ids[2] = {mesh.edge(edge_id, 0).index, mesh.edge(edge_id, 1).index};

      float L = glm::length(v[v_ids[1]] - v[v_ids[0]]);
      glm::vec3 e = glm::normalize(v[v_ids[1]] - v[v_ids[0]]);

      auto connected_tris = bdr_edge_to_tri_LUT[edge_id];
      geometry::Triangle tris [2] = {
        {v[mesh.tri(connected_tris[0], 0).index], v[mesh.tri(connected_tris[0], 1).index], v[mesh.tri(connected_tris[0], 2).index]},
        {v[mesh.tri(connected_tris[1], 0).index], v[mesh.tri(connected_tris[1], 1).index], v[mesh.tri(connected_tris[1], 2).index]}
      };

      glm::vec3 n[2] = {tris[0].unit_normal(), tris[1].unit_normal()};

      float dihedral_angle = atan2(dot(e, glm::cross(n[0], n[1])), dot(n[0], n[1]));

      phi[edge_id] = dihedral_angle;

      for (int j = 0; j < Edge::num_vertices; j++) {
        H[v_ids[j]] += 0.25 * L * dihedral_angle;
      }
    }

    for (int i = 0; i < nbdrvert; i++) {
      int v_id = bdr.vert[i];
      H[v_id] /= A[v_id];
      K[v_id] = O[v_id] / A[v_id];
      n[v_id] = normalize(n[v_id]);
    }

    std::vector< geometry::Triangle > tris(nbdrtri);
    std::vector< AABB<3> > boxes(nbdrtri);
    for (int i = 0; i < nbdrtri; i++) {
      int tri_id = bdr.tri[i];
      uint32_t v_ids[3] = {mesh.tri(tri_id, 0).index, mesh.tri(tri_id, 1).index, mesh.tri(tri_id, 2).index};

      float min[3] = {v[v_ids[0]][0], v[v_ids[0]][1], v[v_ids[0]][2]};
      float max[3] = {v[v_ids[0]][0], v[v_ids[0]][1], v[v_ids[0]][2]};
      for (int i = 1; i < 3; i++) {
        min[0] = std::min(min[0], v[v_ids[i]][0]);
        min[1] = std::min(min[1], v[v_ids[i]][1]);
        min[2] = std::min(min[2], v[v_ids[i]][2]);

        max[0] = std::max(max[0], v[v_ids[i]][0]);
        max[1] = std::max(max[1], v[v_ids[i]][1]);
        max[2] = std::max(max[2], v[v_ids[i]][2]);
      }

      tris[i] = {v[v_ids[0]], v[v_ids[1]], v[v_ids[2]]};
      boxes[i] = {{min[0], min[1], min[2]}, {max[0], max[1], max[2]}};
    }

    BVH<3> bvh(boxes);

    femto::timer stopwatch;
    stopwatch.start();
    for (int i = 0; i < nbdrvert; i++) {
      int v_id = bdr.vert[i];
      glm::vec3 x = v[v_id];
      glm::vec3 nhat = n[v_id];

      float r = 0.001f;
      bool early_exit = false;
      for (int j = 0; j < 6; j++) {
        float min_distance = r;

        glm::vec3 c = x + 1.01f * r * nhat;
        AABB<3> box{
          {c[0]-r, c[1]-r, c[2]-r}, 
          {c[0]+r, c[1]+r, c[2]+r}
        };

        bvh.query(box, [&](int j){
          min_distance = std::min(min_distance, tris[j].SDF(c)); 
        });
        
        if (min_distance < r) { 
          d[v_id] = min_distance;
          early_exit = true;
          break;
        } else {
          r *= 2;
        }
      }
      if (!early_exit) {
        d[v_id] = r/2;
      }
    }
    stopwatch.stop();
    std::cout << "BVH query time: " << stopwatch.elapsed() * 1000.0 << "ms" << std::endl;

  }

  void redraw() {

    spheres.clear();
    cylinders.clear();
    triangles.clear();

    spheres.set_color(dark_gray);
    cylinders.set_color(dark_gray);
    triangles.set_color(light_gray);

    for (auto i : bdr.tri) {
      auto tri = mesh.tri(i);
      triangles.append({v[tri[0].index], v[tri[1].index], v[tri[2].index]});
    }

    for (uint32_t i = 0; i < bdr.edge.size(); i++) {
      auto edge = mesh.edge(bdr.edge[i]);
      float t = phi[bdr.edge[i]];
      if (mode == show_dihedral_angle) {
        cylinders.set_color(blend(palettes::blue_to_red, (t - data_range[mode][0]) / (data_range[mode][1] - data_range[mode][0])));
      }
      cylinders.append(GCylinder{{{v[edge[0].index], 0.0005}, {v[edge[1].index], 0.0005}}});
    }

    cylinders.set_color(dark_gray);

    float r = (mode == show_dihedral_angle) ? 0.0005 : 0.002;
    for (auto i : bdr.vert) {
      if (mode != show_dihedral_angle) {
        float t;
        if (mode == show_angle_defect) { t = O[i]; }
        if (mode == show_gauss_curv) { t = K[i]; }
        if (mode == show_mean_curv) { t = H[i]; }
        if (mode == show_distance) { t = d[i]; }
        spheres.set_color(blend(palettes::blue_to_red, (t - data_range[mode][0]) / (data_range[mode][1] - data_range[mode][0]))); 
      }
      spheres.append(GSphere{v[i], 0.002});

      if (show_normals) {
        cylinders.append(GCylinder{{{v[i], 0.0005}, {v[i] + d[i] * n[i], 0.0005}}});
      }
    }

  }

  MeshViewer() : Application("Mesh<> Viewer") {

    //mesh = import_gmsh("/home/sam/Dropbox/rocket_league_models/octane_fine.msh");
    mesh = Mesh<>::load(FEMTO_MESH_DIR"octane_fine.msh");
    //mesh = Mesh<>::load(FEMTO_MESH_DIR"ball.json");

    bdr = boundary_of(mesh);

    std::cout << bdr.vert.size() << std::endl;
    std::cout << bdr.edge.size() << std::endl;
    std::cout << bdr.tri.size() << std::endl;

    bdr_edge_to_tri_LUT.resize(mesh.edge.size(), {-1, -1}); 
    for (int i = 0; i < bdr.tri.size(); i++) {
      int tri_id = bdr.tri[i];
      auto tri_edge_ids = mesh.tri(tri_id).data() + Triangle::edge_offset;

      for (int j = 0; j < Triangle::num_edges; j++) {
        auto & entry = bdr_edge_to_tri_LUT[tri_edge_ids[j].index];
        if (tri_edge_ids[j].sign() == Sign::Positive) {
          entry[0] = tri_id; 
        } else {
          entry[1] = tri_id; 
        }
      }
    }

    precalculate_quantities();

    show_normals = false;
    mode = show_angle_defect;
    light_intensity = 0.7f;

    fov = 1.0;
    camera_speed = 0.02;
    camera.lookAt(glm::vec3(1, 1, 1), glm::vec3(0, 0, 0));
    camera.perspective(fov, getWindowRatio(), 0.01f, 100.0f);

    redraw();

  }

 protected:

  virtual void loop() {

    // exit on window close button pressed
    if (glfwWindowShouldClose(getWindow()))
      exit();

    update_camera_position();

    // clear
    glClearColor(0.1f, 0.1f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    spheres.draw(camera);
    cylinders.draw(camera);
    triangles.draw(camera);

    // feed inputs to dear imgui, start new frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // render your GUI
    ImGui::Begin("Demo window");

    if (ImGui::DragFloat("fov", &fov, 0.01f, 0.05f, 1.5f)) {
      camera.perspective(fov, getWindowRatio(), 0.01f, 100.0f);
    }

    if (ImGui::RadioButton("angle_defect", &mode, 0)) { redraw(); }
    if (ImGui::RadioButton("gauss_curv", &mode, 1)) { redraw(); }
    if (ImGui::RadioButton("mean_curv", &mode, 2)) { redraw(); }
    if (ImGui::RadioButton("dihedral_angle", &mode, 3)) { redraw(); }
    if (ImGui::RadioButton("proximity", &mode, 4)) { redraw(); }

    if (ImGui::DragFloat("light intensity", &light_intensity, 0.01f, 0.0f, 1.0f)) { 
      glm::vec3 direction(0.721995, 0.618853, 0.309426);
      triangles.set_light(direction, light_intensity);
    }

    if (ImGui::Checkbox("show normals", &show_normals)) { redraw(); }

    ImGui::End();

    // Render dear imgui into screen
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

  }

 private:

  float light_intensity;

  Mesh<> mesh;
  SubMesh bdr;

  std::vector< glm::vec3 > v; // vertex coordinates
  std::vector< float > A;     // area (vertex)
  std::vector< float > phi;   // dihedral angle (edge)
  std::vector< float > H;     // mean curvature (vertex)
  std::vector< float > K;     // gaussian curvature (vertex)
  std::vector< float > O;     // angle-defect (vertex)
  std::vector< float > d;     // distance (vertex)
  std::vector< glm::vec3 > n; // normal (vertex)

  std::vector< std::array<int, 2> > bdr_edge_to_tri_LUT;

  int mode;
  bool show_normals;

  float fov;
  Graphics::Spheres spheres;
  Graphics::Cylinders cylinders;
  Graphics::Triangles triangles;
  std::vector< float > qualities;
};

int main(int argc, const char* argv[]) {
  MeshViewer app;
  app.run();
  return 0;
}
