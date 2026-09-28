// 2D acoustics on a disk: why unbounded domains need absorbing boundaries.
// A pressure pulse set off inside the unit disk should spread out and leave,
// as it does in the analytic solution of the wave equation on the whole
// plane.  A finite element mesh has to stop somewhere, and what happens at
// its rim is the point of the example: a plain rim (a hard wall, the natural
// boundary condition) echoes the pulse straight back; the first-order
// absorbing boundary condition of Bayliss and Turkel lets the parts arriving
// head-on leave and reflects the rest; a ring of perfectly matched layer
// elements around the rim absorbs nearly everything.
//
// P1 triangles or Q1 quadrilaterals from Mesh::disk, the PML ring extruded
// from its rim.  Lumped mass, SSP-RK3 in time.  The PML is ScalarPML<2> of
// materials/pml.hpp with the radial layer, the Grote-Sim auxiliary system of
// docs/absorbing_boundaries, with one auxiliary vector per quadrature
// point in the (radial, tangential) frame.  The analytic field is
// the Hankel transform of the Gaussian pulse, a sum of Bessel modes with
// cos(c k t) time dependence, tabulated in radius per pulse.  Clicks add
// pulses; the same pulses drive both fields.

#include "example_common.hpp"

#include "femto/field.hpp"
#include "forall.hpp"
#include "materials/pml.hpp"

#include <cmath>
#include <iostream>

using namespace femto;

// J0(x) to 1e-8 (Numerical Recipes' rational fits; libc++ has no cyl_bessel_j)
static double bessel_j0(double x) {
  double ax = std::fabs(x);
  if (ax < 8.0) {
    double y = x * x;
    double n = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))));
    double d = 57568490411.0 + y * (1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y * 1.0))));
    return n / d;
  }
  double z = 8.0 / ax, y = z * z, xx = ax - 0.785398164;
  double p = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
  double q = -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934935152e-7)));
  return std::sqrt(0.636619772 / ax) * (std::cos(xx) * p - z * std::sin(xx) * q);
}

struct WaveDisk {

  enum Boundary : int { WALL = 0, ABC = 1, PML = 2 };

  struct Params {
    int element_type = 0;          // 0 triangles, 1 quadrilaterals
    int boundary = 2;              // Boundary
    int resolution = 5;            // element size 2^-resolution
    double c = 1.0;                // wave speed
    double cfl = 0.25;             // of the smallest element height (SSP-RK3 with lumped P1 triangles allows about 0.35)
    int pml_layers = 8;            // elements across the layer
    double pml_decay = 1.4;        // alpha_max * layer element size / c: the decay per element at the outer edge
  };

  // a pulse: u += exp(-|x - x0|^2 / w^2) at t0, and the analytic field it
  // radiates, tabulated in radius as a sum of Bessel modes
  struct Pulse {
    vec2 x0;
    double t0, width;
    static constexpr int NR = 600, NK = 1000;
    static constexpr double R_MAX = 3.0;
    std::vector<double> k, modes;   // modes(i, j) = ghat(k_j) J0(k_j r_i) k_j dk
    std::vector<double> table;      // u(r_i) at the last time asked for

    Pulse(vec2 x, double t, double w) : x0(x), t0(t), width(w), k(NK), modes(size_t(NR) * NK), table(NR) {
      double k_max = 7.0 / w, dk = k_max / NK;
      for (int j = 0; j < NK; j++) { k[j] = (j + 0.5) * dk; }
      for (int i = 0; i < NR; i++) {
        double r = R_MAX * i / (NR - 1);
        for (int j = 0; j < NK; j++) {
          double ghat = 0.5 * w * w * std::exp(-0.25 * k[j] * k[j] * w * w);
          modes[size_t(i) * NK + j] = ghat * bessel_j0(k[j] * r) * k[j] * dk;
        }
      }
    }

    void tabulate(double t, double c) {
      double tau = std::max(0.0, t - t0);
      std::vector<double> cs(NK);
      for (int j = 0; j < NK; j++) { cs[j] = std::cos(c * k[j] * tau); }
      for (int i = 0; i < NR; i++) {
        double sum = 0;
        const double * m = &modes[size_t(i) * NK];
        for (int j = 0; j < NK; j++) { sum += m[j] * cs[j]; }
        table[i] = sum;
      }
    }

    double at(vec2 x) const {
      double s = norm(x - x0) / R_MAX * (NR - 1);
      int i = std::min(NR - 2, int(s));
      double f = std::min(1.0, s - i);
      return (1 - f) * table[i] + f * table[i + 1];
    }
  };

  Params params;
  double dt = 0, t = 0, layer_thickness = 0, alpha_max = 0;
  std::vector<int> region;                  // per cell: 0 physical, 1 PML
  Mesh<> mesh;
  Field<Family::H1> u_field;
  BasisFunction<Family::H1> phi;
  Domain<> domain;
  uint32_t nn = 0, nq = 0;
  nd::cpu_array<double, 2> Ml, w_rim;       // (n, 1): lumped mass, and lumped boundary measure at the rim
  nd::cpu_array<double, 3> frame_q;         // (q, 2, 2): rows, the radial and tangential unit vectors
  nd::cpu_array<double, 2> alpha_q;         // (q, 2): the profile (alpha_r, alpha_t)
  nd::cpu_array<double, 2> alpha_n;         // (n, 2): the profile at the nodes (lumped damping)
  nd::cpu_array<double, 2> u, v, aux;     // the state: nodal u, v (n, 1) and aux (q, 2)
  nd::cpu_array<double, 3> du_q;            // rate()'s work arrays, allocated once
  nd::cpu_array<double, 2> flux_q;
  Residual<Family::H1> r;
  nd::cpu_array<double, 2> a, auxdot, u1, v1, aux1;   // step()'s stage state and rates
  std::vector<Pulse> pulses;
  std::vector<double> exact_cache;
  double exact_time = -1;

  vec2 X(uint32_t i) const { return load<vec2>(mesh.X.data, i); }

  // the nodal acceleration: the lumped residual, the PML's nodal damping, and
  // at the rim either the absorbing condition or the PML's outer wall
  struct NodalRate {
    int boundary;
    double c;
    void operator()(const double & r, const double & Ml, const double & w_rim, const vec2 & alpha,
                    const double & u, const double & v, double & a) const {
      a = -r / Ml - (alpha[0] + alpha[1]) * v - alpha[0] * alpha[1] * u;
      if (boundary == ABC) {
        // Bayliss-Turkel: d_r u + (1/c) d_t u + u / (2 R) = 0 on the unit circle
        a -= w_rim * (c * v + 0.5 * c * c * u) / Ml;
      } else if (boundary == PML && w_rim > 0) {
        a = 0.0;    // the outer wall stays at rest
      }
    }
  };

  WaveDisk(Params p) : params(p), mesh(build_mesh()), u_field(create_field<Family::H1>(mesh, 1, 1)),
                       phi(u_field), domain(mesh, MeshQuadratureRule(2)) {
    nn = mesh.vert.shape[0];
    nq = total(domain.num_qpts);

    // the frame and the profile at the quadrature points, the profile at the nodes
    pml::Radial<2> layer{vec2{0.0, 0.0}, 1.0, layer_thickness, alpha_max};
    std::function<vec2(const vec2 &)> alpha_of([&](const vec2 & x) { return layer.alpha(x); });
    nd::cpu_array<double, 3> x_q = evaluate(mesh.X, domain);
    frame_q = forall(std::function<mat2(const vec2 &)>([&](const vec2 & x) { return layer.frame(x); }), x_q);
    alpha_q = forall(alpha_of, x_q);
    alpha_n = forall(alpha_of, mesh.X.data);

    nd::cpu_array<double, 1> ones_q({nq});
    for (uint32_t q = 0; q < nq; q++) { ones_q(q) = 1.0; }
    Residual<Family::H1> m = integrate(dot(ones_q, phi), domain);
    Ml = m.data;

    // the rim's lumped measure: the absorbing condition's weights, or (with
    // a PML) the outer wall, which is held at u = 0
    Domain rim(boundary_of(mesh), MeshQuadratureRule(2));
    nd::cpu_array<double, 1> ones_b({total(rim.num_qpts)});
    for (uint32_t q = 0; q < ones_b.shape[0]; q++) { ones_b(q) = 1.0; }
    Residual<Family::H1> wb = integrate(dot(ones_b, phi), rim);
    w_rim = wb.data;

    // the time step from the smallest element height
    double h_min = 1e300;
    auto height = [&](Geometry g, uint32_t e, int nv) {
      vec2 x[4];
      for (int j = 0; j < nv; j++) { x[j] = X(mesh[g](e, j).index); }
      double area = 0, longest = 0;
      for (int j = 0; j < nv; j++) {
        vec2 a = x[j], b = x[(j + 1) % nv];
        area += 0.5 * dot(cross(a), b);
        longest = std::max(longest, norm(b - a));
      }
      return (nv == 3 ? 2.0 : 1.0) * std::fabs(area) / longest;
    };
    for (uint32_t e = 0; e < mesh.tri.shape[0]; e++) { h_min = std::min(h_min, height(Geometry::Triangle, e, 3)); }
    for (uint32_t e = 0; e < mesh.quad.shape[0]; e++) { h_min = std::min(h_min, height(Geometry::Quadrilateral, e, 4)); }
    dt = params.cfl * h_min / params.c;

    u.resize({nn, 1}); v.resize({nn, 1}); aux.resize({nq, 2});   // zeroed
    a.resize({nn, 1}); auxdot.resize({nq, 2}); u1.resize({nn, 1}); v1.resize({nn, 1}); aux1.resize({nq, 2});
  }

  // the unit disk, and with a PML the rim extruded radially by pml_layers
  // rows of cells; sets region, layer_thickness and alpha_max on the way
  Mesh<> build_mesh() {
    Geometry type = params.element_type == 0 ? Geometry::Triangle : Geometry::Quadrilateral;
    double h = std::pow(2.0, -params.resolution);
    Mesh<> disk = Mesh<>::disk(vec2{0.0, 0.0}, 1.0, h, 1, type);
    uint32_t nv = disk.vert.shape[0];
    uint32_t ncell = disk[type].shape[0];
    int nvc = type == Geometry::Triangle ? 3 : 4;

    // the rim's vertices, in order around the circle
    SubMesh<> bdr = boundary_of(disk);
    std::vector<std::pair<double, uint32_t>> rim;
    for (uint32_t i = 0; i < bdr.vert.shape[0]; i++) {
      uint32_t vtx = bdr.vert(i);
      rim.push_back({std::atan2(disk.X.data(vtx, 1), disk.X.data(vtx, 0)), vtx});
    }
    std::sort(rim.begin(), rim.end());
    // the rows are as thick as the disk's own cells at the rim (h, not the
    // rim edge length: a size jump at the interface reflects on its own)
    uint32_t nrim = uint32_t(rim.size());
    int layers = params.boundary == PML ? params.pml_layers : 0;
    layer_thickness = layers * h;
    alpha_max = params.pml_decay * params.c / h;

    nd::cpu_array<double, 2> nodes({nv + layers * nrim, 2});
    for (uint32_t i = 0; i < nv; i++) { nodes(i, 0) = disk.X.data(i, 0); nodes(i, 1) = disk.X.data(i, 1); }
    auto ring = [&](int layer, uint32_t k) { return layer == 0 ? rim[k % nrim].second : nv + (layer - 1) * nrim + (k % nrim); };
    for (int layer = 1; layer <= layers; layer++) {
      double r = 1.0 + layer * h;
      for (uint32_t k = 0; k < nrim; k++) {
        nodes(ring(layer, k), 0) = r * std::cos(rim[k].first);
        nodes(ring(layer, k), 1) = r * std::sin(rim[k].first);
      }
    }

    uint32_t ring_cells = layers * nrim * (nvc == 3 ? 2 : 1);
    nd::cpu_array<uint32_t, 2> cells({ncell + ring_cells, uint32_t(nvc)});
    for (uint32_t e = 0; e < ncell; e++) { for (int j = 0; j < nvc; j++) { cells(e, j) = disk[type](e, j).index; } }
    region.assign(ncell + ring_cells, 0);
    uint32_t e = ncell;
    for (int layer = 0; layer < layers; layer++) {
      for (uint32_t k = 0; k < nrim; k++) {
        // counterclockwise like the disk's own cells: outward first, then along the rim
        uint32_t a = ring(layer, k), b = ring(layer + 1, k), c = ring(layer + 1, k + 1), d = ring(layer, k + 1);
        if (nvc == 4) {
          cells(e, 0) = a; cells(e, 1) = b; cells(e, 2) = c; cells(e, 3) = d;
          region[e++] = 1;
        } else if ((k + layer) % 2 == 0) {
          cells(e, 0) = a; cells(e, 1) = b; cells(e, 2) = c; region[e++] = 1;
          cells(e, 0) = a; cells(e, 1) = c; cells(e, 2) = d; region[e++] = 1;
        } else {
          cells(e, 0) = a; cells(e, 1) = b; cells(e, 2) = d; region[e++] = 1;
          cells(e, 0) = b; cells(e, 1) = c; cells(e, 2) = d; region[e++] = 1;
        }
      }
    }
    nd::cpu_array<uint32_t, 2> none({0, 0});
    return nvc == 3 ? Mesh<>::create_2D(nodes, 1, cells, none) : Mesh<>::create_2D(nodes, 1, none, cells);
  }

  // the rate of the state (u, v, aux): udot = v, vdot = a, auxdot.  The
  // work arrays are members, so evaluate() and integrate() refill them in place
  void rate(const nd::cpu_array<double, 2> & u_, const nd::cpu_array<double, 2> & v_, const nd::cpu_array<double, 2> & aux_,
            nd::cpu_array<double, 2> & a_, nd::cpu_array<double, 2> & auxdot) {
    u_field.data = u_;
    du_q = evaluate(grad(u_field), domain);
    forall(ScalarPML<2>::Rate{params.c * params.c}, frame_q, alpha_q, du_q, aux_, flux_q, auxdot);
    r = integrate(dot(flux_q, grad(phi)), domain);
    forall(NodalRate{params.boundary, params.c}, r.data, Ml, w_rim, alpha_n, u_, v_, a_);
  }

  void step() {
    // one stage: x <- w0 x_n + w1 (x0 + dt xdot), x_n the state at the start of the step
    auto blend = [&](auto & xn, auto & x0, auto & xdot, double w0, double w1, auto & out) {
      for (uint32_t k = 0; k < xn.sz; k++) { out.data()[k] = w0 * xn.data()[k] + w1 * (x0.data()[k] + dt * xdot.data()[k]); }
    };
    auto update = [&](auto & u0, auto & v0, auto & aux0, double w0, double w1, auto & uo, auto & vo, auto & alo) {
      blend(u, u0, v0, w0, w1, uo);
      blend(v, v0, a, w0, w1, vo);
      blend(aux, aux0, auxdot, w0, w1, alo);
    };
    // SSP-RK3
    rate(u, v, aux, a, auxdot);
    update(u, v, aux, 0.0, 1.0, u1, v1, aux1);
    rate(u1, v1, aux1, a, auxdot);
    update(u1, v1, aux1, 0.75, 0.25, u1, v1, aux1);
    rate(u1, v1, aux1, a, auxdot);
    update(u1, v1, aux1, 1.0 / 3.0, 2.0 / 3.0, u, v, aux);
    t += dt;
  }

  void advance(int steps) { for (int k = 0; k < steps; k++) { step(); } }
  void advance_to(double time) { while (t < time - 0.5 * dt) { step(); } }

  // set off a pulse at x0 now: the nodal Gaussian in the finite element
  // field, the exact one in the analytic field
  void pulse(double x, double y, double width) {
    vec2 x0{x, y};
    for (uint32_t i = 0; i < nn; i++) {
      if (params.boundary == PML && w_rim(i, 0) > 0) continue;
      u(i, 0) += std::exp(-norm_squared(X(i) - x0) / (width * width));
    }
    pulses.push_back(Pulse(x0, t, width));
    exact_time = -1;
  }

  // the analytic field at the nodes, now
  const std::vector<double> & exact() {
    if (exact_time != t) {
      exact_cache.assign(nn, 0.0);
      for (Pulse & p : pulses) {
        if (t < p.t0) continue;
        p.tabulate(t, params.c);
        for (uint32_t i = 0; i < nn; i++) { exact_cache[i] += p.at(X(i)); }
      }
      exact_time = t;
    }
    return exact_cache;
  }

  // rms of (finite element - analytic) over the nodes of the physical disk
  double error() {
    const std::vector<double> & ex = exact();
    double sum = 0; int count = 0;
    for (uint32_t i = 0; i < nn; i++) {
      if (norm(X(i)) > 1.0 + 1e-9) continue;
      sum += (u(i, 0) - ex[i]) * (u(i, 0) - ex[i]);
      count++;
    }
    return std::sqrt(sum / std::max(count, 1));
  }

  std::vector<double> nodes() const { return vertex_coordinates(mesh); }
  std::vector<uint32_t> cells() const { return cell_vertices(mesh, params.element_type == 0 ? Geometry::Triangle : Geometry::Quadrilateral); }

};

#ifdef __EMSCRIPTEN__

using namespace emscripten;

EMSCRIPTEN_BINDINGS(wave_disk_2D) {
  value_object<WaveDisk::Params>("WaveDiskParams")
    .field("elementType", &WaveDisk::Params::element_type)
    .field("boundary", &WaveDisk::Params::boundary)
    .field("resolution", &WaveDisk::Params::resolution)
    .field("c", &WaveDisk::Params::c)
    .field("cfl", &WaveDisk::Params::cfl)
    .field("pmlLayers", &WaveDisk::Params::pml_layers)
    .field("pmlDecay", &WaveDisk::Params::pml_decay);

  class_<WaveDisk>("WaveDisk")
    .constructor<WaveDisk::Params>()
    .function("step", &WaveDisk::step)
    .function("advance", &WaveDisk::advance)
    .function("advanceTo", &WaveDisk::advance_to)
    .function("pulse", &WaveDisk::pulse)
    .function("error", &WaveDisk::error)
    .function("time", +[](const WaveDisk & w) { return w.t; })
    .function("timeStep", +[](const WaveDisk & w) { return w.dt; })
    .function("layerThickness", +[](const WaveDisk & w) { return w.layer_thickness; })
    .function("pressure", +[](const WaveDisk & w) { return to_f64(w.u); })
    .function("exact", +[](WaveDisk & w) { return to_f64(w.exact()); })
    .function("region", +[](const WaveDisk & w) { return to_i32(w.region); })
    .function("nodes", +[](const WaveDisk & w) { return to_f64(w.nodes()); })
    .function("cells", +[](const WaveDisk & w) { return to_u32(w.cells()); });
}

#else

int main() {
  // a pulse near the rim, so part of it meets the boundary at a grazing
  // angle: run until the analytic field has all but left the disk, when
  // whatever is left in the finite element field is reflection
  const char * names[] = {"hard wall", "absorbing condition", "PML ring"};
  const char * elements[] = {"triangles", "quadrilaterals"};
  for (int type = 0; type < 2; type++) {
    for (int bc = 0; bc < 3; bc++) {
      WaveDisk::Params p;
      p.element_type = type;
      p.boundary = bc;
      WaveDisk wave(p);
      wave.pulse(0.6, 0.2, 0.1);
      std::cout << elements[type] << ", " << names[bc] << ": "
                << wave.mesh[type == 0 ? Geometry::Triangle : Geometry::Quadrilateral].shape[0] << " cells, dt = " << wave.dt
                << ", rms error vs analytic";
      for (double time : {0.5, 1.5, 2.5, 4.0}) {
        wave.advance_to(time);
        std::cout << " " << wave.error() << " (t = " << time << ")";
      }
      std::cout << std::endl;
    }
  }
  return 0;
}

#endif
