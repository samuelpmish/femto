"use strict";

// shared by the example pages: a WebGL viewer drawing layers of finite
// element geometry colored by a scalar (per vertex or per cell), with an orbit
// camera in 3D or pan/zoom in 2D, plus small helpers for the control panel
// (sliders, button groups, a coalescing runner for slow solves, line charts).
//
// Geometry uses unshared vertices, one per polygon corner, so per-cell values
// and flat shading need no extra bookkeeping; the example meshes are small
// enough that the duplication does not matter.

// ---------------------------------------------------------------- color map
const RAMP = [[0.15, 0.09, 0.42], [0.12, 0.48, 0.66], [0.22, 0.77, 0.54], [0.96, 0.89, 0.35]];
function ramp(s) {
  s = Math.min(1, Math.max(0, s)) * 3;
  const i = Math.min(2, Math.floor(s)), t = s - i;
  return [0, 1, 2].map(k => RAMP[i][k] + t * (RAMP[i + 1][k] - RAMP[i][k]));
}
const RAMP_CSS = {
  sequential: "linear-gradient(90deg, #26186b, #1f7aa8, #37c48a, #f4e45a)",
  diverging: "linear-gradient(90deg, #2154cc, #ededf2, #d9382e)"
};

// ---------------------------------------------------------------- geometry
// facets: size-prefixed vertex id lists (3 or 4 ids each, the format of the
// examples' surface() export); cellIds: optional cell per facet
class Geometry {
  constructor(nodes, dim, facets, cellIds) {
    this.nodes = nodes; this.dim = dim;
    let nc = 0, nt = 0, ne = 0, k = 0, f = 0;
    while (k < facets.length) { const s = facets[k]; nc += s; nt += 3 * (s - 2); ne += 2 * s; k += s + 1; f++; }
    this.corner = new Uint32Array(nc);
    this.cell = new Int32Array(nc).fill(-1);
    this.positions = new Float32Array(3 * nc);
    this.base = null;
    this.scalars = new Float32Array(nc);
    this.triangles = new Uint32Array(nt);
    this.edges = new Uint32Array(ne);
    let c = 0, t = 0, e = 0;
    k = 0; f = 0;
    while (k < facets.length) {
      const s = facets[k];
      for (let j = 0; j < s; j++) {
        this.corner[c + j] = facets[k + 1 + j];
        if (cellIds) this.cell[c + j] = cellIds[f];
      }
      for (let j = 1; j + 1 < s; j++) { this.triangles.set([c, c + j, c + j + 1], t); t += 3; }
      for (let j = 0; j < s; j++) { this.edges.set([c + j, c + (j + 1) % s], e); e += 2; }
      c += s; k += s + 1; f++;
    }
    this.dirty = { positions: true, scalars: true };
    this.setPositions(nodes);
  }

  setPositions(nodes) {
    const d = this.dim;
    for (let i = 0; i < this.corner.length; i++) {
      const v = this.corner[i];
      this.positions[3 * i] = nodes[d * v];
      this.positions[3 * i + 1] = nodes[d * v + 1];
      this.positions[3 * i + 2] = d === 3 ? nodes[d * v + 2] : 0;
    }
    this.dirty.positions = true;
  }

  // per mesh vertex values with `stride` components: one component, or the magnitude
  vertexScalar(values, stride = 1, component = 0) {
    for (let i = 0; i < this.corner.length; i++) {
      const v = this.corner[i];
      if (component === "norm") {
        let m = 0;
        for (let c = 0; c < stride; c++) m += values[stride * v + c] ** 2;
        this.scalars[i] = Math.sqrt(m);
      } else {
        this.scalars[i] = values[stride * v + component];
      }
    }
    this.dirty.scalars = true;
    return this;
  }

  cellScalar(values) {
    for (let i = 0; i < this.corner.length; i++) this.scalars[i] = this.cell[i] >= 0 ? values[this.cell[i]] : 0;
    this.dirty.scalars = true;
    return this;
  }

  // positions = rest + scale * displacement (dim components per mesh vertex)
  deform(disp, scale) {
    const d = this.dim;
    if (!this.base) { this.base = new Float32Array(this.positions); }
    for (let i = 0; i < this.corner.length; i++) {
      const v = this.corner[i];
      for (let c = 0; c < d; c++) this.positions[3 * i + c] = this.base[3 * i + c] + scale * disp[d * v + c];
    }
    this.dirty.positions = true;
    return this;
  }

  range() {
    let lo = Infinity, hi = -Infinity;
    for (const s of this.scalars) { if (s < lo) lo = s; if (s > hi) hi = s; }
    return [lo, hi];
  }
}

// a set of line segments (arrows, outlines), 3 coordinates per endpoint
class Lines {
  constructor(positions) { this.positions = positions; this.dirty = { positions: true }; }
  set(positions) { this.positions = positions; this.dirty.positions = true; }
}

// the cells of a 2D mesh as facets, with their cell ids (nv = 3 triangles, 4
// quadrilaterals)
function cellFacets(cells, nv, keep = () => true) {
  const facets = [], cellIds = [];
  for (let e = 0; e < cells.length / nv; e++) {
    if (!keep(e)) continue;
    facets.push(nv);
    for (let j = 0; j < nv; j++) facets.push(cells[nv * e + j]);
    cellIds.push(e);
  }
  return { facets: new Uint32Array(facets), cellIds: new Uint32Array(cellIds) };
}

// the outer faces of a subset of cells (tetrahedra or hexahedra, in femto's
// vertex order): every interior face is seen twice and dropped
const CELL_FACES = {
  4: [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
  8: [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
};
function cellBoundary(cells, nv, keep = () => true) {
  const faces = CELL_FACES[nv];
  const seen = new Map();
  for (let e = 0; e < cells.length / nv; e++) {
    if (!keep(e)) continue;
    for (const f of faces) {
      const ids = f.map(k => cells[nv * e + k]);
      const key = ids.slice().sort((a, b) => a - b).join(",");
      if (seen.has(key)) seen.delete(key); else seen.set(key, { ids, cell: e });
    }
  }
  const n = faces[0].length;
  const facets = new Uint32Array((n + 1) * seen.size), cellIds = new Uint32Array(seen.size);
  let k = 0;
  for (const { ids, cell } of seen.values()) {
    facets[(n + 1) * k] = n;
    facets.set(ids, (n + 1) * k + 1);
    cellIds[k++] = cell;
  }
  return { facets, cellIds };
}

// centroids of cells with `nv` vertices each
function centroids(nodes, dim, cells, nv) {
  const n = cells.length / nv, out = new Float64Array(dim * n);
  for (let e = 0; e < n; e++) {
    for (let j = 0; j < nv; j++) {
      for (let c = 0; c < dim; c++) out[dim * e + c] += nodes[dim * cells[nv * e + j] + c] / nv;
    }
  }
  return out;
}

// arrow segments (shaft + two barbs) at points along vectors, with the length
// given by len(magnitude); 2D input is drawn in the plane z = 0
function arrows(points, vectors, dim, len) {
  const n = points.length / dim, out = [];
  for (let i = 0; i < n; i++) {
    const v = [0, 0, 0], p = [0, 0, 0];
    for (let c = 0; c < dim; c++) { v[c] = vectors[dim * i + c]; p[c] = points[dim * i + c]; }
    const m = Math.hypot(v[0], v[1], v[2]);
    const L = len(m);
    if (!(m > 0) || !(L > 0)) continue;
    const d = v.map(x => x / m);
    let q = dim === 2 ? [-d[1], d[0], 0] : (Math.abs(d[2]) < 0.9 ? [-d[1], d[0], 0] : [0, -d[2], d[1]]);
    const ql = Math.hypot(...q); q = q.map(x => x / ql);
    const tip = [p[0] + L * d[0], p[1] + L * d[1], p[2] + L * d[2]];
    out.push(...p, ...tip);
    const b = 0.3 * L;
    for (const sgn of [1, -1]) {
      out.push(...tip, tip[0] - b * (d[0] - sgn * 0.6 * q[0]), tip[1] - b * (d[1] - sgn * 0.6 * q[1]), tip[2] - b * (d[2] - sgn * 0.6 * q[2]));
    }
  }
  return new Float32Array(out);
}

function bounds(positions) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i += 3) {
    for (let c = 0; c < 3; c++) { lo[c] = Math.min(lo[c], positions[i + c]); hi[c] = Math.max(hi[c], positions[i + c]); }
  }
  return { lo, hi, center: [0, 1, 2].map(c => 0.5 * (lo[c] + hi[c])), radius: 0.5 * Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) };
}

// ---------------------------------------------------------------- viewer
const VS = `#version 300 es
  in vec3 position;
  in float scalar;
  uniform mat4 mvp;
  out vec3 vpos;
  out float vscalar;
  void main() { vpos = position; vscalar = scalar; gl_Position = mvp * vec4(position, 1.0); }`;

const FS = `#version 300 es
  precision highp float;
  in vec3 vpos;
  in float vscalar;
  uniform vec3 color;
  uniform float useScalar, lo, hi, alpha, lit, palette;
  out vec4 frag;
  vec3 ramp(float s) {
    s = clamp(s, 0.0, 1.0);
    if (palette > 0.5) {   // diverging, for signed fields: blue through near-white to red
      vec3 c0 = vec3(0.13, 0.33, 0.80), c1 = vec3(0.93, 0.93, 0.95), c2 = vec3(0.85, 0.22, 0.18);
      return s < 0.5 ? mix(c0, c1, 2.0 * s) : mix(c1, c2, 2.0 * s - 1.0);
    }
    vec3 c0 = vec3(0.15, 0.09, 0.42), c1 = vec3(0.12, 0.48, 0.66), c2 = vec3(0.22, 0.77, 0.54), c3 = vec3(0.96, 0.89, 0.35);
    s *= 3.0;
    if (s < 1.0) return mix(c0, c1, s);
    if (s < 2.0) return mix(c1, c2, s - 1.0);
    return mix(c2, c3, s - 2.0);
  }
  void main() {
    vec3 base = mix(color, ramp((vscalar - lo) / max(hi - lo, 1e-30)), useScalar);
    float light = 1.0;
    if (lit > 0.5) {
      vec3 n = normalize(cross(dFdx(vpos), dFdy(vpos)));
      float d1 = abs(dot(n, normalize(vec3( 0.4,  0.3, 0.85))));
      float d2 = abs(dot(n, normalize(vec3(-0.6, -0.4, 0.10))));
      light = 0.25 + 0.6 * d1 + 0.2 * d2;
    }
    frag = vec4(base * light, alpha);
  }`;

const M4 = {
  mul(a, b) {
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++)
      for (let r = 0; r < 4; r++)
        o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    return o;
  },
  perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
  },
  lookAt(eye, target, up) {
    const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const norm = a => { const l = Math.hypot(...a); return [a[0] / l, a[1] / l, a[2] / l]; };
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const z = norm(sub(eye, target)), x = norm(cross(up, z)), y = cross(z, x);
    return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1]);
  },
  ortho(pan, hx, hy, shift = 0) {
    return new Float32Array([1 / hx, 0, 0, 0, 0, 1 / hy, 0, 0, 0, 0, -0.001, 0, shift - pan[0] / hx, -pan[1] / hy, 0, 1]);
  }
};

class Viewer {
  constructor(canvas, mode) {
    this.canvas = canvas;
    this.mode = mode;
    const gl = this.gl = canvas.getContext("webgl2", { antialias: true });
    const compile = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw gl.getShaderInfoLog(s);
      return s;
    };
    const p = this.program = gl.createProgram();
    gl.attachShader(p, compile(gl.VERTEX_SHADER, VS));
    gl.attachShader(p, compile(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw gl.getProgramInfoLog(p);
    this.loc = {
      position: gl.getAttribLocation(p, "position"), scalar: gl.getAttribLocation(p, "scalar"),
      mvp: gl.getUniformLocation(p, "mvp"), color: gl.getUniformLocation(p, "color"),
      useScalar: gl.getUniformLocation(p, "useScalar"), lo: gl.getUniformLocation(p, "lo"), hi: gl.getUniformLocation(p, "hi"),
      alpha: gl.getUniformLocation(p, "alpha"), lit: gl.getUniformLocation(p, "lit"), palette: gl.getUniformLocation(p, "palette")
    };
    this.layers = [];
    this.range = [0, 1];
    this.palette = "sequential";   // or "diverging" (blue-white-red) for signed fields
    this.panelWidth = 332;   // the control panel's footprint, which the model is kept clear of
    this.camera = mode === "3d"
      ? { yaw: 0.7, pitch: 0.4, dist: 10, target: [0, 0, 0], fovy: 40 * Math.PI / 180 }
      : { pan: [0, 0], halfHeight: 1 };
    this.installControls();
  }

  // style: color [r,g,b], scalar (color by the geometry's scalars), alpha,
  // edges (draw the polygon outlines), lit (shaded by the facet normal),
  // edgeColor, depthTest (false draws the layer over everything, for arrows
  // that would otherwise sit inside the surface they belong to)
  add(object, style = {}) {
    const layer = Object.assign({ object, color: [0.8, 0.8, 0.8], scalar: false, alpha: 1, edges: false, lit: this.mode === "3d",
                                  edgeColor: [0.1, 0.1, 0.12], visible: true, depthTest: true }, style);
    this.layers.push(layer);
    return layer;
  }
  remove(layer) { this.layers = this.layers.filter(l => l !== layer); }
  clear() { this.layers = []; }

  // the fraction of the width the panel covers: the projection is shifted by
  // it so the model is centered in what is left, and fit() zooms out to match
  inset() {
    const w = Math.max(1, this.canvas.clientWidth);
    return Math.min(this.panelWidth, 0.45 * w) / w;
  }

  fit(positions, margin = 1.1) {
    const b = bounds(positions);
    const room = margin / (1 - this.inset());
    if (this.mode === "3d") {
      this.camera.target = b.center;
      this.camera.dist = room * b.radius / Math.sin(this.camera.fovy / 2);
    } else {
      this.camera.pan = [b.center[0], b.center[1]];
      const aspect = this.canvas.clientWidth / Math.max(1, this.canvas.clientHeight);
      this.camera.halfHeight = Math.max(margin * 0.5 * (b.hi[1] - b.lo[1]), room * 0.5 * (b.hi[0] - b.lo[0]) / aspect);
    }
  }

  buffers(object) {
    const gl = this.gl;
    if (!object.gpu) {
      object.gpu = { position: gl.createBuffer(), scalar: gl.createBuffer(), triangles: gl.createBuffer(), edges: gl.createBuffer() };
      if (object instanceof Geometry) {
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, object.gpu.triangles);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, object.triangles, gl.STATIC_DRAW);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, object.gpu.edges);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, object.edges, gl.STATIC_DRAW);
      }
    }
    if (object.dirty.positions) {
      gl.bindBuffer(gl.ARRAY_BUFFER, object.gpu.position);
      gl.bufferData(gl.ARRAY_BUFFER, object.positions, gl.DYNAMIC_DRAW);
      object.dirty.positions = false;
    }
    if (object.dirty.scalars) {
      gl.bindBuffer(gl.ARRAY_BUFFER, object.gpu.scalar);
      gl.bufferData(gl.ARRAY_BUFFER, object.scalars, gl.DYNAMIC_DRAW);
      object.dirty.scalars = false;
    }
    return object.gpu;
  }

  mvp(w, h) {
    const c = this.camera, shift = this.inset();
    if (this.mode === "3d") {
      const cp = Math.cos(c.pitch), sp = Math.sin(c.pitch);
      const eye = [c.target[0] + c.dist * cp * Math.cos(c.yaw), c.target[1] + c.dist * cp * Math.sin(c.yaw), c.target[2] + c.dist * sp];
      const proj = M4.perspective(c.fovy, w / h, c.dist * 0.01, c.dist * 10);
      proj[8] = -shift;    // off-axis projection: everything moves right by `shift`
      return M4.mul(proj, M4.lookAt(eye, c.target, [0, 0, 1]));
    }
    return M4.ortho(c.pan, c.halfHeight * w / h, c.halfHeight, shift);
  }

  draw() {
    const gl = this.gl, canvas = this.canvas, loc = this.loc;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    gl.viewport(0, 0, w, h);
    gl.clearColor(0.086, 0.094, 0.114, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(this.program);
    gl.uniformMatrix4fv(loc.mvp, false, this.mvp(w, h));
    gl.uniform1f(loc.lo, this.range[0]);
    gl.uniform1f(loc.hi, this.range[1]);
    gl.uniform1f(loc.palette, this.palette === "diverging" ? 1 : 0);

    // opaque layers first, then translucent ones without depth writes
    const order = this.layers.filter(l => l.visible && l.alpha >= 1).concat(this.layers.filter(l => l.visible && l.alpha < 1));
    for (const layer of order) {
      const object = layer.object, gpu = this.buffers(object);
      const depth = this.mode === "3d" && layer.depthTest !== false;
      if (depth) { gl.enable(gl.DEPTH_TEST); gl.depthMask(layer.alpha >= 1); } else { gl.disable(gl.DEPTH_TEST); }
      gl.bindBuffer(gl.ARRAY_BUFFER, gpu.position);
      gl.enableVertexAttribArray(loc.position);
      gl.vertexAttribPointer(loc.position, 3, gl.FLOAT, false, 0, 0);
      if (object instanceof Lines) {
        gl.disableVertexAttribArray(loc.scalar);
        gl.vertexAttrib1f(loc.scalar, 0);
        gl.uniform3fv(loc.color, layer.color);
        gl.uniform1f(loc.useScalar, 0);
        gl.uniform1f(loc.alpha, layer.alpha);
        gl.uniform1f(loc.lit, 0);
        gl.drawArrays(gl.LINES, 0, object.positions.length / 3);
        continue;
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, gpu.scalar);
      gl.enableVertexAttribArray(loc.scalar);
      gl.vertexAttribPointer(loc.scalar, 1, gl.FLOAT, false, 0, 0);
      if (layer.fill !== false) {
        gl.uniform3fv(loc.color, layer.color);
        gl.uniform1f(loc.useScalar, layer.scalar ? 1 : 0);
        gl.uniform1f(loc.alpha, layer.alpha);
        gl.uniform1f(loc.lit, layer.lit ? 1 : 0);
        gl.enable(gl.POLYGON_OFFSET_FILL);
        gl.polygonOffset(1.0, 1.0);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gpu.triangles);
        gl.drawElements(gl.TRIANGLES, object.triangles.length, gl.UNSIGNED_INT, 0);
        gl.disable(gl.POLYGON_OFFSET_FILL);
      }
      if (layer.edges) {
        gl.uniform3fv(loc.color, layer.edgeColor);
        gl.uniform1f(loc.useScalar, 0);
        gl.uniform1f(loc.alpha, 1);
        gl.uniform1f(loc.lit, 0);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gpu.edges);
        gl.drawElements(gl.LINES, object.edges.length, gl.UNSIGNED_INT, 0);
      }
    }
    gl.depthMask(true);
  }

  // the render loop; onFrame(timestamp) runs before every draw
  start(onFrame = () => {}) {
    const frame = t => { onFrame(t); this.draw(); requestAnimationFrame(frame); };
    requestAnimationFrame(frame);
  }

  // a client position as normalized device coordinates, the panel's shift undone
  ndc(clientX, clientY) {
    const r = this.canvas.getBoundingClientRect();
    return [2 * (clientX - r.left) / this.canvas.clientWidth - 1 - this.inset(), 1 - 2 * (clientY - r.top) / this.canvas.clientHeight];
  }

  // the world point under a client position (2D mode), for pages that take clicks
  worldAt(clientX, clientY) {
    const c = this.camera, [fx, fy] = this.ndc(clientX, clientY);
    const aspect = this.canvas.clientWidth / Math.max(1, this.canvas.clientHeight);
    return [c.pan[0] + fx * c.halfHeight * aspect, c.pan[1] + fy * c.halfHeight];
  }

  // one pointer orbits (3D) or pans (2D), with shift or the right button
  // panning in 3D; two fingers pinch to zoom about their midpoint and pan with
  // it; the wheel zooms about the cursor
  installControls() {
    const canvas = this.canvas, c = this.camera;
    const pointers = new Map();   // pointerId -> last [clientX, clientY]
    let panning = false;

    const panBy = (dx, dy) => {
      if (this.mode === "3d") {
        const s = c.dist * 0.0016;
        c.target[0] += s * Math.sin(c.yaw) * dx;
        c.target[1] += s * -Math.cos(c.yaw) * dx;
        c.target[2] += s * dy;
      } else {
        const s = 2 * c.halfHeight / Math.max(1, canvas.clientHeight);
        c.pan[0] -= s * dx;
        c.pan[1] += s * dy;
      }
    };

    // scale the view by `factor`, keeping the point under (clientX, clientY) put
    const zoomAt = (clientX, clientY, factor) => {
      if (this.mode === "3d") { c.dist *= factor; return; }
      const [fx, fy] = this.ndc(clientX, clientY);
      const aspect = canvas.clientWidth / Math.max(1, canvas.clientHeight);
      const before = this.worldAt(clientX, clientY);
      c.halfHeight *= factor;
      c.pan[0] = before[0] - fx * c.halfHeight * aspect;
      c.pan[1] = before[1] - fy * c.halfHeight;
    };

    canvas.addEventListener("pointerdown", e => {
      pointers.set(e.pointerId, [e.clientX, e.clientY]);
      if (pointers.size === 1) { panning = e.shiftKey || e.button === 2; }
      canvas.setPointerCapture(e.pointerId);
    });
    const release = e => { pointers.delete(e.pointerId); };
    canvas.addEventListener("pointerup", release);
    canvas.addEventListener("pointercancel", release);   // iOS ends touches this way
    canvas.addEventListener("pointermove", e => {
      const last = pointers.get(e.pointerId);
      if (!last) return;
      const dx = e.clientX - last[0], dy = e.clientY - last[1];
      if (pointers.size === 1) {
        if (this.mode === "3d" && !panning) {
          c.yaw -= dx * 0.008;
          c.pitch = Math.min(1.5, Math.max(-1.5, c.pitch + dy * 0.008));
        } else {
          panBy(dx, dy);
        }
      } else if (pointers.size === 2) {
        const other = [...pointers].find(([id]) => id !== e.pointerId)[1];
        const d0 = Math.hypot(last[0] - other[0], last[1] - other[1]);
        const d1 = Math.hypot(e.clientX - other[0], e.clientY - other[1]);
        const mid0 = [0.5 * (last[0] + other[0]), 0.5 * (last[1] + other[1])];
        const mid1 = [0.5 * (e.clientX + other[0]), 0.5 * (e.clientY + other[1])];
        if (d0 > 1 && d1 > 1) { zoomAt(mid0[0], mid0[1], d0 / d1); }
        panBy(mid1[0] - mid0[0], mid1[1] - mid0[1]);
      }
      pointers.set(e.pointerId, [e.clientX, e.clientY]);
    });
    canvas.addEventListener("wheel", e => {
      e.preventDefault();
      zoomAt(e.clientX, e.clientY, Math.exp(e.deltaY * 0.0012));
    }, { passive: false });
    canvas.addEventListener("contextmenu", e => e.preventDefault());
  }
}

// ---------------------------------------------------------------- panel
// the color bar in the corner: label, and the range the viewer maps
function legend(viewer, lo, hi, label, fmt = v => v.toPrecision(3)) {
  const el = document.getElementById("legend");
  if (!el) return;
  viewer.range = [lo, hi];
  el.style.display = "flex";
  el.querySelector(".bar").style.background = RAMP_CSS[viewer.palette] || RAMP_CSS.sequential;
  el.querySelector(".label").textContent = label;
  el.querySelector(".min").textContent = fmt(lo);
  el.querySelector(".max").textContent = fmt(hi);
}

// every range input in the panel mirrors its value into <output id="<id>_out">
// (formatted by `formats[id]`, or to the slider's step); returns a getter for
// all values, and onChange(id) fires as a slider moves
function sliders(onChange, formats = {}) {
  const inputs = [...document.querySelectorAll("#panel input[type=range]")];
  const decimals = s => { const step = String(s.step || "1"); return step.includes(".") ? step.split(".")[1].length : 0; };
  const show = s => { const out = document.getElementById(s.id + "_out"); if (out) out.value = (formats[s.id] || (v => v.toFixed(decimals(s))))(+s.value); };
  for (const s of inputs) { s.addEventListener("input", () => { show(s); onChange(s.id); }); show(s); }
  return () => Object.fromEntries(inputs.map(s => [s.id, +s.value]));
}

// a group of buttons where one is active; returns a getter for the active index
function buttonGroup(ids, onChange, initial = 0) {
  let active = initial;
  const buttons = ids.map(id => document.getElementById(id));
  const show = () => buttons.forEach((b, k) => b.classList.toggle("active", k === active));
  buttons.forEach((b, k) => b.addEventListener("click", () => { if (k === active) return; active = k; show(); onChange(k); }));
  show();
  return () => active;
}

// runs slow synchronous work off the input event so the panel repaints first;
// requests that arrive while it runs collapse into one more run
function runner(work) {
  let running = false, pending = false;
  const go = async () => {
    pending = false; running = true;
    try { await work(); } finally { running = false; }   // work may be async (fetching a mesh)
    if (pending) setTimeout(go, 0);
  };
  return () => { pending = true; if (!running) setTimeout(go, 0); };
}

// a line chart on a 2D canvas: series = [{x, y, color, dots, label}]
function chart(canvas, series, { xlabel = "", ylabel = "", xrange, yrange } = {}) {
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);
  const all = a => series.flatMap(s => Array.from(s[a]));
  const ext = (vals, given) => {
    if (given) return given;
    let lo = Math.min(...vals), hi = Math.max(...vals);
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    if (hi - lo < 1e-12) { hi = lo + 1; }
    const pad = 0.05 * (hi - lo);
    return [lo - pad, hi + pad];
  };
  const [x0, x1] = ext(all("x"), xrange), [y0, y1] = ext(all("y"), yrange);
  const L = 34, R = 8, T = 8, B = 22;
  const sx = x => L + (x - x0) / (x1 - x0) * (W - L - R), sy = y => T + (y1 - y) / (y1 - y0) * (H - T - B);
  ctx.strokeStyle = "#ffffff33"; ctx.lineWidth = 1;
  ctx.strokeRect(L, T, W - L - R, H - T - B);
  if (y0 < 0 && y1 > 0) { ctx.beginPath(); ctx.moveTo(L, sy(0)); ctx.lineTo(W - R, sy(0)); ctx.stroke(); }
  if (x0 < 0 && x1 > 0) { ctx.beginPath(); ctx.moveTo(sx(0), T); ctx.lineTo(sx(0), H - B); ctx.stroke(); }
  ctx.fillStyle = "#9aa3b2"; ctx.font = "10px system-ui, sans-serif";
  const f = v => Math.abs(v) < 1e-12 ? "0" : v.toPrecision(3).replace(/\.?0+$/, "");
  ctx.textAlign = "left"; ctx.fillText(f(x0), L, H - 8); ctx.textAlign = "right"; ctx.fillText(f(x1), W - R, H - 8);
  ctx.textAlign = "center"; ctx.fillText(xlabel, 0.5 * (L + W - R), H - 8);
  ctx.save(); ctx.translate(10, 0.5 * (T + H - B)); ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center"; ctx.fillText(ylabel, 0, 0); ctx.restore();
  ctx.save(); ctx.translate(L - 4, 0); ctx.textAlign = "right";
  ctx.fillText(f(y1), 0, T + 9); ctx.fillText(f(y0), 0, H - B); ctx.restore();
  for (const s of series) {
    ctx.strokeStyle = ctx.fillStyle = s.color || "#e8a838";
    ctx.lineWidth = 1.5;
    if (s.dots) {
      for (let i = 0; i < s.x.length; i++) { ctx.beginPath(); ctx.arc(sx(s.x[i]), sy(s.y[i]), 2.2, 0, 2 * Math.PI); ctx.fill(); }
    } else {
      ctx.beginPath();
      for (let i = 0; i < s.x.length; i++) { i ? ctx.lineTo(sx(s.x[i]), sy(s.y[i])) : ctx.moveTo(sx(s.x[i]), sy(s.y[i])); }
      ctx.stroke();
    }
  }
}

function elapsed(t0) { return `${(performance.now() - t0).toFixed(0)} ms`; }

// the examples that read a mesh file expect it in the module's filesystem at
// /<name>; the pages fetch only the mesh they need rather than carrying every
// mesh inside the module
async function loadMesh(wasm, name) {
  if (wasm.FS.analyzePath("/" + name).exists) return;
  const response = await fetch(name);
  if (!response.ok) throw new Error(`${name}: ${response.status} ${response.statusText}`);
  wasm.FS.writeFile("/" + name, new Uint8Array(await response.arrayBuffer()));
}

// why a module failed to load, in the #loading element rather than only in the
// console: almost always the page is not cross-origin isolated, so the
// multithreaded module has no SharedArrayBuffer to hand its workers
function reportLoadFailure(e) {
  console.error(e);
  const why = window.crossOriginIsolated ? ""
    : " — the page is not cross-origin isolated, so the wasm threads it needs are unavailable (serve it with serve.py, or over https/localhost so enable-threads.js can install its service worker)";
  const loading = document.getElementById("loading");
  if (loading) loading.textContent = "the wasm module failed to load: " + ((e && e.message) || e) + why;
}
