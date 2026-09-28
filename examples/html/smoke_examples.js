// headless check of the examples and the viewer helpers they feed: node
// smoke_examples.js (from the build-web-mt/examples/html directory).  Every
// example lives in the one module the pages share, so this loads it once, runs
// each example with a small mesh, prints a headline number, and pushes the
// mesh it exports through viewer.js the way the pages do.
const path = require('path');
const fs = require('fs');
const here = process.argv[2] || process.cwd();   // where the *_web.js modules are

// viewer.js is a plain script for the browser; its top level only defines
// things, so it can be evaluated here to get at the geometry helpers
const V = new Function(fs.readFileSync(path.join(here, 'viewer.js'), 'utf8') +
                       '; return { Geometry, Lines, cellFacets, cellBoundary, centroids, arrows, bounds };')();

let failures = 0;
function check(name, condition, detail) {
  if (!condition) { failures++; console.error(`  FAILED ${name}${detail ? ': ' + detail : ''}`); }
}

// the faces viewer.js peels off the cells must be the boundary facets the
// example itself exports (same count, same vertex sets)
function checkSurface(name, sim, nv) {
  const cells = sim.cells(), surface = sim.surface();
  const mine = V.cellBoundary(cells, nv);
  const key = (a, i, n) => Array.from(a.slice(i, i + n)).sort((x, y) => x - y).join(',');
  const theirs = new Set();
  for (let k = 0; k < surface.length; k += surface[k] + 1) { theirs.add(key(surface, k + 1, surface[k])); }
  check(name, theirs.size === mine.cellIds.length, `${theirs.size} exported facets vs ${mine.cellIds.length} found`);
  let missing = 0;
  for (let k = 0; k < mine.facets.length; k += mine.facets[k] + 1) {
    if (!theirs.has(key(mine.facets, k + 1, mine.facets[k]))) missing++;
  }
  check(name, missing === 0, `${missing} facets not in the exported surface`);
  return mine;
}

// a geometry the viewer can draw: finite coordinates, scalars in range
function checkGeometry(name, nodes, dim, facets, cellIds, values, stride) {
  const g = new V.Geometry(nodes, dim, facets, cellIds);
  check(name, g.positions.every(Number.isFinite), 'non-finite vertex coordinate');
  if (values) {
    g.vertexScalar(values, stride || 1, stride > 1 ? 'norm' : 0);
    const [lo, hi] = g.range();
    check(name, Number.isFinite(lo) && Number.isFinite(hi) && hi >= lo, `scalar range ${lo} .. ${hi}`);
  }
  const b = V.bounds(g.positions);
  check(name, b.radius > 0, 'empty bounding box');
  return g;
}

// the pages fetch a mesh and write it into the module's filesystem; here the
// files sit next to this script
function loadMesh(name) {
  if (!wasm.FS.analyzePath('/' + name).exists) {
    wasm.FS.writeFile('/' + name, fs.readFileSync(path.join(here, name)));
  }
}

let wasm = null;
async function run(name, body) {
  const t0 = Date.now();
  const result = body(wasm);
  console.log(`${name}: ${result} (${Date.now() - t0} ms)`);
}

(async () => {
  const t0 = Date.now();
  wasm = await require(path.join(here, 'femto_examples.js'))();
  console.log(`femto_examples.js loaded (${Date.now() - t0} ms)`);

  await run('capacitor_2D', m => {
    const c = new m.Capacitor({Lx: 40, Ly: 30, plateWidth: 16, plateThickness: 1, gap: 4, dielectricWidth: 8, epsR: 4, voltage: 1, h: 1});
    c.solve();
    const { facets, cellIds } = V.cellFacets(c.cells(), 4);
    checkGeometry('capacitor_2D', c.nodes(), 2, facets, cellIds, c.voltage(), 1);
    return `C = ${c.capacitance().toFixed(3)} eps0 (ideal ${c.idealCapacitance()}), ${c.voltage().length} nodes, ${c.cells().length / 4} cells`;
  });
  await run('heatsink_3D', m => {
    loadMesh(m.HeatSink.meshFile(1));
    const s = new m.HeatSink({design: 1, power: 8, hAir: 2.5e-5, TAir: 25});
    s.solve();
    const { facets, cellIds } = checkSurface('heatsink_3D', s, 4);
    checkGeometry('heatsink_3D', s.nodes(), 3, facets, cellIds, s.temperature(), 1);
    check('heatsink_3D', s.thermalResistance() > 1 && s.thermalResistance() < 100, 'implausible thermal resistance');
    return `junction at ${s.junctionTemperature().toFixed(1)} C, ${s.thermalResistance().toFixed(1)} K/W`;
  });
  await run('fracture_2D', m => {
    loadMesh('fracture_specimen.msh');
    const f = new m.FractureSpecimen({material: 1, state: 1, E: 1000, nu: 0.3, opening: 2, loadSteps: 3});
    while (f.stepCount() < 3) f.step();
    const curve = f.curve();
    const { facets, cellIds } = V.cellFacets(f.cells(), 3);
    const g = checkGeometry('fracture_2D', f.nodes(), 2, facets, cellIds);
    g.deform(f.displacement(), 5).cellScalar(f.vonMises());
    check('fracture_2D', g.range()[1] > 0, 'no stress after three load steps');
    return `reaction ${curve[curve.length - 1].toFixed(2)} N/mm after ${f.stepCount()} steps, peak von Mises ${Math.max(...f.vonMises()).toFixed(1)} MPa`;
  });
  await run('coffee_cup_modal', m => {
    const c = new m.CoffeeCupModal({baseRadius: 0.035, rimRadius: 0.04, height: 0.095, thickness: 0.005, handles: 1, p: 1, rho: 2400, E: 70e9, nu: 0.2, numModes: 3});
    c.solve();
    const { facets, cellIds } = checkSurface('coffee_cup_modal', c, 8);
    checkGeometry('coffee_cup_modal', c.nodes(), 3, facets, cellIds, c.mode(0), 3);
    return `modes ${Array.from(c.frequencies()).map(f => f.toFixed(0)).join(', ')} Hz, mode 0 has ${c.mode(0).length / 3} vertices`;
  });
  await run('lid_driven_cavity_2D', m => {
    const c = new m.LidDrivenCavity({L: 1, n: 16, lidSpeed: 1, viscosity: 0.01, tolerance: 1e-8});
    c.solve();
    check('lid_driven_cavity_2D', c.residual() < 1e-8, `Newton did not converge: ${c.residual()}`);
    const line = c.centerline();
    let umin = 0; for (let k = 1; k < line.length; k += 2) umin = Math.min(umin, line[k]);
    const { facets, cellIds } = V.cellFacets(c.cells(), 4);
    const g = checkGeometry('lid_driven_cavity_2D', c.nodes(), 2, facets, cellIds, c.velocity(), 2);
    const centers = V.centroids(c.nodes(), 2, c.cells(), 4);
    check('lid_driven_cavity_2D', V.arrows(centers, centers, 2, () => 0.01).length > 0, 'no arrow segments');
    check('lid_driven_cavity_2D', umin < -0.15 && umin > -0.25, `centerline minimum ${umin} far from Ghia's -0.211`);
    return `Re ${c.reynolds()}, ${c.iterations()} Newton steps, centerline min u_x ${umin.toFixed(3)}`;
  });
  await run('wave_disk_2D', m => {
    const w = new m.WaveDisk({elementType: 1, boundary: 2, resolution: 4, c: 1, cfl: 0.25, pmlLayers: 8, pmlDecay: 1.4});
    w.pulse(0.6, 0.2, 0.1);
    w.advanceTo(4);
    const err = w.error(), region = w.region();
    check('wave_disk_2D', err < 5e-3, `PML residual ${err} vs analytic`);
    const { facets, cellIds } = V.cellFacets(w.cells(), 4, e => region[e] === 0);
    checkGeometry('wave_disk_2D', w.nodes(), 2, facets, cellIds, w.exact(), 1);
    return `${w.cells().length / 4} quads, PML rms error ${err.toExponential(2)} at t = 4`;
  });
  await run('magnetostatics_3D', m => {
    loadMesh('induction_coil.msh');
    const s = new m.Magnetostatics({current: 60, muR: 1000, gauge: 1e-6});
    s.solve();
    const cells = s.cells(), region = s.region();
    const core = V.cellBoundary(cells, 4, e => region[e] === 1);
    check('magnetostatics_3D', core.cellIds.length > 0, 'no core cells');
    checkGeometry('magnetostatics_3D', s.nodes(), 3, core.facets, core.cellIds);
    check('magnetostatics_3D', s.centerField() > 10 * s.solenoidEstimate(), 'the ferrite is not concentrating the field');
    return `B on the axis ${(s.centerField() * 1e3).toFixed(1)} mT, peak ${(s.maxField() * 1e3).toFixed(0)} mT`;
  });
  if (failures) { console.error(`${failures} check(s) failed`); process.exit(1); }
  console.log('all modules and viewer helpers OK');
})().catch(e => { console.error(e); process.exit(1); });
