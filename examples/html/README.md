# browser viewers for the physics examples

One page per example in `../cpp`, all driving the same webassembly module.
`viewer.js` and `viewer.css` are shared too; everything else in a page is
specific to the physics it shows.

```
../cpp/<name>.cpp     the simulation: struct + Params + main(), or
                      EMSCRIPTEN_BINDINGS under __EMSCRIPTEN__
<name>.html           its viewer: panel on the left, canvas behind
femto_examples.js     every example in one module (built, not checked in),
  + .wasm             loaded by every page as createFemtoExamples()
viewer.js             Geometry/Lines, the WebGL Viewer, panel helpers
viewer.css            the shared dark panel/legend/hint styling
index.html            the list of pages
smoke_examples.js     node check: runs every example, exercises viewer.js
```

One module rather than one per page: the examples are the small part of it.
femto, PaStiX and OpenBLAS come to about 2.3 MB and an example adds about
0.2 MB, so six separate modules meant 16 MB on the server and 2.5 MB per page,
against 3.1 MB downloaded once and cached across every page.

## building and serving

```sh
source ~/emsdk/emsdk_env.sh
emcmake cmake -B build-web-mt -DCMAKE_BUILD_TYPE=Release
cmake --build build-web-mt --target femto_examples_html
cd build-web-mt/examples/html && python3 serve.py 8000      # then open localhost:8000
node smoke_examples.js                                      # headless, no browser
```

The modules are **multithreaded by default** (`FEMTO_WASM_THREADS=ON`), which
is why the page has to be cross-origin isolated — see below (`serve.py` and
`enable-threads.js` live here and are staged next to the modules by the build).

## adding an example

1. Write `../cpp/<name>.cpp`: one struct holding `Params`, the mesh, the
   fields and the results; `solve()` or `step()`; accessors returning
   `std::vector` (`to_f64`/`to_u32`/`to_i32` in `example_common.hpp` wrap them
   as typed arrays). End the file with `#ifdef __EMSCRIPTEN__` bindings, `#else`
   a `main()` that runs a self-check.
2. Add the name to the `foreach` in `../cpp/CMakeLists.txt` and to
   `FEMTO_HTML_EXAMPLES` in `CMakeLists.txt` here — that one list puts it in
   the shared module and stages its page. A data file it reads at runtime goes
   in `FEMTO_HTML_EMBEDDED_FILES` next to it.
3. Copy an existing page. Keep the parts that are not decoration: the
   `enable-threads.js` script at the END of `<head>`, `createFemtoExamples()`
   with `.catch(reportLoadFailure)`, and the `#legend` / `#loading` / `#hint`
   elements the helpers write into.
4. Add a line to `index.html`, and a case to `smoke_examples.js` — it is the
   only automated check these pages have.

## viewer.js

* `new Viewer(canvas, "2d" | "3d")` — orbit camera in 3D, pan/zoom in 2D.
  `viewer.add(object, style)` returns a layer (`{color, scalar, alpha, edges,
  edgeColor, lit, visible, fill, depthTest}`), `viewer.fit(positions)`,
  `viewer.start(onFrame)`, `viewer.range = [lo, hi]`.
* `new Geometry(nodes, dim, facets, cellIds)` where `facets` is the
  size-prefixed vertex-id list the examples' `surface()` exports. Fill its
  scalars with `vertexScalar(values, stride, component | "norm")` or
  `cellScalar(values)`, and move it with `deform(displacement, scale)`.
* `cellFacets(cells, verticesPerCell, keep)` for a 2D mesh, and
  `cellBoundary(cells, verticesPerCell, keep)` for the outside of any subset of
  tetrahedra or hexahedra — this is how cut-aways, material views and slices
  are drawn. Also `centroids`, `arrows`, `bounds`.
* `loadMesh(wasm, name)` fetches a mesh staged next to the page and writes it
  into the module's filesystem at `/<name>`, which is where the examples that
  read one expect it. It is async, and `runner()` awaits its work, so a
  rebuild that needs a mesh is just `runner(async () => { await loadMesh(...) })`.
* Panel: `sliders(onChange, formats)` (mirrors every range input into
  `<output id="<id>_out">` and returns a getter), `buttonGroup(ids, onChange)`,
  `runner(work)` (coalesces slider changes and lets the panel repaint before a
  slow solve), `chart(canvas, series, opts)`, `legend(viewer, lo, hi, label)`.

## things that bit us

**iOS and Safari.** Fixes taken from the published pages
(`~/code/samuelpmish.github.io/programming/femto`):

* `enable-threads.js` must set `Cross-Origin-Embedder-Policy: require-corp`.
  WebKit does not implement `credentialless`, so with it no browser on iOS ever
  becomes cross-origin isolated and the module dies handing its shared memory
  to the workers (`DataCloneError: The object can not be cloned`). Every
  resource these pages load is same-origin, so `require-corp` costs nothing.
* Keep that script same-origin (service workers must be) and at the end of
  `<head>`. It exits early when the server already sends the headers, and it
  writes the reason into `#loading` when it cannot help (private window,
  non-secure context).
* A failed module load must say so: `.catch(reportLoadFailure)` names the
  isolation problem instead of leaving the page on "loading …".
* Touch needs a `pointerId -> [x, y]` Map, not one drag position: two fingers
  pinch about their midpoint and pan with it, and `pointercancel` has to
  release (iOS ends touches that way). `touch-action: none` on the canvas.
* `-webkit-backdrop-filter` alongside `backdrop-filter` for the panel blur.

**The rest:**

* The control panel covers the left ~332 px, so `Viewer.inset()` shifts the
  projection (perspective `m[8]`, ortho translation) and `fit()` zooms out to
  match. Anything that maps screen coordinates back to the world has to undo
  the same shift.
* To cut a mesh on a plane, keep the cells the plane actually crosses (compare
  the min and max vertex coordinate), not the cells whose centroid is within
  some slab: a graded mesh has no single thickness that suits both the
  millimeter elements and the centimeter ones. A centroid slab test also
  selects **nothing** on a uniform mesh with an even cell count, because the
  centroids then sit exactly half a cell from the plane.
* Arrows drawn inside a slab are hidden by it: give that layer
  `depthTest: false`.
* A solve over about a second belongs behind a button with a "solving …"
  label, not on a slider (`magnetostatics_3D.html` prints an estimate of how
  long the current resolution will take). Anything faster can go through
  `runner()`.
* Report the geometry the example actually built rather than the slider value
  when the mesh quantizes it — `capacitor_2D` binds its snapped `params` as
  `geometry()` for exactly this.
* Every embind object needs `delete()` before it is replaced, or the wasm heap
  grows with every slider move.
* A matrix integrand is `dot(trial, C, test)`: the **last** argument gives
  the rows, and `C` is indexed `(q, test component, test shape, trial
  component, trial shape)`. With one space on both sides only the transpose
  is at stake; with two (`lid_driven_cavity_2D`'s Q2 velocity against Q1
  pressure) the wrong order yields a matrix of the transposed size, the shape
  check does not catch it because the singleton dimensions are squeezed, and
  the assembly writes off the end of the heap.
* `wave_disk_2D` compares three rim treatments against the analytic
  whole-plane solution (a Hankel-transform table per pulse). Its lessons: the
  PML ring's cells must wind the same way as the disk's (a clockwise ring gave
  negative lumped masses and a blow-up that looked like a PML instability but
  was independent of sigma and dt); the ring's radial spacing has to match the
  disk's cell size, not the rim edge length, or the size jump reflects; the
  decay per element sigma_max h / c = 1.4 sits at the optimum for an
  explicit scheme too; and lumped P1 triangles with SSP-RK3 need
  c dt / h_min below about 0.35 (0.25 used), where Q1 quads take 0.4. The
  page's clicks come from `viewer.worldAt` (2D) and a press/release pair
  moving under 4 px, so panning still works; the diverging palette is
  `viewer.palette = "diverging"`.
* Steady problems are solved as steady problems. The cavity page used to
  time-step to a steady state and took minutes at Re = 1000; Newton on the
  coupled velocity-pressure system, with the Reynolds number raised 4x a
  stage from 100, takes about a dozen steps and a few seconds. The Jacobian
  is one unsymmetric matrix over `[u; p]` with identity rows at the Dirichlet
  dofs and the pressure pin, and `inv()` picks LU for it. Its pattern is
  built once (`from_triplets`) and the blocks are scattered into it each
  step, and the factorization is `update()`d rather than rebuilt: in wasm
  the triplet rebuild and the ordering were together half the solve time.
* An example that loads a mesh from disk works in the browser by fetching it:
  add the file to `FEMTO_HTML_MESHES` so the build stages it next to the pages,
  call `loadMesh` before constructing, and give the source an
  `#ifdef __EMSCRIPTEN__` picking `/<name>.msh` over `FEMTO_MESH_DIR`. The
  meshes are bigger than the module, so embedding them would make every page
  pay for all of them.
* The meshes themselves come from `tools/generate_example_meshes.py`, which
  builds closed surfaces and fills them with fTetWild. Read its header before
  changing a geometry constant: the examples classify elements with the same
  analytic tests, so the two files have to agree. Its own hard-won rules are
  that parts must overlap rather than meet face to face, that it runs
  single-threaded to be reproducible, and that every region's volume and every
  material boundary is checked afterwards — which is the only reason the
  face-to-face failures were noticed at all. A boundary the classification
  relies on has to be an input surface that covers the whole region it
  separates: a tab poking out past a heatsink's plate, through a wider hole or
  below its bottom edge, has no surface to follow and comes out half copper,
  half aluminum.
* `serve.py` sends `Cache-Control: no-store`, so the 3 MB module is refetched
  on every page during local development. A real host should let it cache, and
  should serve it compressed — the wasm gzips about 3.5 to 1.
