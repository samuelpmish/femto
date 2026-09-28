#!/usr/bin/env python3
"""Generate the tetrahedral meshes that the browser examples load.

    python3 -m pip install wildmeshing meshio
    python3 tools/generate_example_meshes.py [--out data/meshes] [--only NAME]

Triangulated surfaces are built here (a TO-220 package on each of three common
heatsinks, and an induction-heating work coil around a ferrite core in a box of
air) and filled with fTetWild, through the `wildmeshing` package.  Every
surface is a closed, outward-oriented shell; fTetWild keeps the tetrahedra
inside the union of them and conforms to the interfaces in between, which is
what lets the examples tell copper from epoxy from air.

The examples classify elements by the same analytic predicates used to build
the geometry, so the constants below MUST MATCH the ones in
examples/cpp/heatsink_3D.cpp and examples/cpp/magnetostatics_3D.cpp.

Lengths are millimeters in both files.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

# --------------------------------------------------------------- TO-220AB
BODY_W = 10.16      # x, package width
BODY_H = 9.9        # z, plastic body height
BODY_T = 4.6        # y, package thickness, tab included
TAB_T = 1.3         # y, copper tab
TAB_H = 15.9        # z, tab height, 6 mm of it above the plastic
HOLE_D = 3.6        # mounting hole
HOLE_Z = TAB_H - 2.8
DIE_W, DIE_T, DIE_H = 4.0, 0.6, 4.0     # silicon, soldered to the tab
DIE_Z0 = 3.0
LEAD_W, LEAD_T, LEAD_L = 0.8, 0.5, 13.0
LEAD_PITCH = 2.54
LEAD_Y0 = 2.0
SINK_OVERLAP = 0.3  # how far the tab reaches into the heatsink, to avoid coplanar faces

# --------------------------------------------------- induction coil + core
CORE_R, CORE_LEN = 8.0, 60.0            # ferrite rod
COIL_R, WIRE_R = 14.0, 2.5              # helix centerline radius, copper radius
COIL_TURNS, COIL_PITCH = 6, 8.0
AIR_XY, AIR_Z = 40.0, 50.0              # half sizes of the box of air


# ------------------------------------------------------------- primitives
# every primitive returns (vertices, triangles) of a closed shell whose
# triangles wind counterclockwise seen from outside

def box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    V = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                  [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
    F = np.array([[0, 3, 2], [0, 2, 1],        # -z
                  [4, 5, 6], [4, 6, 7],        # +z
                  [0, 1, 5], [0, 5, 4],        # -y
                  [2, 3, 7], [2, 7, 6],        # +y
                  [1, 2, 6], [1, 6, 5],        # +x
                  [3, 0, 4], [3, 4, 7]])       # -x
    return V, F


def quad_strip(ring_a, ring_b, closed=True):
    """triangles joining two equally sampled rings, a -> b outward"""
    n = len(ring_a)
    F = []
    last = n if closed else n - 1
    for i in range(last):
        j = (i + 1) % n
        F.append([i, n + i, n + j])
        F.append([i, n + j, j])
    return np.vstack([ring_a, ring_b]), np.array(F)


def plate_with_hole(x0, x1, z0, z1, y0, y1, cx, cz, r, n=48):
    """a rectangular plate in the xz-plane with a round hole through y.

    The plate is star shaped about the hole center, so a fan of quads from the
    hole to the border triangulates it; the border samples include the four
    corners so the outer wall is exactly the rectangle."""
    corners = [(x1, z1), (x0, z1), (x0, z0), (x1, z0)]
    angles = sorted(set([2 * math.pi * k / n for k in range(n)] +
                        [math.atan2(cz_ - cz, cx_ - cx) % (2 * math.pi) for cx_, cz_ in corners]))

    def border(a):
        # where the ray leaves the rectangle
        dx, dz = math.cos(a), math.sin(a)
        t = math.inf
        if dx > 1e-12: t = min(t, (x1 - cx) / dx)
        if dx < -1e-12: t = min(t, (x0 - cx) / dx)
        if dz > 1e-12: t = min(t, (z1 - cz) / dz)
        if dz < -1e-12: t = min(t, (z0 - cz) / dz)
        return cx + t * dx, cz + t * dz

    inner = [(cx + r * math.cos(a), cz + r * math.sin(a)) for a in angles]
    outer = [border(a) for a in angles]
    m = len(angles)

    V, F = [], []
    for y in (y0, y1):
        for x, z in inner: V.append([x, y, z])
        for x, z in outer: V.append([x, y, z])
    back, front = 0, 2 * m          # vertex offsets of the y0 and y1 faces

    for i in range(m):
        j = (i + 1) % m
        # y0 face, outward normal -y
        F += [[back + i, back + m + i, back + m + j], [back + i, back + m + j, back + j]]
        # y1 face, outward normal +y
        F += [[front + i, front + m + j, front + m + i], [front + i, front + j, front + m + j]]
        # outer wall (+ outward), hole wall (inward)
        F += [[back + m + i, front + m + i, front + m + j], [back + m + i, front + m + j, back + m + j]]
        F += [[back + i, back + j, front + j], [back + i, front + j, front + i]]
    return np.array(V, float), np.array(F)


def cylinder(radius, z0, z1, n=48, center=(0.0, 0.0)):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ring = np.stack([center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)], axis=1)
    V = np.vstack([np.column_stack([ring, np.full(n, z0)]), np.column_stack([ring, np.full(n, z1)])])
    F = [[i, n + (i + 1) % n, n + i] for i in range(n)] + [[i, (i + 1) % n, n + (i + 1) % n] for i in range(n)]
    lo, hi = len(V), len(V) + 1
    V = np.vstack([V, [[center[0], center[1], z0], [center[0], center[1], z1]]])
    F += [[lo, (i + 1) % n, i] for i in range(n)]                       # -z cap
    F += [[hi, n + i, n + (i + 1) % n] for i in range(n)]               # +z cap
    return V, np.array(F)


def tube(points, radius, n=24):
    """a capped tube of the given radius around an open polyline, swept with a
    parallel-transported frame so the cross sections do not spin"""
    P = np.asarray(points, float)
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1)[:, None]

    # parallel transport: rotate the previous normal onto the new tangent
    normals = np.zeros_like(P)
    seed = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(seed, T[0])) > 0.9: seed = np.array([1.0, 0.0, 0.0])
    normals[0] = np.cross(T[0], seed)
    normals[0] /= np.linalg.norm(normals[0])
    for i in range(1, len(P)):
        v = np.cross(T[i - 1], T[i])
        s = np.linalg.norm(v)
        if s < 1e-12:
            normals[i] = normals[i - 1]
        else:
            axis, angle = v / s, math.atan2(s, np.dot(T[i - 1], T[i]))
            c, sn = math.cos(angle), math.sin(angle)
            normals[i] = (normals[i - 1] * c + np.cross(axis, normals[i - 1]) * sn +
                          axis * np.dot(axis, normals[i - 1]) * (1 - c))
        normals[i] -= T[i] * np.dot(T[i], normals[i])
        normals[i] /= np.linalg.norm(normals[i])
    binormals = np.cross(T, normals)

    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    V = np.concatenate([P[i][None, :] + radius * (np.cos(a)[:, None] * normals[i] + np.sin(a)[:, None] * binormals[i])
                        for i in range(len(P))])
    F = []
    for i in range(len(P) - 1):
        for k in range(n):
            j = (k + 1) % n
            F.append([i * n + k, (i + 1) * n + j, (i + 1) * n + k])
            F.append([i * n + k, i * n + j, (i + 1) * n + j])
    lo, hi = len(V), len(V) + 1
    V = np.vstack([V, P[0][None, :], P[-1][None, :]])
    F += [[lo, k, (k + 1) % n] for k in range(n)]
    F += [[hi, (len(P) - 1) * n + (k + 1) % n, (len(P) - 1) * n + k] for k in range(n)]
    return V, np.array(F)


def helix_points(radius, pitch, turns, samples_per_turn=64):
    phi = np.linspace(-math.pi * turns, math.pi * turns, int(turns * samples_per_turn) + 1)
    return np.stack([radius * np.cos(phi), radius * np.sin(phi), pitch * phi / (2 * math.pi)], axis=1)


def combine(parts):
    V, F, offset = [], [], 0
    for v, f in parts:
        V.append(v)
        F.append(np.asarray(f) + offset)
        offset += len(v)
    return np.vstack(V), np.vstack(F).astype(np.int32)


def signed_volume(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return float(np.sum(np.einsum('ij,ij->i', a, np.cross(b, c))) / 6.0)


# ------------------------------------------------------------- assemblies
def to220_parts(sink_overlap=0.0):
    """the package itself: tab (with its mounting hole), plastic body, die, leads.

    Parts overlap on purpose wherever two of them would otherwise share a face:
    coplanar input faces are where fTetWild loses material.  With the plastic
    and the die sitting exactly on the tab's front face, the largest assembly
    came out with the tab 13% thin at any resolution or envelope; sinking them
    0.3 mm into it fixes that.  The classification planes are unaffected, since
    each material's own face is still there -- the slivers that end up inside
    the tab simply count as copper.

    With a heatsink behind it the tab is likewise extended `sink_overlap` into
    the sink."""
    w = BODY_W / 2
    bite = 0.3
    parts = [plate_with_hole(-w, w, 0.0, TAB_H, -sink_overlap, TAB_T, 0.0, HOLE_Z, HOLE_D / 2),
             box((-w, TAB_T - bite, 0.0), (w, BODY_T, BODY_H)),
             box((-DIE_W / 2, TAB_T - bite, DIE_Z0), (DIE_W / 2, TAB_T + DIE_T, DIE_Z0 + DIE_H))]
    for k in (-1, 0, 1):
        x = k * LEAD_PITCH
        parts.append(box((x - LEAD_W / 2, LEAD_Y0, -LEAD_L), (x + LEAD_W / 2, LEAD_Y0 + LEAD_T, bite)))
    return parts


def plate_area(width, height, hole_r):
    return width * height - math.pi * hole_r ** 2


def finned_sink(half_width, z0, z1, base, fins, fin_thickness, depth):
    """a clip-on extrusion: a base plate on the back of the tab (y from -base to
    0) with `fins` slabs standing off it in -y, running its full height.

    The fins reach a little way into the base rather than meeting it face to
    face; see to220_parts for why."""
    parts = [box((-half_width, -base, z0), (half_width, 0.0, z1))]
    pitch = (2 * half_width - fin_thickness) / (fins - 1)
    for k in range(fins):
        x = -half_width + k * pitch
        parts.append(box((x, -depth, z0), (x + fin_thickness, -base + 0.4, z1)))
    volume = 2 * half_width * base * (z1 - z0) + fins * fin_thickness * (depth - base) * (z1 - z0)
    return parts, volume


def channel_sink(width=19.0, height=20.0, depth=17.0, wall=1.6, fingers=5, slot=1.8):
    """the small bolt-on heatsink: a U of sheet aluminum whose back plate bolts
    to the tab, its two wings cut into fingers by slots from their free edge.

    Its hole is a little smaller than the tab's, so the two hole walls are not
    coincident surfaces and, more to the point, so that the tab never pokes
    through the plate: with a wider plate hole the ring of tab between the two
    radii had no interface surface at y = 0 to follow, and a hundred elements
    around the hole came out half copper, half aluminum."""
    hole_r = 0.5 * HOLE_D - 0.2
    z0 = -0.5         # the plate covers the whole tab: see check_interfaces for why
    parts = [plate_with_hole(-width / 2, width / 2, z0, z0 + height, -wall, 0.0, 0.0, HOLE_Z, hole_r)]
    finger = (height - (fingers - 1) * slot) / fingers
    for side in (-1.0, 1.0):
        x0, x1 = side * (width / 2 - wall), side * width / 2
        for k in range(fingers):
            z = z0 + k * (finger + slot)
            parts.append(box((min(x0, x1), -depth, z), (max(x0, x1), -wall + 0.4, z + finger)))
    volume = plate_area(width, height, hole_r) * wall + 2 * fingers * finger * wall * (depth - wall)
    return parts, volume


def multiwatt_sink(width=41.9, length=63.5, base=6.0, fins=3, fin_thickness=2.6,
                   fin_height=17.0, channel=6.5):
    """the large extrusion: a thick base plate whose front face carries two
    banks of tall fins with a channel between them, so the package sits down in
    the channel with fins to either side of it (Aavid 6400BG, 42 x 63 x 25 mm).

    The fins stand proud of the mounting face, which is why the examples call
    anything wider than the package a heatsink rather than going by y alone."""
    z0 = -14.0                                  # centers the extrusion on the package
    parts = [box((-width / 2, -base, z0), (width / 2, 0.0, z0 + length))]
    pitch = (width / 2 - channel - fin_thickness) / (fins - 1)
    for side in (-1.0, 1.0):
        for k in range(fins):
            x0 = side * (channel + k * pitch)
            x1 = x0 + side * fin_thickness
            parts.append(box((min(x0, x1), -base + 0.4, z0), (max(x0, x1), fin_height, z0 + length)))
    volume = width * base * length + 2 * fins * fin_thickness * fin_height * length
    return parts, volume


# each design as (parts, union volume, coarsest element).  The element cap is
# set from the design's thinnest wall: fTetWild holds surfaces to its envelope
# but erodes a feature it cannot fit an element or two across, which is how the
# multiwatt extrusion ate a fifth of the 1.6 mm copper tab.
HEATSINKS = {
    # bare package: the tab is the only thing losing heat
    "to220_bare": ([], 0.0, 2.0),
    # a clip-on extrusion, about 25 mm square
    "to220_clip": (*finned_sink(half_width=12.5, z0=-1.0, z1=24.0, base=3.0, fins=5,
                                fin_thickness=1.6, depth=14.0), 1.1),
    # the small bolt-on, and the large extrusion a multiwatt package would use
    "to220_channel": (*channel_sink(), 1.1),
    "to220_multiwatt": (*multiwatt_sink(), 2.5),
}


def package_sizing(cap):
    """small elements through the package, where the die and the tab are, and
    up to `cap` out in the heatsink, which is nearly isothermal anyway"""
    def size(p):
        d = max(abs(p[0]) - 0.5 * BODY_W, -p[1], p[1] - BODY_T,
                -p[2] - LEAD_L, p[2] - TAB_H)
        return min(cap, 0.8 + 0.25 * max(0.0, d))
    return size


def coil_parts():
    return [box((-AIR_XY, -AIR_XY, -AIR_Z), (AIR_XY, AIR_XY, AIR_Z)),
            cylinder(CORE_R, -CORE_LEN / 2, CORE_LEN / 2, n=48),
            tube(helix_points(COIL_R, COIL_PITCH, COIL_TURNS), WIRE_R, n=24)]


# --------------------------------------------------------------- pipeline
def tetrahedralize(parts, envelope, edge_length_r, name, threads=1, sizing=None):
    import wildmeshing as wm

    for i, (v, f) in enumerate(parts):
        if signed_volume(v, f) <= 0:
            raise SystemExit(f"{name}: part {i} is inside out ({signed_volume(v, f):.3f} mm^3)")
    V, F = combine(parts)

    # fTetWild takes its envelope relative to the bounding box diagonal, but
    # what matters is its size against the thinnest wall: at the default it
    # shaved 15% off the 1.3 mm copper tab in the largest assembly, whose box
    # is four times the package's.  Ask in millimeters and convert
    diagonal = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    epsilon = envelope / diagonal

    # one thread by default: fTetWild's parallel passes are order dependent, so
    # the result varies run to run, and a bad roll can drop a whole part (the
    # copper tab went missing once, where its back face is coplanar with the
    # heatsink's mounting face).  check_volumes below is what caught it
    mesher = wm.Tetrahedralizer(epsilon=epsilon, edge_length_r=edge_length_r, stop_quality=8,
                                coarsen=True, max_threads=threads)
    mesher.set_mesh(V, F)
    if sizing is not None:
        mesher.set_sizing_field_from_func(sizing)
    mesher.tetrahedralize()
    VT, T, _ = mesher.get_tet_mesh()     # the third return is a per-tet tag fTetWild leaves unset
    return np.asarray(VT), np.asarray(T, dtype=np.int32)


def tet_volumes(V, T):
    return np.abs(np.einsum('ij,ij->i', V[T[:, 1]] - V[T[:, 0]],
                            np.cross(V[T[:, 2]] - V[T[:, 0]], V[T[:, 3]] - V[T[:, 0]]))) / 6.0


def check_volumes(name, V, T, regions, expected, tol=0.06):
    """every region has to hold the volume the geometry says it should: this is
    what tells us fTetWild kept each part and conformed to the interfaces.

    The tolerance is loose because surfaces are free to move inside fTetWild's
    envelope; it is here to catch a part going missing, which happens."""
    vol = tet_volumes(V, T)
    centers = V[T].mean(axis=1)
    for label, want in expected.items():
        got = vol[regions[label](centers)].sum()
        if abs(got - want) > tol * max(want, 1e-9):
            raise SystemExit(f"{name}: the {label} region holds {got:.1f} mm^3, expected {want:.1f} "
                             f"-- rerun, or lower --envelope")


def check_interfaces(name, V, T, regions, tol=0.0002):
    """no element may span a material boundary.  The examples give every
    element the material at its centroid, so an element that crosses a
    boundary smears the two materials together -- which the volume check
    cannot see, since the volume involved is tiny.  Each element is sampled
    halfway from its centroid to each vertex, which stays clear of the wobble
    fTetWild's envelope allows on the surface itself.

    What this has caught so far: the tab poking out past a heatsink's mounting
    face, where there is then no surface at y = 0 for the mesh to follow --
    once through a plate hole wider than the tab's, once below a plate that
    started above the tab's bottom edge.  A heatsink face has to cover the
    tab's whole footprint, overlap slab included."""
    centers = V[T].mean(axis=1)
    label = np.full(len(T), -1)
    for k, predicate in enumerate(regions.values()):
        label[predicate(centers)] = k
    crossing = np.zeros(len(T), bool)
    for j in range(4):
        probe = 0.5 * (centers + V[T[:, j]])
        probe_label = np.full(len(T), -1)
        for k, predicate in enumerate(regions.values()):
            probe_label[predicate(probe)] = k
        crossing |= probe_label != label
    if crossing.mean() > tol:
        where = centers[crossing]
        raise SystemExit(f"{name}: {crossing.sum()} elements ({100 * crossing.mean():.2f}%) straddle a material "
                         f"boundary, around x {where[:, 0].min():.1f}..{where[:, 0].max():.1f}, "
                         f"y {where[:, 1].min():.1f}..{where[:, 1].max():.1f}, z {where[:, 2].min():.1f}..{where[:, 2].max():.1f}")
    return int(crossing.sum())


def report(name, V, T, regions):
    vol = tet_volumes(V, T)
    print(f"{name}: {len(T):,} tets, {len(V):,} nodes, {vol.sum():,.0f} mm^3, "
          f"edge sizes {np.cbrt(vol).min():.2f} .. {np.cbrt(vol).max():.2f} mm")
    centers = V[T].mean(axis=1)
    for label, predicate in regions.items():
        keep = predicate(centers)
        print(f"    {label:<10s} {keep.sum():>7,d} tets  {vol[keep].sum():>10,.1f} mm^3")


def heatsink_expected(sink_volume):
    """what each region should hold.  The heatsinks report their own union
    volume (their parts overlap on purpose), and the tab pokes SINK_OVERLAP
    into the sink, so that slab of it counts as aluminum."""
    die = DIE_W * DIE_T * DIE_H
    tab_area = signed_volume(*plate_with_hole(-BODY_W / 2, BODY_W / 2, 0.0, TAB_H, 0.0, 1.0,
                                              0.0, HOLE_Z, HOLE_D / 2))
    return {"sink": sink_volume + (tab_area * SINK_OVERLAP if sink_volume else 0.0),
            "tab": tab_area * TAB_T,
            "die": die,
            "leads": 3 * LEAD_W * LEAD_T * LEAD_L,
            "plastic": BODY_W * (BODY_T - TAB_T) * BODY_H - die}


def heatsink_regions():
    """which material an element is made of, from its centroid.  Anything
    behind the tab or wider than the package is heatsink: the multiwatt fins
    stand proud of the mounting face, on either side of the package."""
    def sink(c):
        return (c[:, 1] < 0.0) | (np.abs(c[:, 0]) > 0.6 * BODY_W)

    def die(c):
        return ((np.abs(c[:, 0]) < DIE_W / 2) & (c[:, 1] > TAB_T) & (c[:, 1] < TAB_T + DIE_T) &
                (c[:, 2] > DIE_Z0) & (c[:, 2] < DIE_Z0 + DIE_H))
    return {"sink": sink,
            "tab": lambda c: ~sink(c) & (c[:, 1] < TAB_T),
            "die": lambda c: ~sink(c) & die(c),
            "leads": lambda c: ~sink(c) & (c[:, 1] >= TAB_T) & (c[:, 2] < 0.0),
            "plastic": lambda c: ~sink(c) & (c[:, 1] >= TAB_T) & (c[:, 2] >= 0.0) & ~die(c)}


def coil_sizing():
    """elements sized for the physics rather than for the geometry: the ferrite
    needs several elements across it before lowest-order edge elements resolve
    a 1000:1 permeability jump, and the air far from the coil needs none"""
    def size(p):
        r = math.hypot(p[0], p[1])
        to_core = max(r - CORE_R, abs(p[2]) - CORE_LEN / 2)
        to_coil = helix_distance(p[0], p[1], p[2]) - WIRE_R
        d = max(0.0, min(to_core, to_coil))
        return min(10.0, 2.2 + 0.7 * d)
    return size


def helix_distance(x, y, z):
    """distance to the coil's centerline; the same test the example uses"""
    phi = math.atan2(y, x)
    r = math.hypot(x, y)
    best = math.inf
    for k in range(-COIL_TURNS, COIL_TURNS + 1):
        angle = phi + 2 * math.pi * k
        if abs(angle) > math.pi * COIL_TURNS:
            continue
        best = min(best, math.hypot(r - COIL_R, z - COIL_PITCH * angle / (2 * math.pi)))
    return best


def coil_expected():
    return {"core": math.pi * CORE_R ** 2 * CORE_LEN,
            "coil": signed_volume(*tube(helix_points(COIL_R, COIL_PITCH, COIL_TURNS), WIRE_R, n=24)),
            "air": 2 * AIR_XY * 2 * AIR_XY * 2 * AIR_Z
                   - math.pi * CORE_R ** 2 * CORE_LEN
                   - signed_volume(*tube(helix_points(COIL_R, COIL_PITCH, COIL_TURNS), WIRE_R, n=24))}


def coil_regions():
    def core(c):
        return (np.hypot(c[:, 0], c[:, 1]) < CORE_R) & (np.abs(c[:, 2]) < CORE_LEN / 2)

    def coil(c):
        # distance to the helix: the wire crosses the point's own azimuth once
        # per turn, so only the few turns nearest in z can be closest
        phi = np.arctan2(c[:, 1], c[:, 0])
        r = np.hypot(c[:, 0], c[:, 1])
        best = np.full(len(c), np.inf)
        half = COIL_TURNS / 2.0
        for k in range(-int(math.ceil(half)) - 1, int(math.ceil(half)) + 2):
            angle = phi + 2 * math.pi * k
            inside = np.abs(angle) <= math.pi * COIL_TURNS
            z = COIL_PITCH * angle / (2 * math.pi)
            d = np.where(inside, np.hypot(r - COIL_R, c[:, 2] - z), np.inf)
            best = np.minimum(best, d)
        return best < WIRE_R
    return {"core": core, "coil": coil, "air": lambda c: ~core(c) & ~coil(c)}


def main():
    here = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=here / "data" / "meshes")
    parser.add_argument("--only", help="generate just this mesh")
    parser.add_argument("--envelope", type=float, default=0.025, help="fTetWild envelope, in mm")
    parser.add_argument("--edge-length", type=float, default=0.05,
                        help="fTetWild target edge length, relative to the diagonal; a bound, the sizing fields do the rest")
    parser.add_argument("--ascii", action="store_true", help="write ascii instead of binary gmsh 2.2")
    parser.add_argument("--threads", type=int, default=1, help="fTetWild threads; >1 is faster but not reproducible")
    args = parser.parse_args()

    import meshio

    jobs = {name: (to220_parts(SINK_OVERLAP if parts else 0.0) + parts, heatsink_regions(), heatsink_expected(volume))
            for name, (parts, volume, _) in HEATSINKS.items()}
    jobs["induction_coil"] = (coil_parts(), coil_regions(), coil_expected())
    sizings = {name: package_sizing(cap) for name, (_, _, cap) in HEATSINKS.items()}
    sizings["induction_coil"] = coil_sizing()

    for name, (parts, regions, expected) in jobs.items():
        if args.only and args.only != name:
            continue
        # the coil is all smooth surfaces in a big box of air; the packages are
        # thin walls in a small one
        # the coil is all smooth surfaces in a big box of air; the packages are
        # thin walls in a small one.  The heatsinks' element sizes come from
        # their own sizing field, so the target length here is only a bound
        envelope, edge_length = (0.15, 0.05) if name == "induction_coil" else (args.envelope, args.edge_length)
        V, T = tetrahedralize(parts, envelope, edge_length, name, args.threads, sizings.get(name))
        check_volumes(name, V, T, regions, expected)
        crossing = check_interfaces(name, V, T, regions)
        report(name, V, T, regions)
        print(f"    {crossing} elements straddle a material boundary")
        path = args.out / f"{name}.msh"
        meshio.write(path, meshio.Mesh(V, [("tetra", T)]), file_format="gmsh22", binary=not args.ascii)
        print(f"    wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
