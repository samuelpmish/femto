#!/usr/bin/env python3
"""Demote mesh elements to first order and write binary Gmsh 2.2."""

from __future__ import annotations

import argparse
from pathlib import Path


FIRST_ORDER_CELLS = (
    ("hexahedron", 8),
    ("triangle", 3),
    ("pyramid", 5),
    ("wedge", 6),
    ("tetra", 4),
    ("quad", 4),
    ("line", 2),
    ("vertex", 1),
)


def first_order_cell_type(cell_type: str) -> tuple[str, int]:
    for base_type, num_vertices in FIRST_ORDER_CELLS:
        suffix = cell_type[len(base_type) :]
        if cell_type == base_type or (
            cell_type.startswith(base_type) and suffix.isdigit()
        ):
            return base_type, num_vertices

    raise ValueError(f"unsupported cell type: {cell_type}")


def remap_point_sets(point_sets, old_to_new):
    import numpy as np

    remapped = {}
    for name, old_ids in point_sets.items():
        old_ids = np.asarray(old_ids, dtype=np.int64)
        keep = old_to_new[old_ids] >= 0
        remapped[name] = old_to_new[old_ids[keep]]
    return remapped


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read any meshio-supported mesh, demote high-order cells to "
            "first-order cells, prune unused points, and write binary Gmsh v2.2."
        )
    )
    parser.add_argument("input", type=Path, help="input mesh file")
    parser.add_argument("output", type=Path, help="output .msh file")
    args = parser.parse_args()

    try:
        import meshio
        import numpy as np
    except ImportError:
        parser.error("meshio is not installed; run `python -m pip install meshio`")

    mesh = meshio.read(args.input)

    new_cells = []
    demoted_blocks = 0
    original_cell_count = 0
    for block in mesh.cells:
        try:
            base_type, num_vertices = first_order_cell_type(block.type)
        except ValueError as error:
            parser.error(str(error))

        data = np.asarray(block.data)
        if data.ndim != 2 or data.shape[1] < num_vertices:
            parser.error(
                f"cell block {block.type} has shape {data.shape}; expected at "
                f"least {num_vertices} nodes per element"
            )

        if block.type != base_type:
            demoted_blocks += 1

        original_cell_count += len(data)
        new_cells.append(
            meshio.CellBlock(
                base_type,
                data[:, :num_vertices].copy(),
                tags=getattr(block, "tags", None),
            )
        )

    if not new_cells:
        parser.error("input mesh contains no cell blocks")

    used_points = np.unique(
        np.concatenate([block.data.reshape(-1) for block in new_cells])
    )
    old_to_new = np.full(len(mesh.points), -1, dtype=np.int64)
    old_to_new[used_points] = np.arange(len(used_points), dtype=np.int64)

    compact_cells = [
        meshio.CellBlock(
            block.type,
            old_to_new[block.data],
            tags=getattr(block, "tags", None),
        )
        for block in new_cells
    ]

    point_data = {
        name: np.asarray(values)[used_points] for name, values in mesh.point_data.items()
    }
    point_sets = remap_point_sets(mesh.point_sets, old_to_new)

    if mesh.gmsh_periodic:
        print("warning: dropping gmsh_periodic data after point renumbering")

    output_mesh = meshio.Mesh(
        points=mesh.points[used_points],
        cells=compact_cells,
        point_data=point_data,
        cell_data=mesh.cell_data,
        field_data=mesh.field_data,
        point_sets=point_sets,
        cell_sets=mesh.cell_sets,
        info=mesh.info,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(args.output, output_mesh, file_format="gmsh22", binary=True)

    print(
        f"wrote {args.output}: {len(mesh.points)} -> {len(used_points)} points, "
        f"{original_cell_count} cells, demoted {demoted_blocks} cell blocks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
