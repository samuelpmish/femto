#!/usr/bin/env python3
"""Write a tetrahedra-only copy of a mesh as binary Gmsh 2.2."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


def is_tetra_type(cell_type: str) -> bool:
    return cell_type.startswith("tetra")


def kept_cell_indices(cells: Iterable[object]) -> list[int]:
    return [i for i, block in enumerate(cells) if is_tetra_type(block.type)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read any meshio-supported mesh, keep only tetrahedral element "
            "blocks, and write binary Gmsh v2.2."
        )
    )
    parser.add_argument("input", type=Path, help="input mesh file")
    parser.add_argument("output", type=Path, help="output .msh file")
    args = parser.parse_args()

    try:
        import meshio
    except ImportError:
        parser.error("meshio is not installed; run `python -m pip install meshio`")

    mesh = meshio.read(args.input)
    keep = kept_cell_indices(mesh.cells)

    if not keep:
        found = ", ".join(block.type for block in mesh.cells) or "no cell blocks"
        parser.error(f"no tetrahedral cell blocks found; input contains: {found}")

    original_cells = sum(len(block.data) for block in mesh.cells)
    original_blocks = len(mesh.cells)

    mesh.cells = [mesh.cells[i] for i in keep]
    mesh.cell_data = {
        name: [blocks[i] for i in keep] for name, blocks in mesh.cell_data.items()
    }
    mesh.cell_sets = {
        name: [blocks[i] for i in keep] for name, blocks in mesh.cell_sets.items()
    }

    kept_cells = sum(len(block.data) for block in mesh.cells)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(args.output, mesh, file_format="gmsh22", binary=True)

    print(
        f"wrote {args.output}: kept {kept_cells}/{original_cells} cells "
        f"from {len(mesh.cells)}/{original_blocks} cell blocks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
