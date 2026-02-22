#!/usr/bin/env python3
"""Decimate Vention table STL from ~1.2M faces to ~10k faces using pymeshlab."""
import pathlib

import pymeshlab

SRC = pathlib.Path("/home/rongxuan_zhou/Downloads/VentionAssembly_123900_v13.stl")
DST = pathlib.Path(__file__).parent.parent / "models" / "vention_table_10k.stl"
TARGET_FACES = 10_000


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(
            f"Source STL not found: {SRC}\nPlease update the SRC path in this script."
        )

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(SRC))

    original_faces = ms.current_mesh().face_number()
    print(f"原始面数: {original_faces:,}")

    if original_faces <= TARGET_FACES:
        print(
            f"Warning: original face count ({original_faces:,}) is already <= "
            f"target ({TARGET_FACES:,}). Skipping decimation."
        )
    else:
        # Quadric Edge Collapse Decimation — MeshLab's most robust reducer
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=TARGET_FACES,
            preservetopology=False,
            preservenormal=True,
            qualitythr=0.3,
        )

    result_faces = ms.current_mesh().face_number()
    print(f"精简后面数: {result_faces:,}")

    DST.parent.mkdir(parents=True, exist_ok=True)
    ms.save_current_mesh(str(DST))
    print(f"已保存: {DST}")


if __name__ == "__main__":
    main()
