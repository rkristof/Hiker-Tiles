#!/usr/bin/env python3
"""Preprocess classified natural polygons for low-zoom tile generation."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

LOGGER = logging.getLogger("preprocess_natural")

MAX_WGS84_METERS_PER_DEGREE = 112000.0

OSMIUM_FILTERS = (
    "wr/landuse=forest,grass,farmland,meadow,orchard,vineyard,farmyard,"
    "greenhouse_horticulture,allotments,quarry",
    "wr/natural=wood,grassland,glacier,bare_rock,sand,heath,scrub,scree,"
    "shingle,wetland,fell,beach",
    "wr/wetland=swamp,bog,wet_meadow,marsh",
)

NATURAL_CLASSIFICATION_SQL = """
SELECT geometry,
    CASE
        WHEN landuse IN ('forest', 'grass', 'farmland') THEN landuse
        WHEN natural = 'wood' THEN 'forest'
        WHEN natural = 'grassland' OR landuse = 'meadow' THEN 'grass'
        WHEN landuse IN ('orchard', 'vineyard', 'farmyard', 'greenhouse_horticulture', 'allotments') THEN 'farmland'
        WHEN landuse = 'quarry' THEN 'quarry'
        WHEN natural = 'glacier' THEN 'glacier'
        WHEN natural = 'bare_rock' THEN 'bare_rock'
        WHEN natural IN ('sand', 'beach') THEN 'sand'
        WHEN natural = 'heath' THEN 'heath'
        WHEN natural = 'scrub' THEN 'scrub'
        WHEN natural = 'scree' THEN 'scree'
        WHEN natural = 'shingle' THEN 'shingle'
        WHEN HSTORE_GET_VALUE(other_tags, 'wetland') IN ('swamp', 'bog', 'wet_meadow', 'marsh') THEN 'marsh'
        WHEN natural = 'wetland' THEN 'marsh'
        WHEN natural = 'fell' THEN 'grass'
    END AS kind
FROM multipolygons
WHERE (
        landuse IN ('forest', 'grass', 'farmland', 'meadow', 'orchard', 'vineyard', 'farmyard', 'greenhouse_horticulture', 'allotments', 'quarry')
         OR natural IN ('wood', 'grassland', 'glacier', 'bare_rock', 'sand', 'heath', 'scrub', 'scree', 'shingle', 'wetland', 'fell', 'beach')
            OR HSTORE_GET_VALUE(other_tags, 'wetland') IN ('swamp', 'bog', 'wet_meadow', 'marsh')
)
    -- Use the WGS84 bounding-box area as a cheap conservative prefilter. The
    -- exact projected-area check below remains authoritative for retention.
    AND ST_Area(ST_Envelope(geometry))
        * {max_wgs84_meters_per_degree}
        * {max_wgs84_meters_per_degree} >= {min_area_m2}
    AND ST_Area(ST_Transform(geometry, 3035)) >= {min_area_m2}
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract, classify, dissolve, and export low-zoom natural polygons."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("tiles-latest.osm.pbf"),
        help="Input OSM PBF file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("natural-low.geojsonseq"),
        help="Final WGS84 GeoJSONSeq output for Tilemaker.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path(".natural-preprocess"),
        help="Directory for intermediate files.",
    )
    parser.add_argument(
        "--nb-squarish-tiles",
        type=int,
        default=128,
        help="Approximate number of spatial dissolve tiles.",
    )
    parser.add_argument(
        "--gridsize-meters",
        type=float,
        default=300.0,
        help="Coordinate precision grid used during dissolve, in meters.",
    )
    parser.add_argument(
        "--min-area-m2",
        type=float,
        default=250000.0,
        help="Minimum source polygon area retained before dissolve, in square meters.",
    )
    parser.add_argument(
        "--nb-parallel",
        type=int,
        default=0,
        help="GeoFileOps workers; zero uses its default CPU-based setting.",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=-1,
        help="GeoFileOps indicative batch size; -1 selects automatically.",
    )
    return parser.parse_args()


@contextmanager
def timed_step(name: str) -> Iterator[None]:
    started = time.perf_counter()
    LOGGER.info("START %s", name)
    try:
        yield
    finally:
        LOGGER.info("END %s (%.1fs)", name, time.perf_counter() - started)


def run_command(command: list[str]) -> None:
    LOGGER.info("COMMAND %s", " ".join(command))
    subprocess.run(command, check=True)


def ensure_tools() -> None:
    for tool in ("osmium", "ogr2ogr"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"Required executable not found: {tool}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ensure_tools()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input file does not exist: {args.input}")
    if args.nb_squarish_tiles < 1:
        raise ValueError("--nb-squarish-tiles must be at least 1")
    if args.gridsize_meters < 0:
        raise ValueError("--gridsize-meters cannot be negative")
    if args.min_area_m2 < 0:
        raise ValueError("--min-area-m2 cannot be negative")

    args.workdir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    filtered_pbf = args.workdir / "natural-filtered.osm.pbf"
    filter_file = args.workdir / "natural-filters.txt"
    valid_gpkg = args.workdir / "natural-valid.gpkg"
    dissolved_gpkg = args.workdir / "natural-dissolved.gpkg"

    filter_file.write_text("\n".join(OSMIUM_FILTERS) + "\n", encoding="utf-8")

    with timed_step("extract natural OSM objects"):
        run_command(
            [
                "osmium",
                "tags-filter",
                f"--expressions={filter_file}",
                str(args.input),
                "-o",
                str(filtered_pbf),
                "-O",
            ]
        )

    with timed_step("classify, filter, and convert to EPSG:3035 GeoPackage"):
        run_command(
            [
                "ogr2ogr",
                "-f",
                "GPKG",
                str(valid_gpkg),
                str(filtered_pbf),
                "-dialect",
                "SQLite",
                "-sql",
                NATURAL_CLASSIFICATION_SQL.format(
                    min_area_m2=f"{args.min_area_m2:.12g}",
                    max_wgs84_meters_per_degree=f"{MAX_WGS84_METERS_PER_DEGREE:.12g}",
                ),
                "-nln",
                "natural",
                "-nlt",
                "PROMOTE_TO_MULTI",
                "-dim",
                "XY",
                "-s_srs",
                "EPSG:4326",
                "-t_srs",
                "EPSG:3035",
                "-makevalid",
                "-lco",
                "SPATIAL_INDEX=NO",
                "-gt",
                "1000000",
                "-overwrite",
            ]
        )

    import geofileops as gfo

    with timed_step("dissolve polygons by kind with GeoFileOps"):
        dissolve_kwargs: dict[str, Any] = {
            "input_path": str(valid_gpkg),
            "output_path": str(dissolved_gpkg),
            "input_layer": "natural",
            "output_layer": "natural_low",
            "explodecollections": False,
            "groupby_columns": ["kind"],
            "nb_squarish_tiles": args.nb_squarish_tiles,
            "gridsize": args.gridsize_meters,
            "batchsize": args.batchsize,
            "force": True,
        }
        if args.nb_parallel > 0:
            dissolve_kwargs["nb_parallel"] = args.nb_parallel
        dissolve_gridsizes = [args.gridsize_meters]
        if args.gridsize_meters > 0:
            dissolve_gridsizes.extend(
                args.gridsize_meters / divisor for divisor in (2, 4, 10, 30)
            )
        for grid_index, gridsize in enumerate(dissolve_gridsizes):
            dissolve_kwargs["gridsize"] = gridsize
            try:
                gfo.dissolve(**dissolve_kwargs)
            except RuntimeError as error:
                is_last_grid = grid_index == len(dissolve_gridsizes) - 1
                if is_last_grid or "TopologyException" not in str(error):
                    raise
                LOGGER.warning(
                    "GeoFileOps dissolve failed with %.g m grid; retrying "
                    "with %.g m grid",
                    gridsize,
                    dissolve_gridsizes[grid_index + 1],
                )
            else:
                break

    with timed_step("merge dissolved polygons and export as WGS84 GeoJSONSeq"):
        run_command(
            [
                "ogr2ogr",
                "-f",
                "GeoJSONSeq",
                str(args.output),
                str(dissolved_gpkg),
                "-dialect",
                "SQLite",
                "-sql",
                "SELECT ST_Union(geom) AS geometry, kind "
                "FROM natural_low GROUP BY kind",
                "-nln",
                "natural_low",
                "-nlt",
                "POLYGON",
                "-explodecollections",
                "-t_srs",
                "EPSG:4326",
                "-lco",
                "RS=NO",
                "-lco",
                "COORDINATE_PRECISION=6",
                "-overwrite",
            ]
        )

    LOGGER.info("Natural preprocessing complete: %s", args.output)


if __name__ == "__main__":
    main()
