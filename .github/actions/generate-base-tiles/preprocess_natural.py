#!/usr/bin/env python3
"""Preprocess classified natural polygons for low-zoom tile generation."""

from __future__ import annotations

import argparse
import logging
import math
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from osgeo import gdal, ogr, osr

LOGGER = logging.getLogger("preprocess_natural")

OSMIUM_FILTERS = (
    "wr/landuse=forest,grass,farmland,meadow,orchard,vineyard,farmyard,"
    "greenhouse_horticulture,allotments,quarry",
    "wr/natural=wood,grassland,glacier,bare_rock,sand,heath,scrub,scree,"
    "shingle,wetland,fell,beach",
    "wr/wetland=swamp,bog,wet_meadow,marsh",
)

NATURAL_PRIORITY = (
    "grass",
    "farmland",
    "marsh",
    "bog",
    "wet_meadow",
    "swamp",
    "heath",
    "scrub",
    "sand",
    "shingle",
    "quarry",
    "bare_rock",
    "scree",
    "glacier",
    "forest",
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
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract, classify, rasterize, and export low-zoom natural polygons."
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
        "--cell-size-meters",
        type=float,
        default=1000.0,
        help="Side length of the coarse projected raster cells, in meters.",
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


def polygonize_natural_polygons(
    input_path: Path,
    output_path: Path,
    raster_path: Path,
    cell_size_meters: float,
) -> None:
    """Rasterize each natural class and polygonize its occupied cells."""
    source = ogr.Open(str(input_path))
    if source is None:
        raise RuntimeError(f"Could not open {input_path}")
    source_layer = source.GetLayerByName("natural")
    if source_layer is None:
        raise RuntimeError(f"Layer natural not found in {input_path}")

    min_x, max_x, min_y, max_y = source_layer.GetExtent()
    origin_x = math.floor(min_x / cell_size_meters) * cell_size_meters
    origin_y = math.ceil(max_y / cell_size_meters) * cell_size_meters
    width = max(1, math.ceil((max_x - origin_x) / cell_size_meters))
    height = max(1, math.ceil((origin_y - min_y) / cell_size_meters))

    raster_driver = gdal.GetDriverByName("GTiff")
    raster = raster_driver.Create(
        str(raster_path),
        width,
        height,
        1,
        gdal.GDT_Byte,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    if raster is None:
        raise RuntimeError(f"Could not create {raster_path}")
    raster.SetGeoTransform((origin_x, cell_size_meters, 0, origin_y, 0, -cell_size_meters))
    raster.SetProjection("EPSG:3035")
    raster_band = raster.GetRasterBand(1)
    raster_band.SetNoDataValue(0)

    output_driver = ogr.GetDriverByName("GPKG")
    if output_path.exists():
        output_driver.DeleteDataSource(str(output_path))
    output = output_driver.CreateDataSource(str(output_path))
    if output is None:
        raise RuntimeError(f"Could not create {output_path}")
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(3035)
    output_layer = output.CreateLayer("natural_low", spatial_ref, ogr.wkbPolygon)
    output_layer.CreateField(ogr.FieldDefn("pixel_value", ogr.OFTInteger))
    output_layer.CreateField(ogr.FieldDefn("kind", ogr.OFTString))
    pixel_value_field_index = output_layer.GetLayerDefn().GetFieldIndex("pixel_value")

    source_layer.SetAttributeFilter("kind IS NOT NULL")
    available_kinds = {
        feature.GetField("kind")
        for feature in source_layer
        if feature.GetField("kind") is not None
    }
    kinds = [kind for kind in NATURAL_PRIORITY if kind in available_kinds]
    kinds.extend(
        {
            kind for kind in available_kinds if kind not in NATURAL_PRIORITY
        }
    )
    for pixel_value, kind in enumerate(kinds, start=1):
        LOGGER.info("Rasterizing natural class: %s", kind)
        source_layer.SetAttributeFilter(f"kind = '{kind}'")
        if gdal.RasterizeLayer(
            raster,
            [1],
            source_layer,
            burn_values=[pixel_value],
            options=["ALL_TOUCHED=TRUE"],
        ) != 0:
            raise RuntimeError(f"Could not rasterize natural class: {kind}")

    if gdal.Polygonize(
        raster_band,
        raster_band.GetMaskBand(),
        output_layer,
        pixel_value_field_index,
        ["8CONNECTED=8"],
    ) != 0:
        raise RuntimeError("Could not polygonize natural classes")
    output_layer.ResetReading()
    kind_by_pixel_value = dict(enumerate(kinds, start=1))
    for feature in output_layer:
        kind = kind_by_pixel_value.get(feature.GetField("pixel_value"))
        if kind is None:
            output_layer.DeleteFeature(feature.GetFID())
            continue
        feature.SetField("kind", kind)
        output_layer.SetFeature(feature)
    output_layer.ResetReading()

    source_layer.SetAttributeFilter(None)
    output.FlushCache()
    raster = None
    output = None
    source = None


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ensure_tools()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input file does not exist: {args.input}")
    if args.cell_size_meters <= 0:
        raise ValueError("--cell-size-meters must be positive")

    args.workdir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    filtered_pbf = args.workdir / "natural-filtered.osm.pbf"
    filter_file = args.workdir / "natural-filters.txt"
    valid_gpkg = args.workdir / "natural-valid.gpkg"
    raster_path = args.workdir / "natural-raster.tif"
    polygonized_gpkg = args.workdir / "natural-polygonized.gpkg"

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

    with timed_step("classify and convert to EPSG:3035 GeoPackage"):
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
                NATURAL_CLASSIFICATION_SQL,
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

    with timed_step("rasterize and polygonize natural classes"):
        polygonize_natural_polygons(
            valid_gpkg,
            polygonized_gpkg,
            raster_path,
            args.cell_size_meters,
        )

    with timed_step("export polygonized natural layer as WGS84 GeoJSONSeq"):
        run_command(
            [
                "ogr2ogr",
                "-f",
                "GeoJSONSeq",
                str(args.output),
                str(polygonized_gpkg),
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
