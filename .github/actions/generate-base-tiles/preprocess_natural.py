#!/usr/bin/env python3
"""Rasterize natural polygons for low-zoom tile generation."""

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
MAX_WGS84_METERS_PER_DEGREE = 112000.0

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
    SELECT ST_Transform(geometry, 3035) AS geometry,
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
    END AS kind,
    CASE
        WHEN landuse IN ('forest', 'grass', 'farmland') THEN
            CASE landuse WHEN 'grass' THEN 1 WHEN 'farmland' THEN 2 ELSE 15 END
        WHEN natural = 'wood' THEN 15
        WHEN natural = 'grassland' OR landuse = 'meadow' THEN 1
        WHEN landuse IN ('orchard', 'vineyard', 'farmyard', 'greenhouse_horticulture', 'allotments') THEN 2
        WHEN landuse = 'quarry' THEN 11
        WHEN natural = 'glacier' THEN 14
        WHEN natural = 'bare_rock' THEN 12
        WHEN natural IN ('sand', 'beach') THEN 9
        WHEN natural = 'heath' THEN 7
        WHEN natural = 'scrub' THEN 8
        WHEN natural = 'scree' THEN 13
        WHEN natural = 'shingle' THEN 10
        WHEN HSTORE_GET_VALUE(other_tags, 'wetland') IN ('swamp', 'bog', 'wet_meadow', 'marsh') THEN 3
        WHEN natural = 'wetland' THEN 3
        WHEN natural = 'fell' THEN 1
    END AS pixel_value
FROM multipolygons
WHERE (
        landuse IN ('forest', 'grass', 'farmland', 'meadow', 'orchard', 'vineyard', 'farmyard', 'greenhouse_horticulture', 'allotments', 'quarry')
         OR natural IN ('wood', 'grassland', 'glacier', 'bare_rock', 'sand', 'heath', 'scrub', 'scree', 'shingle', 'wetland', 'fell', 'beach')
            OR HSTORE_GET_VALUE(other_tags, 'wetland') IN ('swamp', 'bog', 'wet_meadow', 'marsh')
)
AND ST_Area(ST_Envelope(geometry))
    * {max_wgs84_meters_per_degree}
    * {max_wgs84_meters_per_degree} >= {min_area_m2}
AND ST_Area(ST_Transform(geometry, 3035)) >= {min_area_m2}
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
        default=300.0,
        help="Side length of the coarse projected raster cells, in meters.",
    )
    parser.add_argument(
        "--min-area-m2",
        type=float,
        default=25000.0,
        help="Minimum source polygon area retained before rasterization.",
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
    for tool in ("osmium",):
        if shutil.which(tool) is None:
            raise RuntimeError(f"Required executable not found: {tool}")


def polygonize_natural_polygons(
    input_path: Path,
    output_path: Path,
    raster_path: Path,
    cell_size_meters: float,
    min_area_m2: float,
) -> None:
    """Rasterize classified PBF geometries and export occupied cells."""
    source = ogr.Open(str(input_path))
    if source is None:
        raise RuntimeError(f"Could not open {input_path}")
    source_layer = source.GetLayerByName("multipolygons")
    if source_layer is None:
        raise RuntimeError(f"Layer multipolygons not found in {input_path}")

    ordered_layer = None
    raster = None
    polygonized = None
    output = None
    try:
        ordered_layer = source.ExecuteSQL(
            NATURAL_CLASSIFICATION_SQL.format(
                min_area_m2=f"{min_area_m2:.12g}",
                max_wgs84_meters_per_degree=f"{MAX_WGS84_METERS_PER_DEGREE:.12g}",
            )
            + "\nORDER BY pixel_value ASC",
            dialect="SQLite",
        )
        if ordered_layer is None:
            raise RuntimeError("Could not classify natural polygons")

        min_x, max_x, min_y, max_y = ordered_layer.GetExtent()
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
        raster.SetGeoTransform(
            (origin_x, cell_size_meters, 0, origin_y, 0, -cell_size_meters)
        )
        raster.SetProjection("EPSG:3035")
        raster_band = raster.GetRasterBand(1)
        raster_band.SetNoDataValue(0)
        raster_band.Fill(0)

        if gdal.RasterizeLayer(
            raster,
            [1],
            ordered_layer,
            options=["ATTRIBUTE=pixel_value"],
        ) != 0:
            raise RuntimeError("Could not rasterize natural classes")
        raster_band.FlushCache()

        memory_driver = ogr.GetDriverByName("Memory")
        polygonized = memory_driver.CreateDataSource("natural_polygonized")
        if polygonized is None:
            raise RuntimeError("Could not create in-memory polygonized layer")
        spatial_ref = osr.SpatialReference()
        spatial_ref.ImportFromEPSG(3035)
        polygonized_layer = polygonized.CreateLayer(
            "natural_low", spatial_ref, ogr.wkbPolygon
        )
        polygonized_layer.CreateField(ogr.FieldDefn("pixel_value", ogr.OFTInteger))
        pixel_value_field_index = polygonized_layer.GetLayerDefn().GetFieldIndex(
            "pixel_value"
        )

        if gdal.Polygonize(
            raster_band,
            raster_band.GetMaskBand(),
            polygonized_layer,
            pixel_value_field_index,
            ["8CONNECTED=8"],
        ) != 0:
            raise RuntimeError("Could not polygonize natural classes")

        geojson_driver = ogr.GetDriverByName("GeoJSONSeq")
        if output_path.exists():
            output_path.unlink()
        output = geojson_driver.CreateDataSource(str(output_path))
        if output is None:
            raise RuntimeError(f"Could not create {output_path}")
        wgs84 = osr.SpatialReference()
        wgs84.ImportFromEPSG(4326)
        output_layer = output.CreateLayer(
            "natural_low",
            wgs84,
            ogr.wkbPolygon,
            options=["RS=NO", "COORDINATE_PRECISION=6"],
        )
        output_layer.CreateField(ogr.FieldDefn("kind", ogr.OFTString))
        output_defn = output_layer.GetLayerDefn()
        polygonized_layer.ResetReading()
        for feature in polygonized_layer:
            pixel_value = feature.GetFieldAsInteger("pixel_value")
            if pixel_value <= 0 or pixel_value > len(NATURAL_PRIORITY):
                continue
            geometry = feature.GetGeometryRef().Clone()
            if geometry.TransformTo(wgs84) != 0:
                raise RuntimeError("Could not transform polygonized natural geometry")
            output_feature = ogr.Feature(output_defn)
            output_feature.SetGeometry(geometry)
            output_feature.SetField("kind", NATURAL_PRIORITY[pixel_value - 1])
            if output_layer.CreateFeature(output_feature) != 0:
                raise RuntimeError("Could not write polygonized natural feature")
        output.FlushCache()
    finally:
        if ordered_layer is not None:
            source.ReleaseResultSet(ordered_layer)
        raster = None
        polygonized = None
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
    if args.min_area_m2 < 0:
        raise ValueError("--min-area-m2 cannot be negative")

    args.workdir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    filtered_pbf = args.workdir / "natural-filtered.osm.pbf"
    filter_file = args.workdir / "natural-filters.txt"
    raster_path = args.workdir / "natural-raster.tif"

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

    with timed_step("classify, rasterize, and polygonize natural classes"):
        polygonize_natural_polygons(
            filtered_pbf,
            args.output,
            raster_path,
            args.cell_size_meters,
            args.min_area_m2,
        )

    LOGGER.info("Natural preprocessing complete: %s", args.output)


if __name__ == "__main__":
    main()
