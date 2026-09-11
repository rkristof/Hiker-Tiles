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

gdal.UseExceptions()
ogr.UseExceptions()

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
    SELECT {geometry_expression},
    CASE
        WHEN landuse = 'grass' THEN 1
        WHEN landuse = 'farmland' THEN 2
        WHEN landuse = 'forest' THEN 12
        WHEN natural = 'wood' THEN 12
        WHEN natural = 'grassland' OR landuse = 'meadow' THEN 1
        WHEN landuse IN ('orchard', 'vineyard', 'farmyard', 'greenhouse_horticulture', 'allotments') THEN 2
        WHEN landuse = 'quarry' THEN 8
        WHEN natural = 'glacier' THEN 11
        WHEN natural = 'bare_rock' THEN 9
        WHEN natural IN ('sand', 'beach') THEN 6
        WHEN natural = 'heath' THEN 4
        WHEN natural = 'scrub' THEN 5
        WHEN natural = 'scree' THEN 10
        WHEN natural = 'shingle' THEN 7
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
AND ST_Area(geometry) * {max_wgs84_meters_per_degree} * {max_wgs84_meters_per_degree} >= {min_area_m2}
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
        default=500.0,
        help="Side length of the coarse projected raster cells, in meters.",
    )
    parser.add_argument(
        "--min-area-m2",
        type=float,
        default=250000.0,
        help="Minimum source polygon area retained before rasterization.",
    )
    parser.add_argument(
        "--smoothing-factor",
        type=float,
        default=0.2,
        help="Chaikin corner-cutting factor for polygonized boundaries.",
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


def memory_vector_driver() -> ogr.Driver:
    return ogr.GetDriverByName("MEM") or ogr.GetDriverByName("Memory")


def ensure_tools() -> None:
    for tool in ("osmium",):
        if shutil.which(tool) is None:
            raise RuntimeError(f"Required executable not found: {tool}")


def classification_sql(min_area_m2: float) -> str:
    return NATURAL_CLASSIFICATION_SQL.format(
        geometry_expression="geometry",
        min_area_m2=f"{min_area_m2:.12g}",
        max_wgs84_meters_per_degree=f"{MAX_WGS84_METERS_PER_DEGREE:.1f}",
    ).strip()


def smooth_ring(ring: ogr.Geometry, smoothing_factor: float) -> ogr.Geometry:
    point_count = ring.GetPointCount()
    if point_count < 4:
        return ring.Clone()

    points = [ring.GetPoint_2D(index) for index in range(point_count - 1)]
    smoothed_ring = ogr.Geometry(ogr.wkbLinearRing)
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        first_point = (
            point[0] + smoothing_factor * (next_point[0] - point[0]),
            point[1] + smoothing_factor * (next_point[1] - point[1]),
        )
        second_point = (
            next_point[0] - smoothing_factor * (next_point[0] - point[0]),
            next_point[1] - smoothing_factor * (next_point[1] - point[1]),
        )
        smoothed_ring.AddPoint_2D(*first_point)
        smoothed_ring.AddPoint_2D(*second_point)
    smoothed_ring.AddPoint_2D(*smoothed_ring.GetPoint_2D(0))
    return smoothed_ring


def smooth_polygon(geometry: ogr.Geometry, smoothing_factor: float) -> ogr.Geometry:
    smoothed_polygon = ogr.Geometry(ogr.wkbPolygon)
    for index in range(geometry.GetGeometryCount()):
        ring = geometry.GetGeometryRef(index)
        smoothed_polygon.AddGeometry(smooth_ring(ring, smoothing_factor))
    return smoothed_polygon


def smooth_geometry(geometry: ogr.Geometry, smoothing_factor: float) -> ogr.Geometry:
    if smoothing_factor == 0:
        return geometry.Clone()
    geometry_type = ogr.GT_Flatten(geometry.GetGeometryType())
    if geometry_type == ogr.wkbPolygon:
        return smooth_polygon(geometry, smoothing_factor)
    if geometry_type == ogr.wkbMultiPolygon:
        smoothed_geometry = ogr.Geometry(ogr.wkbMultiPolygon)
        for index in range(geometry.GetGeometryCount()):
            smoothed_geometry.AddGeometry(
                smooth_polygon(geometry.GetGeometryRef(index), smoothing_factor)
            )
        return smoothed_geometry
    return geometry.Clone()


def polygonize_natural_polygons(
    input_path: Path,
    output_path: Path,
    cell_size_meters: float,
    min_area_m2: float,
    smoothing_factor: float,
) -> None:
    """Rasterize classified PBF geometries and export occupied cells."""
    gdal.SetConfigOption("OGR_INTERLEAVED_READING", "YES")
    source = gdal.OpenEx(str(input_path), gdal.OF_VECTOR)
    if source is None:
        raise RuntimeError(f"Could not open {input_path}")

    projected_source = None
    projected_layer = None
    selected_layer = None
    raster = None
    polygonized = None
    output_data_source = None
    try:
        with timed_step("project and classify natural polygons"):
            projected_source = gdal.VectorTranslate(
                "",
                source,
                format="Memory",
                accessMode="overwrite",
                dstSRS="EPSG:3035",
                SQLStatement=classification_sql(min_area_m2),
                SQLDialect="SQLite",
                layerName="natural_classified",
            )
            if projected_source is None:
                raise RuntimeError("Could not create projected natural dataset")
            projected_layer = projected_source.GetLayerByName("natural_classified")
            if projected_layer is None:
                raise RuntimeError("Layer natural_classified not found")
            selected_layer = projected_source.ExecuteSQL(
                "SELECT geometry, pixel_value FROM natural_classified "
                f"WHERE ST_Area(geometry) >= {min_area_m2:.12g} "
                "ORDER BY pixel_value",
                dialect="SQLite",
            )
            if selected_layer is None:
                raise RuntimeError("Could not filter projected natural dataset")
            projected_spatial_ref = selected_layer.GetSpatialRef()
            if projected_spatial_ref is None:
                raise RuntimeError("Projected natural layer has no spatial reference")
            selected_layer.GetFeatureCount()
            min_x, max_x, min_y, max_y = selected_layer.GetExtent()
        origin_x = math.floor(min_x / cell_size_meters) * cell_size_meters
        origin_y = math.ceil(max_y / cell_size_meters) * cell_size_meters
        width = max(1, math.ceil((max_x - origin_x) / cell_size_meters))
        height = max(1, math.ceil((origin_y - min_y) / cell_size_meters))

        with timed_step("rasterize natural classes"):
            raster_driver = gdal.GetDriverByName("MEM")
            raster = raster_driver.Create("", width, height, 1, gdal.GDT_Byte)
            if raster is None:
                raise RuntimeError("Could not create in-memory raster")
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
                selected_layer,
                options=["ATTRIBUTE=pixel_value"],
            ) != 0:
                raise RuntimeError("Could not rasterize natural classes")

        with timed_step("polygonize and export natural classes"):
            memory_driver = memory_vector_driver()
            if memory_driver is None:
                raise RuntimeError("No in-memory vector driver available")
            polygonized = memory_driver.CreateDataSource("natural_polygonized")
            if polygonized is None:
                raise RuntimeError("Could not create in-memory polygonized layer")
            spatial_ref = projected_spatial_ref
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

            output_driver = ogr.GetDriverByName("GeoJSONSeq")
            if output_driver is None:
                raise RuntimeError("GeoJSONSeq driver not available")
            if output_path.exists():
                output_path.unlink()
            output_data_source = output_driver.CreateDataSource(str(output_path))
            if output_data_source is None:
                raise RuntimeError(f"Could not create {output_path}")

            wgs84 = osr.SpatialReference()
            wgs84.ImportFromEPSG(4326)
            wgs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            coordinate_transform = osr.CoordinateTransformation(
                spatial_ref, wgs84
            )
            output_layer = output_data_source.CreateLayer(
                "natural_low",
                wgs84,
                ogr.wkbPolygon,
                options=["RS=NO", "COORDINATE_PRECISION=6"],
            )
            if output_layer is None:
                raise RuntimeError("Could not create GeoJSONSeq layer")
            output_layer.CreateField(ogr.FieldDefn("kind", ogr.OFTString))
            output_definition = output_layer.GetLayerDefn()

            polygonized_layer.ResetReading()
            for feature in polygonized_layer:
                pixel_value = feature.GetFieldAsInteger("pixel_value")
                if pixel_value == 0:
                    continue
                if not 1 <= pixel_value <= len(NATURAL_PRIORITY):
                    raise RuntimeError(f"Unexpected natural pixel value: {pixel_value}")
                source_geometry = feature.GetGeometryRef()
                if source_geometry is None or source_geometry.IsEmpty():
                    continue
                geometry = source_geometry.Clone()
                if geometry.GetArea() < min_area_m2:
                    continue
                geometry = geometry.Simplify(cell_size_meters)
                if geometry is None or geometry.IsEmpty():
                    continue
                geometry = smooth_geometry(geometry, smoothing_factor)
                gdal.PushErrorHandler("CPLQuietErrorHandler")
                try:
                    geometry_is_valid = geometry.IsValid()
                finally:
                    gdal.PopErrorHandler()
                if not geometry_is_valid:
                    geometry = geometry.MakeValid()
                    if geometry is None or geometry.IsEmpty():
                        continue
                if geometry.Transform(coordinate_transform) != 0:
                    raise RuntimeError("Could not transform natural polygon to WGS84")
                if ogr.GT_Flatten(geometry.GetGeometryType()) == ogr.wkbMultiPolygon:
                    output_geometries = (
                        geometry.GetGeometryRef(index).Clone()
                        for index in range(geometry.GetGeometryCount())
                    )
                else:
                    output_geometries = (geometry,)
                for output_geometry in output_geometries:
                    output_feature = ogr.Feature(output_definition)
                    output_feature.SetGeometry(output_geometry)
                    output_feature.SetField("kind", NATURAL_PRIORITY[pixel_value - 1])
                    if output_layer.CreateFeature(output_feature) != 0:
                        raise RuntimeError(f"Could not write feature to {output_path}")
                    output_feature = None
        output_data_source = None
    finally:
        if projected_source is not None and selected_layer is not None:
            projected_source.ReleaseResultSet(selected_layer)
        selected_layer = None
        projected_layer = None
        projected_source = None
        raster = None
        polygonized = None
        output_data_source = None
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
    if not 0 <= args.smoothing_factor < 0.5:
        raise ValueError("--smoothing-factor must be between 0 and 0.5")

    args.workdir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    filtered_pbf = args.workdir / "natural-filtered.osm.pbf"
    filter_file = args.workdir / "natural-filters.txt"

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
            args.cell_size_meters,
            args.min_area_m2,
            args.smoothing_factor,
        )

    LOGGER.info("Natural preprocessing complete: %s", args.output)


if __name__ == "__main__":
    main()
