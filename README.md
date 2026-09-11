# Hiker-Tiles

Hiker-Tiles contains the GitHub Actions and processing scripts used to build
offline map tiles for Hike-R.

## How it works

- Country workflows download an OpenStreetMap extract, generate base-map data,
  and build hiking-route vector tiles.
- The world-base workflow builds a small global context map from public
  geographic datasets.
- Generated MBTiles archives are published as GitHub Release assets for the
  app to download.

## Data sources

Tile contents may include data from:

- [OpenStreetMap](https://www.openstreetmap.org/), distributed under the
  [ODbL](https://opendatacommons.org/licenses/odbl/).
- [Geofabrik](https://www.geofabrik.de/), which provides regional OSM extracts.
- [Natural Earth](https://www.naturalearthdata.com/), used for global labels.
- [Copernicus DEM](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-digital-elevation-model), used for elevation data.

### Generated map data

Country tiles contain OpenStreetMap data copyright OpenStreetMap contributors,
available under the [Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/).
Regional extracts are provided by Geofabrik GmbH.

Elevation data was produced using Copernicus WorldDEM-30 copyright DLR e.V.
2010-2014 and Airbus Defence and Space GmbH 2014-2018, provided under
COPERNICUS by the European Union and ESA; all rights reserved.

World base tiles contain OSM land polygons available under the ODbL 1.0 and
Natural Earth data available in the public domain.

Generated data remains subject to the terms of its upstream sources.

## Related information

- [Hike-R privacy policy](docs/privacy.md)
- [GNU GPLv3](LICENSE) for repository code
