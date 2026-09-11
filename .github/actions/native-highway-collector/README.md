# Native highway collector

One-pass libosmium collector used by route repair. It buffers highway way node IDs
and fixed-point node locations during one PBF read, then selects direct and
adjacent ways in memory after direct-node closure is complete.

Adjacency levels count the direct highway level: level 1 selects route-touching
ways, and each additional level adds highways sharing nodes with the previous
frontier. Collector always uses three levels. Fixed binary layout keeps selected
geometry in original PBF way order.

The output starts with section counts followed by contiguous sections in this
order: canonical nodes, weighted CSR edges, selected way metadata, spatial
cells, and spatial node postings. Way metadata contains the way ID, node count,
and fixed-width highway type. Node IDs are sorted. Each edge stores its target
node position, way ID, and precomputed Haversine distance.
Each spatial cell stores a contiguous posting range; cells use a uniform `0.01`
degree longitude/latitude grid.

The Python adapter memory-maps these sections with NumPy and exposes only the
explicit route-repair operations `contains_node`, `neighbors`,
`way_node_count`, and `nodes_within_distance`. Dijkstra traversal consumes the
native edge weights directly. Spatial queries use cell postings followed by
exact Haversine filtering.

Build locally:

```bash
brew install cmake libosmium protozero
cmake -S .github/actions/native-highway-collector \
  -B /tmp/native-highway-collector-build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/native-highway-collector-build --parallel
```

The production action downloads the Linux AMD64 binary from the
`native-highway-collector-latest` release. Rebuild and publish it manually with the
`Build Highway Collector` workflow after changing the native source.

Native node geometry and edge data stay in the memory-mapped file.
