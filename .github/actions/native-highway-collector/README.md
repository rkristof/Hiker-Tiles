# Native highway collector

One-pass libosmium collector used by route repair. It buffers highway way node IDs
and fixed-point node locations during one PBF read, then selects direct and
adjacent ways in memory after direct-node closure is complete. A batch request
file lets the collector calculate route-specific access summaries for every
short route during the same PBF read.

Adjacency levels count the direct highway level: level 1 selects route-touching
ways, and each additional level adds highways sharing nodes with the previous
frontier. Collector always uses three levels. Fixed binary layout keeps selected
geometry in original PBF way order.

The command line is:

```text
native-highway-collector --input FILE --route-nodes FILE \
  --route-requests FILE --output FILE
```

The route-node file contains one node ID per line. The route-request file starts
with its request count, followed by one line per relation:

```text
RELATION_ID ROUTE_WAY_COUNT CANDIDATE_NODE_COUNT [WAY_IDS...] [NODE_IDS...]
```

The output starts with five section counts followed by contiguous sections in
this order: canonical nodes, weighted CSR edges, spatial cells, spatial node
postings, and relation-scoped access summaries. Each summary contains relation
ID, node ID, count of distinct external ways, and count of external ways in high
highway classes. Node IDs are sorted. Each edge stores its target node position,
way ID, and precomputed Haversine distance.
Each spatial cell stores a contiguous posting range; cells use a uniform `0.01`
degree longitude/latitude grid.

The Python adapter memory-maps these sections with NumPy. It exposes route
access summaries grouped by relation, plus the route-repair operations
`contains_node`, `neighbors`, and `nodes_within_distance`. Dijkstra traversal
consumes the native edge weights directly. Spatial queries use cell postings
followed by exact Haversine filtering.

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
