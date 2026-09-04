# Native highway collector

One-pass libosmium collector used by route repair. It buffers compact highway way
node IDs and fixed-point node locations during one PBF read, then selects direct
and adjacent ways in memory after direct-node closure is complete.

Adjacency levels count the direct highway level: level 1 selects route-touching
ways, and each additional level adds highways sharing nodes with the previous
frontier. Collector always uses three levels. Fixed binary layout keeps selected
geometry in original PBF way order. Callers may provide excluded way IDs after
adjacency selection when selected geometry is already available from another source.

The fixed header contains section counts followed by contiguous way records, node
records, highway type strings, and a sorted node-to-segment index. The Python
adapter memory-maps these sections with NumPy. Way objects and node sequences
are created lazily, while adjacency lookups expose views over the native index
instead of rebuilding full tuples for every lookup. Direct highway lookup is
derived from the route-node index; no separate direct-way block is serialized.

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

Native node geometry uses NumPy-backed sequences, preserving the graph-repair API
without eagerly creating Python coordinate lists or index tuples.
