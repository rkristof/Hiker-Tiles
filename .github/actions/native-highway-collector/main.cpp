#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <osmium/io/any_input.hpp>
#include <osmium/io/reader.hpp>
#include <osmium/handler.hpp>
#include <osmium/osm/node.hpp>
#include <osmium/osm/way.hpp>
#include <osmium/visitor.hpp>

namespace {

using Id = std::int64_t;
constexpr std::uint32_t ADJACENCY_LEVELS = 3;

struct LocationRecord {
    Id id;
    std::int32_t longitude;
    std::int32_t latitude;
};

struct StoredWay {
    Id id;
    std::uint32_t node_offset;
    std::uint32_t node_count;
    std::string highway_type;
};

struct OutputNode {
    Id id;
    double longitude;
    double latitude;
    std::uint64_t edge_offset;
    std::uint32_t edge_count;
    std::uint32_t reserved;
};

static_assert(sizeof(OutputNode) == 40, "OutputNode format must remain 40 bytes");

struct OutputWay {
    Id id;
    std::uint32_t node_count;
    std::uint32_t reserved;
    char highway_type[32];
};

static_assert(sizeof(OutputWay) == 48, "OutputWay format must remain 48 bytes");

struct OutputEdge {
    std::uint32_t target_node_index;
    std::uint32_t reserved;
    Id way_id;
    double distance_m;
};

static_assert(sizeof(OutputEdge) == 24, "OutputEdge format must remain 24 bytes");

struct OutputSpatialCell {
    std::uint64_t key;
    std::uint64_t entry_offset;
    std::uint32_t entry_count;
    std::uint32_t reserved;
};

static_assert(sizeof(OutputSpatialCell) == 24, "OutputSpatialCell format must remain 24 bytes");

struct OutputHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t reserved;
    std::uint64_t node_count;
    std::uint64_t edge_count;
    std::uint64_t way_count;
    std::uint64_t cell_count;
    std::uint64_t spatial_entry_count;
};

static_assert(sizeof(OutputHeader) == 56, "OutputHeader format must remain 56 bytes");

constexpr double SPATIAL_CELL_SIZE = 0.01;
constexpr std::uint32_t OUTPUT_VERSION = 1;

struct SpatialIndexRef {
    std::uint64_t key;
    std::uint32_t node_index;
};

class OnePassCollector : public osmium::handler::Handler {
public:
    OnePassCollector(
        std::unordered_set<Id> route_node_ids,
        std::unordered_set<Id> excluded_way_ids
    ) :
        route_node_ids_(std::move(route_node_ids)),
        excluded_way_ids_(std::move(excluded_way_ids)) {}

    void node(const osmium::Node& node) {
        const auto location = node.location();
        if (!location.valid()) {
            return;
        }

        if (!locations_.empty() && node.id() < locations_.back().id) {
            locations_sorted_ = false;
        }
        locations_.push_back({node.id(), location.x(), location.y()});
    }

    void way(osmium::Way& way) {
        if (way.tags().get_value_by_key("highway") == nullptr) {
            return;
        }

        StoredWay stored_way{
            way.id(),
            static_cast<std::uint32_t>(way_node_ids_.size()),
            static_cast<std::uint32_t>(way.nodes().size()),
            way.tags().get_value_by_key("highway"),
        };

        bool direct = false;
        for (const auto& node_ref : way.nodes()) {
            const Id node_id = node_ref.ref();
            way_node_ids_.push_back(node_id);
            direct = direct || route_node_ids_.find(node_id) != route_node_ids_.end();
        }

        if (direct) {
            for (std::uint32_t index = 0; index < stored_way.node_count; ++index) {
                direct_node_ids_.insert(way_node_ids_[stored_way.node_offset + index]);
            }
        }
        direct_way_flags_.push_back(direct);
        ways_.push_back(std::move(stored_way));
    }

    void finish() {
        prepare_locations();
        select_ways();
    }

    const std::vector<StoredWay>& ways() const noexcept {
        return ways_;
    }

    const std::vector<Id>& way_node_ids() const noexcept {
        return way_node_ids_;
    }

    const std::vector<LocationRecord>& locations() const noexcept {
        return locations_;
    }

    const std::unordered_set<Id>& direct_node_ids() const noexcept {
        return direct_node_ids_;
    }

    const std::vector<std::size_t>& selected_way_indices() const noexcept {
        return selected_way_indices_;
    }

    const std::vector<std::uint8_t>& direct_way_flags() const noexcept {
        return direct_way_flags_;
    }

    const std::unordered_set<Id>& excluded_way_ids() const noexcept {
        return excluded_way_ids_;
    }

private:
    void prepare_locations() {
        if (locations_sorted_) {
            return;
        }
        std::sort(
            locations_.begin(),
            locations_.end(),
            [](const LocationRecord& first, const LocationRecord& second) {
                return first.id < second.id;
            }
        );
        locations_sorted_ = true;
    }

    void select_ways() {
        selected_way_flags_.assign(ways_.size(), 0);
        selected_way_indices_.clear();
        selected_way_indices_.reserve(ways_.size());

        for (std::size_t way_index = 0; way_index < ways_.size(); ++way_index) {
            if (direct_way_flags_[way_index] != 0) {
                selected_way_flags_[way_index] = 1;
            }
        }

        std::unordered_set<Id> frontier = direct_node_ids_;
        for (std::uint32_t level = 1; level < ADJACENCY_LEVELS && !frontier.empty(); ++level) {
            std::unordered_set<Id> next_frontier;
            for (std::size_t way_index = 0; way_index < ways_.size(); ++way_index) {
                if (selected_way_flags_[way_index] != 0) {
                    continue;
                }
                const auto& way = ways_[way_index];
                bool adjacent = false;
                for (std::uint32_t node_index = 0; node_index < way.node_count; ++node_index) {
                    if (frontier.find(way_node_ids_[way.node_offset + node_index]) != frontier.end()) {
                        adjacent = true;
                        break;
                    }
                }
                if (!adjacent) {
                    continue;
                }

                selected_way_flags_[way_index] = 1;
                next_frontier.insert(
                    way_node_ids_.begin() + way.node_offset,
                    way_node_ids_.begin() + way.node_offset + way.node_count
                );
            }
            frontier.swap(next_frontier);
        }

        for (std::size_t way_index = 0; way_index < ways_.size(); ++way_index) {
            if (selected_way_flags_[way_index] != 0) {
                selected_way_indices_.push_back(way_index);
            }
        }
    }

    std::unordered_set<Id> route_node_ids_;
    std::unordered_set<Id> excluded_way_ids_;
    std::unordered_set<Id> direct_node_ids_;
    std::vector<LocationRecord> locations_;
    std::vector<StoredWay> ways_;
    std::vector<Id> way_node_ids_;
    std::vector<std::uint8_t> direct_way_flags_;
    std::vector<std::uint8_t> selected_way_flags_;
    std::vector<std::size_t> selected_way_indices_;
    bool locations_sorted_ = true;
};

std::unordered_set<Id> read_ids(const std::string& filename) {
    std::ifstream input(filename);
    if (!input) {
        throw std::runtime_error("Unable to open route node file: " + filename);
    }

    std::unordered_set<Id> ids;
    Id id = 0;
    while (input >> id) {
        ids.insert(id);
    }
    return ids;
}

const LocationRecord* find_location(
    const std::vector<LocationRecord>& locations,
    Id node_id
) {
    const auto it = std::lower_bound(
        locations.begin(),
        locations.end(),
        node_id,
        [](const LocationRecord& location, Id id) {
            return location.id < id;
        }
    );
    if (it == locations.end() || it->id != node_id) {
        return nullptr;
    }
    return &*it;
}

template <typename T>
void write_value(std::ofstream& output, const T& value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!output) {
        throw std::runtime_error("Unable to write collector output");
    }
}

template <typename T>
void write_block(std::ofstream& output, const std::vector<T>& values) {
    if (values.empty()) {
        return;
    }
    output.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(T))
    );
    if (!output) {
        throw std::runtime_error("Unable to write collector output");
    }
}

double haversine_distance_m(const OutputNode& first, const OutputNode& second) {
    constexpr double EARTH_RADIUS_M = 6371000.0;
    constexpr double PI = 3.14159265358979323846;
    const double first_latitude = first.latitude * PI / 180.0;
    const double second_latitude = second.latitude * PI / 180.0;
    const double delta_latitude = second_latitude - first_latitude;
    const double delta_longitude = (second.longitude - first.longitude) * PI / 180.0;
    const double haversine_term =
        std::sin(delta_latitude / 2.0) * std::sin(delta_latitude / 2.0)
        + std::cos(first_latitude) * std::cos(second_latitude)
        * std::sin(delta_longitude / 2.0) * std::sin(delta_longitude / 2.0);
    return EARTH_RADIUS_M * 2.0 * std::asin(std::sqrt(haversine_term));
}

std::uint64_t spatial_cell_key(double longitude, double latitude) {
    const auto longitude_cell = static_cast<std::int32_t>(
        std::floor(longitude / SPATIAL_CELL_SIZE)
    );
    const auto latitude_cell = static_cast<std::int32_t>(
        std::floor(latitude / SPATIAL_CELL_SIZE)
    );
    return (
        static_cast<std::uint64_t>(static_cast<std::uint32_t>(longitude_cell)) << 32
    ) | static_cast<std::uint32_t>(latitude_cell);
}

void write_output(
    const std::string& filename,
    OnePassCollector& collector
) {
    std::ofstream output(filename, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Unable to open collector output: " + filename);
    }

    std::vector<OutputWay> output_ways;
    std::vector<OutputNode> output_nodes;
    std::vector<std::vector<std::uint32_t>> way_node_indexes;
    std::unordered_map<Id, std::uint32_t> node_indexes;
    output_ways.reserve(collector.selected_way_indices().size());
    for (const std::size_t source_way_index : collector.selected_way_indices()) {
        const auto& way = collector.ways()[source_way_index];
        if (collector.excluded_way_ids().find(way.id) != collector.excluded_way_ids().end()) {
            continue;
        }
        std::vector<const LocationRecord*> locations;
        locations.reserve(way.node_count);
        bool locations_valid = true;
        for (std::uint32_t node_index = 0; node_index < way.node_count; ++node_index) {
            const Id node_id = collector.way_node_ids()[way.node_offset + node_index];
            const LocationRecord* location = find_location(collector.locations(), node_id);
            if (location == nullptr) {
                locations_valid = false;
                break;
            }
            locations.push_back(location);
        }
        if (!locations_valid) {
            continue;
        }

        if (way.highway_type.size() >= 32) {
            throw std::runtime_error("Highway type exceeds native format limit");
        }
        OutputWay output_way{};
        output_way.id = way.id;
        output_way.node_count = way.node_count;
        std::memcpy(
            output_way.highway_type,
            way.highway_type.data(),
            way.highway_type.size()
        );
        output_ways.push_back(output_way);
        way_node_indexes.emplace_back();
        way_node_indexes.back().reserve(way.node_count);
        for (std::uint32_t node_index = 0; node_index < way.node_count; ++node_index) {
            const Id node_id = collector.way_node_ids()[way.node_offset + node_index];
            const auto existing_node = node_indexes.find(node_id);
            if (existing_node != node_indexes.end()) {
                way_node_indexes.back().push_back(existing_node->second);
                continue;
            }

            const auto node_index_value = static_cast<std::uint32_t>(output_nodes.size());
            node_indexes.emplace(node_id, node_index_value);
            const auto* location = locations[node_index];
            output_nodes.push_back({
                node_id,
                osmium::Location(location->longitude, location->latitude).lon_without_check(),
                osmium::Location(location->longitude, location->latitude).lat_without_check(),
                0,
                0,
                0,
            });
            way_node_indexes.back().push_back(node_index_value);
        }
    }

    std::vector<std::uint32_t> sorted_node_indexes(output_nodes.size());
    std::iota(sorted_node_indexes.begin(), sorted_node_indexes.end(), 0);
    std::sort(
        sorted_node_indexes.begin(),
        sorted_node_indexes.end(),
        [&output_nodes](const std::uint32_t first, const std::uint32_t second) {
            return output_nodes[first].id < output_nodes[second].id;
        }
    );
    std::vector<std::uint32_t> remapped_node_indexes(output_nodes.size());
    std::vector<OutputNode> sorted_nodes;
    sorted_nodes.reserve(output_nodes.size());
    for (std::uint32_t sorted_index = 0; sorted_index < sorted_node_indexes.size(); ++sorted_index) {
        const auto original_index = sorted_node_indexes[sorted_index];
        remapped_node_indexes[original_index] = sorted_index;
        sorted_nodes.push_back(output_nodes[original_index]);
    }
    output_nodes.swap(sorted_nodes);
    for (auto& node_indexes_for_way : way_node_indexes) {
        for (auto& node_index : node_indexes_for_way) {
            node_index = remapped_node_indexes[node_index];
        }
    }

    std::vector<std::vector<OutputEdge>> adjacency(output_nodes.size());
    for (std::size_t way_index = 0; way_index < output_ways.size(); ++way_index) {
        const auto& way = output_ways[way_index];
        const auto& node_indexes_for_way = way_node_indexes[way_index];
        for (std::uint32_t node_index = 0; node_index + 1 < way.node_count; ++node_index) {
            const auto first_node_index = node_indexes_for_way[node_index];
            const auto second_node_index = node_indexes_for_way[node_index + 1];
            if (first_node_index == second_node_index) {
                continue;
            }
            const auto distance = haversine_distance_m(
                output_nodes[first_node_index],
                output_nodes[second_node_index]
            );
            adjacency[first_node_index].push_back({
                second_node_index,
                0,
                way.id,
                distance,
            });
            adjacency[second_node_index].push_back({
                first_node_index,
                0,
                way.id,
                distance,
            });
        }
    }

    std::vector<OutputEdge> output_edges;
    output_edges.reserve(output_nodes.size() * 2);
    for (std::size_t node_index = 0; node_index < output_nodes.size(); ++node_index) {
        output_nodes[node_index].edge_offset = output_edges.size();
        output_nodes[node_index].edge_count = adjacency[node_index].size();
        output_edges.insert(
            output_edges.end(),
            adjacency[node_index].begin(),
            adjacency[node_index].end()
        );
    }

    std::vector<SpatialIndexRef> spatial_refs;
    spatial_refs.reserve(output_nodes.size());
    for (std::uint32_t node_index = 0; node_index < output_nodes.size(); ++node_index) {
        spatial_refs.push_back({
            spatial_cell_key(
                output_nodes[node_index].longitude,
                output_nodes[node_index].latitude
            ),
            node_index,
        });
    }
    std::sort(
        spatial_refs.begin(),
        spatial_refs.end(),
        [](const SpatialIndexRef& first, const SpatialIndexRef& second) {
            if (first.key != second.key) {
                return first.key < second.key;
            }
            return first.node_index < second.node_index;
        }
    );

    std::vector<OutputSpatialCell> spatial_cells;
    std::vector<std::uint32_t> spatial_entries;
    spatial_entries.reserve(spatial_refs.size());
    for (const auto& spatial_ref : spatial_refs) {
        if (spatial_cells.empty() || spatial_cells.back().key != spatial_ref.key) {
            spatial_cells.push_back({
                spatial_ref.key,
                spatial_entries.size(),
                0,
                0,
            });
        }
        spatial_entries.push_back(spatial_ref.node_index);
        ++spatial_cells.back().entry_count;
    }

    const OutputHeader header{
        {'H', 'I', 'K', 'E', 'R', 'I', 'D', 'X'},
        OUTPUT_VERSION,
        0,
        output_nodes.size(),
        output_edges.size(),
        output_ways.size(),
        spatial_cells.size(),
        spatial_entries.size(),
    };
    write_value(output, header);
    write_block(output, output_nodes);
    write_block(output, output_edges);
    write_block(output, output_ways);
    write_block(output, spatial_cells);
    write_block(output, spatial_entries);
    if (!output) {
        throw std::runtime_error("Unable to finalize collector output");
    }

}

} // namespace

int main(int argc, char** argv) {
    if (argc != 7 && argc != 9) {
        std::cerr << "Usage: native-highway-collector --input FILE --route-nodes FILE --output FILE [--exclude-way-ids FILE]\n";
        return 2;
    }

    try {
        std::string input_filename;
        std::string route_nodes_filename;
        std::string output_filename;
        std::string excluded_way_ids_filename;
        for (int index = 1; index < argc; index += 2) {
            const std::string option = argv[index];
            const std::string value = argv[index + 1];
            if (option == "--input") {
                input_filename = value;
            } else if (option == "--route-nodes") {
                route_nodes_filename = value;
            } else if (option == "--output") {
                output_filename = value;
            } else if (option == "--exclude-way-ids") {
                excluded_way_ids_filename = value;
            } else {
                throw std::invalid_argument("unknown option: " + option);
            }
        }
        if (input_filename.empty() || route_nodes_filename.empty() || output_filename.empty()) {
            throw std::invalid_argument("input, route nodes, and output are required");
        }
        auto route_node_ids = read_ids(route_nodes_filename);
        auto excluded_way_ids = excluded_way_ids_filename.empty()
            ? std::unordered_set<Id>()
            : read_ids(excluded_way_ids_filename);
        OnePassCollector collector(
            std::move(route_node_ids),
            std::move(excluded_way_ids)
        );

        osmium::io::Reader reader(
            input_filename,
            osmium::osm_entity_bits::node | osmium::osm_entity_bits::way,
            osmium::io::read_meta::no,
            osmium::io::buffers_type::single
        );
        while (auto buffer = reader.read()) {
            osmium::apply(buffer, collector);
        }
        reader.close();
        collector.finish();

        write_output(output_filename, collector);
    } catch (const std::exception& error) {
        std::cerr << "native-highway-collector: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
