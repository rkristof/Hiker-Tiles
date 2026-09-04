#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
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
    std::uint16_t highway_type;
};

struct OutputNode {
    Id id;
    double longitude;
    double latitude;
};

static_assert(sizeof(OutputNode) == 24, "OutputNode format must remain 24 bytes");

struct OutputWay {
    Id id;
    std::uint64_t node_offset;
    std::uint32_t node_count;
    std::uint32_t highway_type;
};

static_assert(sizeof(OutputWay) == 24, "OutputWay format must remain 24 bytes");

struct OutputType {
    std::uint32_t offset;
    std::uint32_t size;
};

static_assert(sizeof(OutputType) == 8, "OutputType format must remain 8 bytes");

struct OutputIndexNode {
    Id id;
    std::uint64_t entry_offset;
    std::uint64_t entry_count;
};

static_assert(sizeof(OutputIndexNode) == 24, "OutputIndexNode format must remain 24 bytes");

struct OutputIndexEntry {
    std::uint32_t way_index;
    std::uint32_t node_index;
};

static_assert(sizeof(OutputIndexEntry) == 8, "OutputIndexEntry format must remain 8 bytes");

struct OutputIndexRef {
    Id node_id;
    OutputIndexEntry entry;
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
        const char* highway_type = way.tags().get_value_by_key("highway");
        if (highway_type == nullptr) {
            return;
        }

        StoredWay stored_way{
            way.id(),
            static_cast<std::uint32_t>(way_node_ids_.size()),
            static_cast<std::uint32_t>(way.nodes().size()),
            highway_type_id(highway_type),
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

    const std::vector<std::string>& highway_types() const noexcept {
        return highway_types_;
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

    std::uint16_t highway_type_id(const char* highway_type) {
        const auto existing = highway_type_ids_.find(highway_type);
        if (existing != highway_type_ids_.end()) {
            return existing->second;
        }
        if (highway_types_.size() == std::numeric_limits<std::uint16_t>::max()) {
            throw std::runtime_error("too many highway types");
        }
        const auto type_id = static_cast<std::uint16_t>(highway_types_.size());
        highway_types_.emplace_back(highway_type);
        highway_type_ids_.emplace(highway_types_.back(), type_id);
        return type_id;
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
    std::unordered_map<std::string, std::uint16_t> highway_type_ids_;
    std::vector<std::string> highway_types_;
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
    output_ways.reserve(collector.selected_way_indices().size());
    for (const std::size_t source_way_index : collector.selected_way_indices()) {
        const auto& way = collector.ways()[source_way_index];
        if (collector.excluded_way_ids().find(way.id) != collector.excluded_way_ids().end()) {
            continue;
        }
        const auto node_offset = output_nodes.size();
        bool locations_valid = true;
        for (std::uint32_t node_index = 0; node_index < way.node_count; ++node_index) {
            const Id node_id = collector.way_node_ids()[way.node_offset + node_index];
            const LocationRecord* location = find_location(collector.locations(), node_id);
            if (location == nullptr) {
                locations_valid = false;
                break;
            }
            output_nodes.push_back({
                node_id,
                osmium::Location(location->longitude, location->latitude).lon_without_check(),
                osmium::Location(location->longitude, location->latitude).lat_without_check(),
            });
        }
        if (!locations_valid) {
            output_nodes.resize(node_offset);
            continue;
        }

        output_ways.push_back({
            way.id,
            node_offset,
            way.node_count,
            way.highway_type,
        });
    }

    std::vector<OutputIndexRef> index_refs;
    index_refs.reserve(output_nodes.size());
    for (std::size_t way_index = 0; way_index < output_ways.size(); ++way_index) {
        const auto& way = output_ways[way_index];
        for (std::uint32_t node_index = 0; node_index < way.node_count; ++node_index) {
            index_refs.push_back({
                output_nodes[way.node_offset + node_index].id,
                {
                    static_cast<std::uint32_t>(way_index),
                    node_index,
                },
            });
        }
    }
    std::sort(
        index_refs.begin(),
        index_refs.end(),
        [](const OutputIndexRef& first, const OutputIndexRef& second) {
            if (first.node_id != second.node_id) {
                return first.node_id < second.node_id;
            }
            return first.entry.way_index < second.entry.way_index;
        }
    );

    std::vector<OutputIndexNode> index_nodes;
    index_nodes.reserve(index_refs.size());
    std::uint64_t index_entry_offset = 0;
    for (const auto& index_ref : index_refs) {
        if (index_nodes.empty() || index_nodes.back().id != index_ref.node_id) {
            index_nodes.push_back({
                index_ref.node_id,
                index_entry_offset,
                0,
            });
        }
        ++index_nodes.back().entry_count;
        ++index_entry_offset;
    }

    std::vector<OutputType> output_types;
    std::vector<char> type_data;
    output_types.reserve(collector.highway_types().size());
    for (const auto& highway_type : collector.highway_types()) {
        output_types.push_back({
            static_cast<std::uint32_t>(type_data.size()),
            static_cast<std::uint32_t>(highway_type.size()),
        });
        type_data.insert(type_data.end(), highway_type.begin(), highway_type.end());
    }

    write_value(output, static_cast<std::uint64_t>(output_ways.size()));
    write_value(output, static_cast<std::uint64_t>(output_nodes.size()));
    write_value(output, static_cast<std::uint64_t>(index_nodes.size()));
    write_value(output, static_cast<std::uint64_t>(index_refs.size()));
    write_value(output, static_cast<std::uint32_t>(output_types.size()));
    write_value(output, static_cast<std::uint32_t>(type_data.size()));
    write_block(output, output_ways);
    write_block(output, output_nodes);
    write_block(output, output_types);
    if (!type_data.empty()) {
        output.write(type_data.data(), static_cast<std::streamsize>(type_data.size()));
    }
    write_block(output, index_nodes);
    std::vector<OutputIndexEntry> index_entry_buffer;
    index_entry_buffer.reserve(65536);
    for (const auto& index_ref : index_refs) {
        index_entry_buffer.push_back(index_ref.entry);
        if (index_entry_buffer.size() == index_entry_buffer.capacity()) {
            write_block(output, index_entry_buffer);
            index_entry_buffer.clear();
        }
    }
    write_block(output, index_entry_buffer);
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
