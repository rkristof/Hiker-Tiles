import math
import re

from utils import haversine_distance_m


LANDMARK_RULES = (
    ('highway', 'trailhead'),
    ('information', 'guidepost'),
    ('information', 'map'),
    ('information', 'board'),
    ('tourism', 'information'),
)
LANDMARK_MAX_DISTANCE_M = 30
LANDMARK_GRID_SIZE_DEGREES = 0.0005
LANDMARK_TEXT_SCORE = 1
LANDMARK_IDENTITY_SCORE = 0.5
SETTLEMENT_MAX_DISTANCE_M = 5000
SETTLEMENT_PROXIMITY_EXPONENT = 2
SETTLEMENT_GRID_SIZE_DEGREES = 0.05
SETTLEMENT_WEIGHTS = {
    'city': 1,
    'town': 1,
    'village': 0.8,
    'hamlet': 0.4,
}
EXTERNAL_ACCESS_DECAY = 0.5
HIGH_HIGHWAY_ACCESS_BONUS = 1
HIGH_HIGHWAY_TYPES = frozenset((
    'motorway',
    'trunk',
    'primary',
    'secondary',
    'tertiary',
    'unclassified',
    'residential',
))


def landmark_candidate(tags):
    """Return metadata for a high-signal OSM landmark, if applicable."""
    for key, value in LANDMARK_RULES:
        if tags.get(key) == value:
            return {
                'tag_key': key,
                'tag_value': value,
                'name': tags.get('name', ''),
                'description': tags.get('description', ''),
                'ref': tags.get('ref', ''),
            }
    return None


class SpatialIndex:
    """Index point collections so lookups check only nearby candidates."""

    def __init__(self, items, max_distance_m, grid_size_degrees):
        self._items = tuple(items)
        self._max_distance_m = max_distance_m
        self._grid_size_degrees = grid_size_degrees
        self._cells = {}
        for item in self._items:
            for point in item.get('points', ()):
                cell = self._cell(point, self._grid_size_degrees)
                self._cells.setdefault(cell, []).append((item, point))

    @property
    def max_distance_m(self):
        return self._max_distance_m

    def nearby(self, point):
        """Yield indexed points within the maximum distance window."""
        latitude_radius = math.degrees(self._max_distance_m / 6371000)
        longitude_radius = latitude_radius / max(math.cos(math.radians(point[1])), 1e-6)
        latitude_cell, longitude_cell = self._cell(point, self._grid_size_degrees)
        latitude_range = range(
            latitude_cell - math.ceil(latitude_radius / self._grid_size_degrees),
            latitude_cell + math.ceil(latitude_radius / self._grid_size_degrees) + 1,
        )
        longitude_range = range(
            longitude_cell - math.ceil(longitude_radius / self._grid_size_degrees),
            longitude_cell + math.ceil(longitude_radius / self._grid_size_degrees) + 1,
        )
        for latitude_index in latitude_range:
            for longitude_index in longitude_range:
                yield from self._cells.get((latitude_index, longitude_index), ())

    @staticmethod
    def _cell(point, grid_size_degrees):
        return (
            math.floor(point[1] / grid_size_degrees),
            math.floor(point[0] / grid_size_degrees),
        )


class LandmarkIndex(SpatialIndex):
    """Index landmark points so matching checks only nearby candidates."""

    def __init__(self, landmarks):
        super().__init__(
            landmarks,
            LANDMARK_MAX_DISTANCE_M,
            LANDMARK_GRID_SIZE_DEGREES,
        )


class SettlementIndex(SpatialIndex):
    """Index settlement points within the settlement scoring radius."""

    def __init__(self, settlements):
        super().__init__(
            (
                settlement
                for settlement in settlements
                if settlement.get('place') in SETTLEMENT_WEIGHTS
            ),
            SETTLEMENT_MAX_DISTANCE_M,
            SETTLEMENT_GRID_SIZE_DEGREES,
        )


class EligibleNodeFinder:
    """Find and rank route nodes with directly connected highways."""

    MAX_RANKED_NODES = 1

    def __init__(
        self,
        route_relation,
        way_nodes,
        connecting_highways_by_node,
        candidate_node_ids,
        landmark_index=None,
        settlement_index=None,
    ):
        self._route_relation = route_relation
        self._way_nodes = way_nodes
        self._connecting_highways_by_node = connecting_highways_by_node
        self._candidate_node_ids = set(candidate_node_ids)
        self._route_way_ids = set(route_relation.get('way_ids', ()))
        self._landmark_index = landmark_index
        self._settlement_index = settlement_index
        self._relation_node_order = {}

        for node_id in route_relation.get('node_ids', ()):
            self._relation_node_order.setdefault(
                node_id,
                len(self._relation_node_order),
            )
        for way_id in route_relation.get('way_ids', ()):
            for node_id, _ in way_nodes.get(way_id, ()):
                self._relation_node_order.setdefault(
                    node_id,
                    len(self._relation_node_order),
                )

    def externally_accessible_nodes(self):
        """Return candidate route nodes touched by an outside highway way."""
        eligible_nodes = {
            node_id
            for node_id in self._candidate_node_ids
            if self._external_highways(node_id)
        }
        return eligible_nodes

    def rank_eligible_nodes(self):
        """Return externally accessible nodes ordered by weighted start score."""
        eligible_nodes = self.externally_accessible_nodes()
        landmark_scores = self._landmark_scores(eligible_nodes)
        settlement_scores = self._settlement_scores(eligible_nodes)
        external_scores = {
            node_id: self._external_access_score(node_id)
            for node_id in eligible_nodes
        }
        maximum_external_score = max(external_scores.values(), default=0)
        return sorted(
            eligible_nodes,
            key=lambda node_id: self._endpoint_score(
                node_id,
                landmark_scores.get(node_id, 0),
                settlement_scores.get(node_id, 0),
                external_scores[node_id],
                maximum_external_score,
            ),
            reverse=True,
        )[:self.MAX_RANKED_NODES]

    def _landmark_scores(self, node_ids=None):
        if self._landmark_index is None:
            return {}
        route_tokens = self._text_token_list(
            self._route_relation.get('name', ''),
            self._route_relation.get('name_int', ''),
        )

        landmark_scores = {}
        for node_id in self._candidate_node_ids if node_ids is None else node_ids:
            node_points = [
                point
                for way_id in self._route_relation.get('way_ids', ())
                for candidate_node_id, point in self._way_nodes.get(way_id, ())
                if candidate_node_id == node_id
            ]
            for point in node_points:
                for landmark, landmark_point in self._landmark_index.nearby(point):
                    landmark_score = 0
                    landmark_multiplier = self._landmark_score_multiplier(landmark)
                    if landmark.get('node_id') == node_id:
                        landmark_score = LANDMARK_IDENTITY_SCORE
                    if route_tokens and self._matches_landmark(
                        point,
                        landmark_point,
                        route_tokens,
                        landmark,
                    ):
                        landmark_score = max(landmark_score, LANDMARK_TEXT_SCORE)
                    if landmark_score:
                        landmark_scores[node_id] = max(
                            landmark_scores.get(node_id, 0),
                            landmark_score * landmark_multiplier,
                        )
        return landmark_scores

    @staticmethod
    def _landmark_score_multiplier(landmark):
        for value in ('name', 'description'):
            match = re.match(r'^\s*(\d+)\.', landmark.get(value, ''))
            if match:
                return max(0.1, 1 - (int(match.group(1)) - 1) * 0.1)
        return 1

    def _settlement_scores(self, node_ids=None):
        if self._settlement_index is None:
            return {}
        scores = {}
        for node_id in self._candidate_node_ids if node_ids is None else node_ids:
            for point in self._node_points(node_id):
                for settlement, settlement_point in self._settlement_index.nearby(point):
                    distance = haversine_distance_m(point, settlement_point)
                    proximity = max(
                        0,
                        1 - distance / self._settlement_index.max_distance_m,
                    )
                    score = SETTLEMENT_WEIGHTS[settlement['place']] * proximity ** SETTLEMENT_PROXIMITY_EXPONENT
                    if score:
                        scores[node_id] = max(scores.get(node_id, 0), score)
        return scores

    def _matches_landmark(self, route_point, landmark_point, route_tokens, landmark):
        if haversine_distance_m(route_point, landmark_point) > LANDMARK_MAX_DISTANCE_M:
            return False
        landmark_tokens = set(
            self._text_token_list(
                landmark.get('name', ''),
                landmark.get('description', ''),
                landmark.get('ref', ''),
            )
        )
        return len(set(route_tokens) & landmark_tokens) >= 2

    def _endpoint_score(
        self,
        node_id,
        landmark_score,
        settlement_score,
        external_access_score,
        maximum_external_score,
    ):
        order = self._relation_node_order[node_id]
        maximum_order = max(self._relation_node_order.values(), default=0)
        order_score = (
            1
            if maximum_order == 0
            else 1 - order / maximum_order
        )
        external_score = (
            external_access_score / maximum_external_score
            if maximum_external_score
            else 0
        )
        route_degree_score = float(self._route_degree(node_id) == 1)
        return (
            30 * landmark_score
            + 30 * settlement_score
            + 15 * route_degree_score
            + 5 * order_score
            + 20 * external_score
        )

    def _external_access_score(self, node_id):
        external_highways = self._external_highways(node_id)
        access_score = sum(
            EXTERNAL_ACCESS_DECAY ** index
            for index in range(len(external_highways))
        )
        high_highway_count = sum(
            highway_type in HIGH_HIGHWAY_TYPES
            for highway_type in external_highways.values()
        )
        high_highway_score = HIGH_HIGHWAY_ACCESS_BONUS * sum(
            EXTERNAL_ACCESS_DECAY ** index
            for index in range(high_highway_count)
        )
        return access_score + high_highway_score

    def _external_highways(self, node_id):
        return {
            way_id: highway['highway_type']
            for way_id, highway in self._connecting_highways_by_node.get(node_id, {}).items()
            if way_id not in self._route_way_ids
        }

    def _route_degree(self, node_id):
        neighbors = set()
        for way_id in self._route_relation.get('way_ids', ()):
            nodes = self._way_nodes.get(way_id, ())
            for index, (candidate_node_id, _) in enumerate(nodes):
                if candidate_node_id != node_id:
                    continue
                if index > 0 and nodes[index - 1][0] != node_id:
                    neighbors.add(nodes[index - 1][0])
                if index + 1 < len(nodes) and nodes[index + 1][0] != node_id:
                    neighbors.add(nodes[index + 1][0])
        return len(neighbors)

    def _node_points(self, node_id):
        return [
            point
            for way_id in self._route_relation.get('way_ids', ())
            for candidate_node_id, point in self._way_nodes.get(way_id, ())
            if candidate_node_id == node_id
        ]

    @staticmethod
    def _text_token_list(*values):
        return [
            token.casefold()
            for value in values
            for token in re.findall(r'\w+', value, flags=re.UNICODE)
            if len(token) >= 3
        ]
