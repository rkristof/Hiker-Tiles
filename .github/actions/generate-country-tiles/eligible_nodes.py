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


class LandmarkIndex:
    """Index landmark points so matching checks only nearby candidates."""

    def __init__(self, landmarks):
        self._landmarks = tuple(landmarks)
        self._cells = {}
        for landmark in self._landmarks:
            for point in landmark.get('points', ()):
                cell = self._cell(point)
            self._cells.setdefault(cell, []).append((landmark, point))

    def nearby(self, point):
        """Yield indexed landmark points within the maximum distance window."""
        latitude_radius = math.degrees(LANDMARK_MAX_DISTANCE_M / 6371000)
        longitude_radius = latitude_radius / max(math.cos(math.radians(point[1])), 1e-6)
        latitude_cell, longitude_cell = self._cell(point)
        latitude_range = range(
            latitude_cell - math.ceil(latitude_radius / LANDMARK_GRID_SIZE_DEGREES),
            latitude_cell + math.ceil(latitude_radius / LANDMARK_GRID_SIZE_DEGREES) + 1,
        )
        longitude_range = range(
            longitude_cell - math.ceil(longitude_radius / LANDMARK_GRID_SIZE_DEGREES),
            longitude_cell + math.ceil(longitude_radius / LANDMARK_GRID_SIZE_DEGREES) + 1,
        )
        for latitude_index in latitude_range:
            for longitude_index in longitude_range:
                yield from self._cells.get((latitude_index, longitude_index), ())

    @staticmethod
    def _cell(point):
        return (
            math.floor(point[1] / LANDMARK_GRID_SIZE_DEGREES),
            math.floor(point[0] / LANDMARK_GRID_SIZE_DEGREES),
        )


class EligibleNodeFinder:
    """Find and rank externally accessible route nodes."""

    MAX_RANKED_NODES = 10

    def __init__(
        self,
        route_relation,
        way_nodes,
        highway_way_ids_by_node,
        candidate_node_ids,
        landmarks=None,
    ):
        self._route_relation = route_relation
        self._way_nodes = way_nodes
        self._highway_way_ids_by_node = highway_way_ids_by_node
        self._candidate_node_ids = set(candidate_node_ids)
        self._route_way_ids = set(route_relation.get('way_ids', ()))
        self._landmarks = tuple(landmarks or ())
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
        return {
            node_id
            for node_id in self._candidate_node_ids
            if any(
                way_id not in self._route_way_ids
                for way_id in self._highway_way_ids_by_node.get(node_id, ())
            )
        }

    def rank_eligible_nodes(self, is_start=True):
        """Return up to 50 externally accessible nodes ordered by endpoint score."""
        eligible_nodes = self.externally_accessible_nodes()
        landmark_nodes = self._landmark_nodes()
        return sorted(
            eligible_nodes,
            key=lambda node_id: self._endpoint_score(
                node_id,
                landmark_nodes,
                is_start,
            ),
            reverse=True,
        )[:self.MAX_RANKED_NODES]

    def _landmark_nodes(self):
        route_tokens = self._text_token_list(
            self._route_relation.get('name', ''),
            self._route_relation.get('name_int', ''),
        )
        if not route_tokens:
            return set()

        landmark_index = LandmarkIndex(self._landmarks)
        landmark_nodes = set()
        for node_id in self._candidate_node_ids:
            node_points = [
                point
                for way_id in self._route_relation.get('way_ids', ())
                for candidate_node_id, point in self._way_nodes.get(way_id, ())
                if candidate_node_id == node_id
            ]
            for point in node_points:
                for landmark, landmark_point in landmark_index.nearby(point):
                    if self._matches_landmark(point, landmark_point, route_tokens, landmark):
                        landmark_nodes.add(node_id)
                        break
                if node_id in landmark_nodes:
                    break
        return landmark_nodes

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

    def _endpoint_score(self, node_id, landmark_nodes, is_start):
        order = self._relation_node_order[node_id]
        return (node_id in landmark_nodes, -order if is_start else order)

    @staticmethod
    def _text_token_list(*values):
        return [
            token.casefold()
            for value in values
            for token in re.findall(r'\w+', value, flags=re.UNICODE)
            if len(token) >= 3
        ]
