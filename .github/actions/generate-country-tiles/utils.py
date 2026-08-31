import math


def haversine_distance_m(first_point, second_point):
    earth_radius_m = 6371000
    first_latitude = math.radians(first_point[1])
    second_latitude = math.radians(second_point[1])
    delta_latitude = second_latitude - first_latitude
    delta_longitude = math.radians(second_point[0] - first_point[0])
    haversine_term = (
        math.sin(delta_latitude / 2) ** 2
        + math.cos(first_latitude)
        * math.cos(second_latitude)
        * math.sin(delta_longitude / 2) ** 2
    )
    return earth_radius_m * 2 * math.asin(math.sqrt(haversine_term))


def polyline_distance_m(points):
    return sum(
        haversine_distance_m(first_point, second_point)
        for first_point, second_point in zip(points, points[1:])
    )
