import csv
import geojson
import folium
import random
from folium.plugins import AntPath  # Import AntPath for animated paths
from shapely.geometry import Polygon, Point
from folium.plugins import TimestampedGeoJson  # Use TimestampedGeoJson for animation
from datetime import datetime, timedelta

# 1. Data Preparation
with open("output_7.csv", "r") as f:
    reader = csv.DictReader(f)
    route_data = list(reader)

with open("map-izmit.geojson") as f:
    zones = geojson.load(f)


def get_centroid(zone_feature):
    # Calculate centroid of a polygon (you might need a more robust method)
    coords = zone_feature["geometry"]["coordinates"][0]
    x, y = zip(*coords)
    return (sum(x) / len(x), sum(y) / len(y))


zone_centroids = {}
for feature in zones["features"]:
    zone_id = feature["properties"]["name"].replace(
        "Area ", ""
    )  # Assuming 'Area 1' format
    zone_centroids[zone_id] = get_centroid(feature)


def get_random_point_in_polygon(polygon_coords):
    """Generates a random point within a polygon."""
    polygon = Polygon(polygon_coords)
    minx, miny, maxx, maxy = polygon.bounds
    while True:
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(p):
            return (p.y, p.x)  # Return in (latitude, longitude) order


# 2. Route Calculation (with random points and move actions)
m = folium.Map(location=[40.764, 29.922], zoom_start=15)

route_coordinates = []
last_point_before_move = None
randomPointBefore = None
isMove = False
pointFromMove = None
isFirstAction = True

colorHot = "black"
colorBeau = "black"

features = []
base_timestamp = datetime(2017, 9, 14, 13, 24, 40)  # Adjust as needed

for i in range(len(route_data)):
    row = route_data[i]
    from_zone = str(row["Zone_From"])
    action = row["Action"]

    # Calculate timestamp and format it
    timestamp = base_timestamp + timedelta(
        minutes=i * 5
    )  # Adjust time increment as needed
    timestamp_str = timestamp.strftime("%Y-%m-%dT%H:%M:%S")

    if action in ("BEAU", "HOT"):
        zone_polygon = next(
            (
                f["geometry"]["coordinates"][0]
                for f in zones["features"]
                if f["properties"]["name"] == f"Area {from_zone}"
            ),
            None,
        )
        if zone_polygon:
            if isMove:
                point = pointFromMove
                isMove = False
                randomPointBefore = None
            else:
                point = get_random_point_in_polygon(zone_polygon)

            if randomPointBefore is not None:
                # Add the path with timestamp as a feature
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [randomPointBefore, point],
                        },
                        "properties": {
                            "times": [previous_timestamp_str, timestamp_str],
                            "style": {"color": "red", "weight": 2.5, "opacity": 1},
                        },
                    }
                )

            randomPointBefore = point
            previous_timestamp_str = timestamp_str

            # Add marker for BEAU or HOT action
            if action == "BEAU":
                folium.Marker(
                    location=point,
                    popup=f"BEAU - Area {from_zone}",
                    icon=folium.Icon(color=colorBeau),
                ).add_to(m)  # Blue marker for BEAU
            elif action == "HOT":
                folium.Marker(
                    location=point,
                    popup=f"HOT - Area {from_zone}",
                    icon=folium.Icon(color=colorHot),
                ).add_to(m)  # Red marker for HOT

            if isFirstAction:
                isFirstAction = False
                colorHot = "red"
                colorBeau = "blue"

    elif action == "MOVE" and last_point_before_move:
        next_row = route_data[i + 1] if i + 1 < len(route_data) else None
        if next_row:
            next_zone = str(next_row["Zone_From"])
            next_action = next_row["Action"]

            if next_action in ("BEAU", "HOT"):
                next_zone_polygon = next(
                    (
                        f["geometry"]["coordinates"][0]
                        for f in zones["features"]
                        if f["properties"]["name"] == f"Area {next_zone}"
                    ),
                    None,
                )
                if next_zone_polygon:
                    next_point = get_random_point_in_polygon(next_zone_polygon)
                    route_coordinates.append(next_point)

                    # Draw an animated path (AntPath) between the last point before move and the new point
                    AntPath(
                        [last_point_before_move, next_point],
                        color="red",
                        weight=2.5,
                        opacity=1,
                    ).add_to(m)
                    pointFromMove = next_point
                    isMove = True

# 3. Visualization
# Add zones to the map
folium.GeoJson(
    zones,
    style_function=lambda feature: {
        "fillColor": "lightblue",
        "color": "blue",
        "weight": 2,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["name"], aliases=["Zone: "], labels=True, sticky=True
    ),
).add_to(m)

# Add zone names as markers at the centroids
for zone_id, centroid in zone_centroids.items():
    folium.Marker(
        location=[centroid[1], centroid[0]],
        icon=folium.DivIcon(html=f'<div style="font-weight: bold;">{zone_id}</div>'),
        popup=f"Zone {zone_id}",
    ).add_to(m)

# 3. Add TimestampedGeoJson to the map
TimestampedGeoJson(
    {
        "type": "FeatureCollection",
        "features": features,
    },
    period="PT1M",  # Adjust the period as needed
    add_last_point=True,
).add_to(m)

m

# 4. HTML Output
m.save("route_map2.html")
