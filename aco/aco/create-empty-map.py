import geojson
import folium

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

# 2. Route Calculation (with random points and move actions)
m = folium.Map(location=[40.764, 29.922], zoom_start=15)


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


# 4. HTML Output
m.save("1-empty-map.html")
