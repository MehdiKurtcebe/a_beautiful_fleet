import subprocess
import shutil
import csv
import sys
import geopandas as gpd
import random
from shapely.geometry import Point, Polygon
import os
import geojson
import folium
from folium.plugins import AntPath  # Import AntPath for animated paths

if len(sys.argv) > 1:
    # Use the filename provided as a command-line argument
    filepath_argv = 'datasets/cplex/' + sys.argv[1] 
else:
    filepath_argv = "datasets/cplex/testData1.dat"  # Default filename in the datasets directory
    print("No input parameter provided. Using default: testData1.dat")
    #sys.exit(1) 


def copy_dat_file(input_path, output_path):
    """Copy the .dat file from input to output path as binary."""
    with open(input_path, 'rb') as src_file:
        with open(output_path, 'wb') as dest_file:
            shutil.copyfileobj(src_file, dest_file)
    print(f"Copied {input_path} to {output_path} as binary.")

def get_beautificators_value(file_path):
  """
  Extracts the value of 'beautificators' from a .dat file.

  Args:
    file_path: Path to the .dat file.

  Returns:
    The integer value of 'beautificators' or None if not found.
  """
  with open(file_path, 'r') as f:
    for line in f:
      if line.startswith('beautificators'):
        try:
          # Split the line by '=' and extract the value
          value = int(line.split('=')[1].strip().rstrip(';'))
          return value
        except (IndexError, ValueError):
          print(f"Error parsing line: {line}")
          return None
  return None

def run_cplex(run_config):
  """
  Runs an OPL model using the 'oplrun' command as a subprocess.

  Args:
    model_path: Path to the .mod file.
    data_path: Path to the .dat file.
  """
  try:
    # Construct the oplrun command
    command = ["oplrun", "-p", "./cplex-files", run_config]

    # Execute the command as a subprocess
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Get the output and error messages
    stdout, stderr = process.communicate()

    # Decode the output and error messages
    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")

    # Print the output and error messages
    print("Output:\n", stdout)
    if stderr:
      print("Error:\n", stderr)

  except FileNotFoundError:
    print("Error: oplrun command not found. Make sure CPLEX is installed and configured correctly.")
  except Exception as e:
    print("Error:", e)


# Paths to your files
input_dat_file = filepath_argv
model_data = 'cplex-files/model.dat'
run_config = 'model'

# Copy the binary data from testData.dat to data.dat
copy_dat_file(input_dat_file, model_data)

beautificators = get_beautificators_value(input_dat_file)

# Run the CPLEX model
run_cplex(run_config)


# Load the GeoJSON file containing the zones
script_dir = os.path.dirname(os.path.realpath(__file__))  # Get script's directory
geojson_file = os.path.join(script_dir, 'aco', 'map-izmit.geojson')  # Construct absolute path
gdf = gpd.read_file(geojson_file)

#List of CSV files, each csv is output for beatuficator
csv_files = []  # Extend this list for all output files

for i in range(beautificators):
    temp_str = "cplex_bf_" + str(i+1) + ".csv"
    csv_files.append(temp_str)


def generate_random_points(polygon, num_points):
    points = []
    min_x, min_y, max_x, max_y = polygon.bounds
    while len(points) < num_points:
        random_point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if polygon.contains(random_point):
            points.append(random_point)
    return points


for i in range(0,16):
    output_map_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 
        "maps", 
        f"cplex_bf_{i}_route.html"
    )

    # Check if the file exists before attempting to delete it
    if os.path.exists(output_map_file):
        os.remove(output_map_file)
        print(f"File deleted: {output_map_file}")
    else:
        print(f"File not found: {output_map_file}") 

script_dir = os.path.dirname(os.path.realpath(__file__))  # Get script's directory
csv_outputs_dir = os.path.join(script_dir, 'cplex-files', 'results')  # Path to csv-outputs dir

bf_num = 0

# Initialize variables to store the first "HOT"/"BEAU" and last "MOVE" action
first_zone = None
last_move_zone = None


bf_num = 1

for csv_file in csv_files:
    csv_file_path = os.path.join(csv_outputs_dir, csv_file)
    # 1. Data Preparation
    with open(csv_file_path, 'r') as f:
        reader = csv.DictReader(f)
        route_data = list(reader)

    # Load the GeoJSON file containing the zones
    script_dir = os.path.dirname(os.path.realpath(__file__))  # Get script's directory
    geojson_file = os.path.join(script_dir, 'aco', 'map-izmit.geojson')  # Construct absolute path
    with open(geojson_file) as f:
        zones = geojson.load(f)

    def get_centroid(zone_feature):
        # Calculate centroid of a polygon (you might need a more robust method)
        coords = zone_feature['geometry']['coordinates'][0]
        x, y = zip(*coords)
        return (sum(x) / len(x), sum(y) / len(y))

    zone_centroids = {}
    for feature in zones['features']:
        zone_id = feature['properties']['name'].replace('Area ', '')  # Assuming 'Area 1' format
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

    for i in range(len(route_data)):
        row = route_data[i]
        from_zone = str(row['Zone_From'])
        action = row['Action']

        if action in ("BEAU", "HOT"):
            zone_polygon = next((f['geometry']['coordinates'][0] for f in zones['features'] if f['properties']['name'] == f"Area {from_zone}"), None)
            if zone_polygon:
                if isMove:
                    point = pointFromMove
                    isMove = False
                    randomPointBefore = None
                else:
                    point = get_random_point_in_polygon(zone_polygon)
                if randomPointBefore is not None:
                    AntPath([randomPointBefore, point], color="red", weight=2.5, opacity=1).add_to(m)

                randomPointBefore = point
                route_coordinates.append(point)
                last_point_before_move = point

                # Add marker for BEAU or HOT action
                if action == "BEAU":
                    folium.Marker(location=point, popup=f"BEAU - Area {from_zone}", icon=folium.Icon(color=colorBeau)).add_to(m)  # Blue marker for BEAU
                elif action == "HOT":
                    folium.Marker(location=point, popup=f"HOT - Area {from_zone}", icon=folium.Icon(color=colorHot)).add_to(m)  # Red marker for HOT

                if isFirstAction:
                    isFirstAction = False
                    colorHot = "red"
                    colorBeau = "blue"

        elif action == "MOVE" and last_point_before_move:
            next_row = route_data[i + 1] if i + 1 < len(route_data) else None
            if next_row:
                next_zone = str(next_row['Zone_From'])
                next_action = next_row['Action']

                if next_action in ("BEAU", "HOT"):
                    next_zone_polygon = next((f['geometry']['coordinates'][0] for f in zones['features'] if f['properties']['name'] == f"Area {next_zone}"), None)
                    if next_zone_polygon:
                        next_point = get_random_point_in_polygon(next_zone_polygon)
                        route_coordinates.append(next_point)

                        # Draw an animated path (AntPath) between the last point before move and the new point
                        AntPath([last_point_before_move, next_point], color="red", weight=2.5, opacity=1).add_to(m)
                        pointFromMove = next_point
                        isMove = True

    # 3. Visualization
    # Add zones to the map
    folium.GeoJson(zones, 
                style_function=lambda feature: {'fillColor': 'lightblue', 'color': 'blue', 'weight': 2},
                tooltip=folium.GeoJsonTooltip(fields=['name'], aliases=['Zone: '], labels=True, sticky=True)
                ).add_to(m)

    # Add zone names as markers at the centroids
    for zone_id, centroid in zone_centroids.items():
        folium.Marker(location=[centroid[1], centroid[0]], 
                    icon=folium.DivIcon(html=f'<div style="font-weight: bold;">{zone_id}</div>'),
                    popup=f"Zone {zone_id}"
                    ).add_to(m)

    #Save the plot with a dynamic filename based on the CSV file
    output_html_name = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 
        "maps", 
        f"cplex_bf_{bf_num}_route.html"
    )
    m.save(output_html_name)

    bf_num += 1


