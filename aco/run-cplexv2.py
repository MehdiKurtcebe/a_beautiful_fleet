import subprocess
import shutil
import numpy as np
import csv
import time
import re
import sys
import geopandas as gpd
import random
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import os

if len(sys.argv) > 1:
    # Use the filename provided as a command-line argument
    filepath_argv = 'datasets/cplex/' + sys.argv[1] 
else:
    filepath_argv = "datasets/cplex/testData-cplex.dat"  # Default filename in the datasets directory
    print("No input parameter provided. Using default: testData-cplex.dat")
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

def run_cplex(model_path, data_path):
  """
  Runs an OPL model using the 'oplrun' command as a subprocess.

  Args:
    model_path: Path to the .mod file.
    data_path: Path to the .dat file.
  """
  try:
    # Construct the oplrun command
    command = ["oplrun", model_path, data_path]

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
model_file = 'cplex-files/model.mod'

# Copy the binary data from testData.dat to data.dat
copy_dat_file(input_dat_file, model_data)

beautificators = get_beautificators_value(input_dat_file)

# Run the CPLEX model
run_cplex(model_file, model_data)


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

script_dir = os.path.dirname(os.path.realpath(__file__))  # Get script's directory
csv_outputs_dir = os.path.join(script_dir, 'cplex-files', 'results')  # Path to csv-outputs dir

bf_num = 0

for csv_file in csv_files:
    
    bf_num += 1

    csv_file_path = os.path.join(csv_outputs_dir, csv_file)  # Absolute path to the CSV file

    df = pd.read_csv(csv_file_path) 
    
    fig, ax = plt.subplots(figsize=(12, 8))
    

    points_per_action = {}

    
    for index, row in df.iterrows():
        zone_from = row['Zone_From']
        zone_to = row['Zone_To']
        action = row['Action']
        time_from = row['Time_From']
        time_to = row['Time_To']
        
        #Plot HOT and BEAU points in the zones
        if action == 'HOT' or action == 'BEAU':
            zone_polygon = gdf[gdf['name'] == f'Area {zone_from}'].geometry.iloc[0]
            random_points = generate_random_points(zone_polygon, 1)
            points_gdf = gpd.GeoDataFrame({'geometry': random_points}, crs=gdf.crs)
            
            #Store the points for this action (with index)
            points_per_action[index] = random_points
            
            #Plot all points for HOT and BEAU actions
            if action == 'HOT':
                points_gdf.plot(ax=ax, color='red', markersize=50)
            elif action == 'BEAU':
                points_gdf.plot(ax=ax, color='blue', markersize=50)

        #Plot MOVE arrows
        if action == 'MOVE':
            zone_from_polygon = gdf[gdf['name'] == f'Area {zone_from}'].geometry.iloc[0]
            zone_to_polygon = gdf[gdf['name'] == f'Area {zone_to}'].geometry.iloc[0]
            
            #Draw an arrow from zone_from to zone_to
            line = LineString([zone_from_polygon.centroid, zone_to_polygon.centroid])
            ax.plot(*line.xy, color='green', linewidth=2, marker='o')

    #Plot all zones in the background
    gdf.plot(ax=ax, color='lightgrey', edgecolor='black', alpha=0.5)

    #Draw lines between consecutive actions (index-based) only
    for i in range(1, len(df)):
        if df.iloc[i]['Action'] in ['HOT', 'BEAU'] and df.iloc[i-1]['Action'] in ['HOT', 'BEAU']:
            points_from = points_per_action.get(i-1, [])
            points_to = points_per_action.get(i, [])
            
            # If both have points, draw lines between them (first point from the previous and first point from current)
            if points_from and points_to:
                line = LineString([points_from[0], points_to[0]])
                ax.plot(*line.xy, color='black', linewidth=1)

    #Markers
    hot_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='HOT')
    beau_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='BEAU')
    ax.legend(handles=[hot_patch, beau_patch], loc='best')

    #Add the names of each zone to the plot
    for idx, row in gdf.iterrows():
        #Get the centroid of the zone
        zone_centroid = row['geometry'].centroid
        #Plot the zone name at the centroid
        ax.text(zone_centroid.x, zone_centroid.y, row['name'], fontsize=10, ha='center', color='black')

    #Customize the plot
    plt.title(f"CPLEX - Route of a Beautificator ({bf_num})")
    
    #Save the plot with a dynamic filename based on the CSV file
    output_image_name = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 
        "maps", 
        f"cplex_bf_{bf_num}_route.png"
    )

    # Save the plot 
    plt.savefig(output_image_name)
    plt.close()  
