import numpy as np
import csv
import time
import re
import sys
import geopandas as gpd
import random
from shapely.geometry import Point, Polygon
import os
import geojson
import folium
from folium.plugins import AntPath  # Import AntPath for animated paths

def read_dat_file(filepath):
    """Reads a .dat file and returns the data."""

    # Get the absolute path to the datasets directory
    script_dir = os.path.dirname(os.path.realpath(__file__))
    datasets_dir = os.path.join(script_dir, '..', 'datasets', 'aco') 

    # Construct the absolute path to the data file
    filepath = os.path.join(datasets_dir, filepath)  

    variables = {}
    with open(filepath, 'r') as f:
        content = f.read()

        for match in re.finditer(r"(\w+)\s*=\s*(.*?);", content, re.DOTALL):
            name = match.group(1).strip()
            value_str = match.group(2).strip()

            if '[' in value_str:
                value_str = value_str.replace('[', '').replace(']', '').strip()
                if ';' in value_str:  # Matrix
                    rows = [row.strip() for row in value_str.split(';')]
                    value = []
                    for row in rows:
                        try:
                            inner_values = [int(x.strip()) for x in row.split(',')]
                            value.append(inner_values)
                        except ValueError as e:
                            print(f"Error parsing row: {row}. Error: {e}")
                            raise
                    try:
                        value = np.array(value, dtype=int)
                    except ValueError as e:
                        print(f"Error converting to NumPy array: {e}")
                        print("Value:", value)
                        raise
                elif ',' in value_str:  # List
                    value = [int(x.strip()) for x in value_str.split(',')]
                else:  # Single integer in brackets
                    try:
                        value = int(value_str)
                    except ValueError:
                        pass
            else:
                try:
                    value = int(value_str)
                except ValueError:
                    try:
                        value = float(value_str)
                    except ValueError:
                        value = value_str

            variables[name] = value
    return variables


if len(sys.argv) > 1:
    # Use the filename provided as a command-line argument
    filepath = sys.argv[1] 
else:
    filepath = "testData1.dat"  # Default filename in the datasets directory
    print("No input parameter provided. Using default: testData1.dat")
    #sys.exit(1) 

data = read_dat_file(filepath)


zones = data['zones']			
beautificators = data['beautificators']
t_max = data['t_max']	

if( type(data['z0']) is int ):
    z0 = []
    z0.append(data['z0'])

else:
    z0 = data['z0']



nOUT = data['nOUT']
nHOT = data['nHOT']


mBEAU = data['mBEAU']
mHOT = data['mHOT']
mWAIT = data['mWAIT']

mMOVE = [	
  [0, 1, 2, 3, 5, 6, 7, 8, 2, 3, 4, 6, 7, 4, 5, 6],
  [1, 0, 1, 2, 4, 5, 6, 7, 1, 2, 3, 5, 6, 3, 4, 5],
  [2, 1, 0, 1, 3, 4, 5, 6, 2, 1, 2, 4, 5, 4, 3, 4],
  [3, 2, 1, 0, 2, 3, 4, 5, 3, 2, 1, 3, 4, 5, 4, 3],
  [5, 4, 3, 2, 0, 1, 2, 3, 5, 4, 3, 1, 2, 7, 6, 5],
  [6, 5, 4, 3, 1, 0, 1, 2, 6, 5, 4, 2, 1, 8, 7, 6],
  [7, 6, 5, 4, 2, 1, 0, 1, 7, 6, 5, 3, 2, 9, 8, 7],
  [8, 7, 6, 5, 3, 2, 1, 0, 8, 7, 6, 4, 3,10, 9, 8],
  [2, 1, 2, 3, 5, 6, 7, 8, 0, 1, 2, 4, 5, 2, 3, 4],
  [3, 2, 1, 2, 4, 5, 6, 7, 1, 0, 1, 3, 4, 3, 2, 3],
  [4, 3, 2, 1, 3, 4, 5, 6, 2, 1, 0, 2, 3, 4, 3, 2],
  [6, 5, 4, 3, 1, 2, 3, 4, 4, 3, 2, 0, 1, 6, 5, 4],
  [7, 6, 5, 4, 2, 1, 2, 3, 5, 4, 3, 1, 0, 7, 6, 5],
  [4, 3, 4, 5, 7, 8, 9,10, 2, 3, 4, 6, 7, 0, 1, 2],
  [5, 4, 3, 4, 6, 7, 8, 9, 3, 2, 3, 5, 6, 1, 0, 1],
  [6, 5, 4, 3, 5, 6, 7, 8, 4, 3, 2, 4, 5, 2, 1, 0]
]

piMOVE = [	
  [  0,  -5, -10, -15, -25, -30, -35, -40, -10, -15, -20, -30, -35, -20, -25, -30],
  [ -5,   0,  -5, -10, -20, -25, -30, -35,  -5, -10, -15, -25, -30, -15, -20, -25],
  [-10,  -5,   0,  -5, -15, -20, -25, -30, -10,  -5, -10, -20, -25, -20, -15, -20],
  [-15, -10,  -5,   0, -10, -15, -20, -25, -15, -10,  -5, -15, -20, -25, -20, -15],
  [-25, -20, -15, -10,   0,  -5, -10, -15, -25, -20, -15,  -5, -10, -35, -30, -25],
  [-30, -25, -20, -15,  -5,   0,  -5, -10, -30, -25, -20, -10,  -5, -40, -35, -30],
  [-35, -30, -25, -20, -10,  -5,   0,  -5, -35, -30, -25, -15, -10, -45, -40, -35],
  [-40, -35, -30, -25, -15, -10,  -5,   0, -40, -35, -30, -20, -15, -50, -45, -40],
  [-10,  -5, -10, -15, -25, -30, -35, -40,   0,  -5, -10, -20, -25, -10, -15, -20],
  [-15, -10,  -5, -10, -20, -25, -30, -35,  -5,   0,  -5, -15, -20, -15, -10, -15],
  [-20, -15, -10,  -5, -15, -20, -25, -30, -10,  -5,   0, -10, -15, -20, -15, -10],
  [-30, -25, -20, -15,  -5, -10, -15, -20, -20, -15, -10,   0,  -5, -30, -25, -20],
  [-35, -30, -25, -20, -10,  -5, -10, -15, -25, -20, -15,  -5,   0, -35, -30, -25],
  [-20, -15, -20, -25, -35, -40, -45, -50, -10, -15, -20, -30, -35,   0,  -5, -10],
  [-25, -20, -15, -20, -30, -35, -40, -45, -15, -10, -15, -25, -30,  -5,   0,  -5],
  [-30, -25, -20, -15, -25, -30, -35, -40, -20, -15, -10, -20, -25, -10,  -5,   0]
]


profits = {"BEAU": 5.0, "HOT": 10.0, "WAIT": 0.0, "MOVE": np.array(piMOVE)}  #be sure movement is negative


num_ants = 100
num_iterations = 50
evaporation_rate = 0.02
pheromone_importance = 1.0
heuristic_importance = 1.0

#Pheromone Initialization with HashMap
pheromones = {}
for z1 in range(1, zones + 1):
    for z2 in range(1, zones + 1):
        for aco_time in range(0, t_max):
            pheromones[(z1, z2, aco_time)] = {
                "MOVE": 1.0 + 0.1 * max(0, profits["MOVE"][z1 - 1][z2 - 1]),
                "BEAU": 1.0 + 0.1 * profits["BEAU"] * (t_max/mBEAU), 
                "HOT": 1.0 + 0.1 * profits["HOT"] * (t_max/mBEAU), 
                "WAIT": 1.0 
            }


class Ant:
    def __init__(self):
        self.paths = [[] for _ in range(beautificators)]  #Store paths for each beautificator
        self.total_profits = [0] * beautificators
        self.action_durations = [0] * beautificators  #Track action durations
        self.current_zones = z0.copy()  # rack current zones for each beautificator

    def choose_action(self, current_zone, current_time):
        actions = []
        
        for z2 in range(1, zones + 1):
            if current_zone != z2 and n[z2 - 1] + aco_nOut[z2 - 1] > 1:  #bigger than 1 because to be sure not move for nothing
                move_cost = mMOVE[current_zone - 1][z2 - 1]
                if current_time + move_cost < t_max and float(n[z2 - 1] + aco_nOut[z2 - 1]) / float(move_cost / mBEAU) > 1.0: #ratio is better checking option
                    actions.append(("MOVE", z2, current_time + move_cost))
        
        if n[current_zone - 1] > 0 and current_time + mBEAU < t_max:
            actions.append(("BEAU", current_zone, current_time + mBEAU))
        
        if aco_nOut[current_zone - 1] > 0 and current_time + mHOT < t_max:
            actions.append(("HOT", current_zone, current_time + mHOT))
        
        if current_time + mWAIT < t_max:
            actions.append(("WAIT", current_zone, current_time + mWAIT))

        if not actions:
            return None  

        #Prioritize actions based on profit (minimize MOVE loss)
        probabilities = []
        for action in actions:
            action_type, z2, _ = action
            
            if action_type == "MOVE":
                heuristic = max(0, profits["MOVE"][current_zone - 1][z2 - 1])
            else:
                heuristic = profits[action_type]
                if(action_type == "BEAU"):
                    heuristic /= (mBEAU)
                if(action_type == "HOT"):
                    heuristic /= (mHOT)
                

            pheromone = pheromones[(current_zone, z2, current_time)][action_type]

            probabilities.append((pheromone ** pheromone_importance) * (heuristic ** heuristic_importance))

        probabilities = np.array(probabilities)

        if probabilities.sum() == 0:
            probabilities = np.ones(len(actions)) / len(actions)  #Equal probabilities
        else:
            probabilities /= probabilities.sum()  #Normalize probabilities

        # Choose an action
        chosen_index = np.random.choice(len(actions), p=probabilities)
        

        return actions[chosen_index]

    def traverse(self):
        for b in range(beautificators):
            self.paths[b] = []
            self.total_profits[b] = 0
            self.action_durations[b] = 0 
            self.current_zones = z0.copy()  

        for aco_time in range(t_max):
            for b in range(beautificators):
                if self.action_durations[b] == 0:  #Check if action is not in progress
                    action = self.choose_action(self.current_zones[b], aco_time)
                    if not action:  #Stop if no valid actions
                        continue
                    action_type, next_zone, next_time = action
                    self.paths[b].append((b, self.current_zones[b], aco_time, next_zone, next_time, action_type))
                    self.total_profits[b] += heuristic_profit(action_type, self.current_zones[b], next_zone)
                    # nout - nhot - n
                    #  5      3     8
                #beau  5      -     7
                #beau  5      -     6
                #beau  5            5
                #beau  4            4
                #hot   3            3
                #hot   2            2
                    #Update scooters and state
                    if action_type == "BEAU":
                        n[self.current_zones[b] - 1] -= 1
                        if(n[self.current_zones[b] - 1] < aco_nOut[self.current_zones[b] - 1]):
                            aco_nOut[self.current_zones[b] - 1] = n[self.current_zones[b] - 1]
                        self.action_durations[b] = mBEAU - 1
                    elif action_type == "HOT":
                        aco_nOut[self.current_zones[b] - 1] -= 1
                        n[self.current_zones[b] - 1] -= 1
                        self.action_durations[b] = mHOT - 1
                    else:  #MOVE or WAIT
                        self.action_durations[b] = next_time - aco_time - 1
                    self.current_zones[b] = next_zone
                else:
                    self.action_durations[b] -= 1  #Decrement action duration
     


def ant_colony_optimization():
    global pheromones
    best_profits = [-float("inf")] * beautificators
    best_paths = [[] for _ in range(beautificators)]
    pheromone_increment = 0.02
    move_pheromone_decrement = 0.5

    for iteration in range(num_iterations):
        global n
        global aco_nOut
        global aco_nHot
        
        n = [nOUT[i] + nHOT[i] for i in range(zones)]
        aco_nOut = nOUT.copy()
        aco_nHot = nHOT.copy()

        ants = [Ant() for _ in range(num_ants)]
        for ant in ants:
            ant.traverse()
            n = [nOUT[i] + nHOT[i] for i in range(zones)]
            aco_nOut = nOUT.copy()
            aco_nHot = nHOT.copy()

        
        #Evaporate pheromones
        for key in pheromones:
            for action_type in pheromones[key]:
                pheromones[key][action_type] *= (1 - evaporation_rate)

        #Find the best ant based on the sum of all beautificators' profits
        best_total_profit = sum(best_profits)  # Initialize to a very low value
        best_ant = None


        for ant in ants:
            total_profit = sum(ant.total_profits)  #Sum the profits for all beautificators for this ant
            if total_profit > best_total_profit:
                best_total_profit = total_profit
                best_ant = ant
                #Once the best ant is found, assign the best paths and best profits
                best_paths = best_ant.paths
                best_profits = best_ant.total_profits


        #Tüm karıncaların toplam karlarını hesapla ve sırala
        ant_profits = [sum(ant.total_profits) for ant in ants]
        
        sorted_ants = sorted(zip(ants, ant_profits), key=lambda item: item[1], reverse=True)
        
        #En iyi karıncaya en fazla pheromone ekle, sonraki karıncalara azalan miktarda ekle
        for i, (ant, profit) in enumerate(sorted_ants):
            pheromone_amount = pheromone_increment * (profit/best_total_profit)  #Azalan pheromone miktarı
            #print("pheromoneamount:", pheromone_amount, "ant total profit:", sum(ant.total_profits))
            for b in range(beautificators):
                for _, z1, aco_time, z2, _, action in ant.paths[b]:
                    if action == "MOVE":
                        pheromones[(z1, z2, aco_time)]["MOVE"] += move_pheromone_decrement * pheromone_amount
                        #print("pheromonenow:", pheromones[(z1, z2, time)]["MOVE"])
                    else:
                        pheromones[(z1, z1, aco_time)][action] += pheromone_amount
                        #print("pheromonenow:", pheromones[(z1, z2, time)][action])

        #print(pheromones)
        print(f"Iteration {iteration + 1}, Best Total Profit: {best_total_profit}")

    return best_paths, best_profits, best_total_profit


def heuristic_profit(action_type, z1, z2):
    if action_type == "BEAU":
        return profits["BEAU"]
    elif action_type == "HOT":
        return profits["HOT"]
    elif action_type == "WAIT":
        return profits["WAIT"]
    elif action_type == "MOVE" and z1 != z2:
        return profits["MOVE"][z1 - 1][z2 - 1]  # Negative cost
    return 0



start_time = time.time()
best_paths, best_profits, bestTotalProfit = ant_colony_optimization()
end_time = time.time()

elapsed_time = end_time - start_time

print(f"The process took {elapsed_time} seconds to complete.")

print(f"Best total profit: {bestTotalProfit}")

"""
Total Beautificators: 2
Total Scooters: 239
Total Profit: 635
"""
script_dir = os.path.dirname(os.path.realpath(__file__))
generalOutput = os.path.join(script_dir, "generalOutput.txt")

totalScooters = 0
for i in range(0,16):
    totalScooters += nOUT[i] + nHOT[i]

with open(generalOutput, 'w') as f:
    f.write("-----ACO-----\n")
    f.write(f"Total Beautificators: {beautificators}\n")
    for z in range(len(z0)):
        f.write("Beautificator " + str(z+1) + " Start Zone: " + str(z0[z]) + "\n")
    f.write(f"Total Scooters: {totalScooters}\n")
    f.write(f"Total Profit: {bestTotalProfit}\n")


# Output Results
script_dir = os.path.dirname(os.path.realpath(__file__))  # Get script's directory
output_dir = os.path.join(script_dir, '..', 'csv-outputs')  # Absolute path to output directory

for b in range(beautificators):
    output_file = os.path.join(output_dir, f"output_{b+1}.csv")  # Construct absolute path
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Beautificator", "Zone_From", "Time_From", "Zone_To", "Time_To", "Action"])
        for path in best_paths[b]:
            writer.writerow(path)
    print(f"Best profit for Beautificator {b+1}: {best_profits[b]}") 



for i in range(0,16):
    output_map_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        "maps", 
        f"aco_bf_{i}_route.html"
    )

    # Check if the file exists before attempting to delete it
    if os.path.exists(output_map_file):
        os.remove(output_map_file)
        print(f"File deleted: {output_map_file}")
    else:
        print(f"File not found: {output_map_file}") 


# Load the GeoJSON file containing the zones
script_dir = os.path.dirname(os.path.realpath(__file__))  # Get script's directory
geojson_file = os.path.join(script_dir, 'map-izmit.geojson')  # Construct absolute path
gdf = gpd.read_file(geojson_file)

#List of CSV files, each csv is output for beatuficator
csv_files = []  # Extend this list for all output files

for i in range(beautificators):
    temp_str = "output_" + str(i+1) + ".csv"
    csv_files.append(temp_str)


script_dir = os.path.dirname(os.path.realpath(__file__))  # Get script's directory
csv_outputs_dir = os.path.join(script_dir, '..', 'csv-outputs')  # Path to csv-outputs dir


bf_num = 1

for csv_file in csv_files:
    csv_file_path = os.path.join(csv_outputs_dir, csv_file)
    # 1. Data Preparation
    with open(csv_file_path, 'r') as f:
        reader = csv.DictReader(f)
        route_data = list(reader)

    script_dir = os.path.dirname(os.path.realpath(__file__)) 
    geojson_file_path_temp = os.path.join(script_dir, "map-izmit.geojson")
    with open(geojson_file_path_temp) as f:
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

    # 4. HTML Output
    output_html_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        "maps", 
        f"aco_bf_{bf_num}_route.html"
    )
    m.save(output_html_path)

    bf_num += 1

