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

def read_dat_file(filepath):
    """Reads a .dat file and returns the data."""

    # Get the absolute path to the datasets directory
    script_dir = os.path.dirname(os.path.realpath(__file__))
    datasets_dir = os.path.join(script_dir, '..', 'datasets') 

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
    filepath = "testData.dat"  # Default filename in the datasets directory
    print("No input parameter provided. Using default: testData.dat")
    #sys.exit(1) 

data = read_dat_file(filepath)


zones = data['zones']			
beautificators = data['beautificators']
t_max = data['t_max']	

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
];

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
];


profits = {"BEAU": 5.0, "HOT": 10.0, "WAIT": 0.0, "MOVE": np.array(piMOVE)}  #be sure movement is negative


num_ants = 100
num_iterations = 50
evaporation_rate = 0.02
pheromone_importance = 1.0
heuristic_importance = 2.0

#Pheromone Initialization with HashMap
pheromones = {}
for z1 in range(1, zones + 1):
    for z2 in range(1, zones + 1):
        for aco_time in range(0, t_max):
            pheromones[(z1, z2, aco_time)] = {"MOVE": 1.0, "BEAU": 1.0, "HOT": 1.0, "WAIT": 1.0}


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

                    #Update scooters and state
                    if action_type == "BEAU":
                        n[self.current_zones[b] - 1] -= 1
                        self.action_durations[b] = mBEAU - 1
                    elif action_type == "HOT":
                        aco_nOut[self.current_zones[b] - 1] -= 1
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
    pheromone_increment = 0.1  
    move_pheromone_decrement = 0.05 

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


# Load the GeoJSON file containing the zones
script_dir = os.path.dirname(os.path.realpath(__file__))  # Get script's directory
geojson_file = os.path.join(script_dir, 'map-izmit.geojson')  # Construct absolute path
gdf = gpd.read_file(geojson_file)

#List of CSV files, each csv is output for beatuficator
csv_files = []  # Extend this list for all output files

for i in range(beautificators):
    temp_str = "output_" + str(i+1) + ".csv"
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
csv_outputs_dir = os.path.join(script_dir, '..', 'csv-outputs')  # Path to csv-outputs dir

for csv_file in csv_files:
    
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
    plt.title(f"Route of a Beautificator ({csv_file})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    #Save the plot with a dynamic filename based on the CSV file
    output_image_name = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        "maps", 
        f"aco_{csv_file.replace('.csv', '')}_route.png"
    )

    # Save the plot 
    plt.savefig(output_image_name)
    plt.close()  
