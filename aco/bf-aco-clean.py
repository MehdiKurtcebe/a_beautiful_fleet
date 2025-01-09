import random
import numpy as np
import csv
import time

zones = 16			
beautificators = 5
t_max = 72		

z0 = [3, 5, 8, 11, 15];	


nOUT = [ 5,  9,  9,  6,  8, 11,  7, 13, 17, 19, 11,  8,  9,  8, 11,  6];
nHOT = [ 1,  2,  5,  1,  3,  5,  2,  7,  5, 11,  7,  3,  5,  3,  6,  2];


mBEAU = 1
mHOT = 3
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
mWAIT = 1


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

#Profits
profits = {"BEAU": 5.0, "HOT": 10.0, "WAIT": 0.0, "MOVE": np.array(piMOVE)}  #be sure movement is negative


num_ants = 50
num_iterations = 100
evaporation_rate = 0.02
pheromone_importance = 1.0
heuristic_importance = 2.0

#Pheromone Initialization with HashMap
pheromones = {}
for z1 in range(1, zones + 1):
    for z2 in range(1, zones + 1):
        for aco_time in range(0, t_max):
            pheromones[(z1, z2, aco_time)] = {"MOVE": 1.0, "BEAU": 1.0, "HOT": 1.0, "WAIT": 1.0}

# Ant Class
class Ant:
    def __init__(self):
        self.paths = [[] for _ in range(beautificators)]  
        self.total_profits = [0] * beautificators
        self.action_durations = [0] * beautificators  
        self.current_zones = z0.copy()  

    def choose_action(self, current_zone, current_time):
        actions = []
        #MOVE
        for z2 in range(1, zones + 1):
            if current_zone != z2 and n[z2 - 1] + aco_nOut[z2 - 1] > 1:  #bigger than 1 because to be sure not move for nothing
                move_cost = mMOVE[current_zone - 1][z2 - 1]
                if current_time + move_cost < t_max and float(n[z2 - 1] + aco_nOut[z2 - 1]) / float(move_cost / mBEAU) > 1.0: #ratio is better checking option
                    actions.append(("MOVE", z2, current_time + move_cost))
        #BEAU
        if n[current_zone - 1] > 0 and current_time + mBEAU < t_max:
            actions.append(("BEAU", current_zone, current_time + mBEAU))
        #HOT
        if aco_nOut[current_zone - 1] > 0 and current_time + mHOT < t_max:
            actions.append(("HOT", current_zone, current_time + mHOT))
        #WAIT
        if current_time + mWAIT < t_max:
            actions.append(("WAIT", current_zone, current_time + mWAIT))

        if not actions:
            return None 

        #Prioritize actions based on profit(minimize MOVE loss)
        probabilities = []
        for action in actions:
            action_type, z2, _ = action
            #Calculate heuristic based on action type
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

        chosen_index = np.random.choice(len(actions), p=probabilities)
        

        return actions[chosen_index]

    def traverse(self):
        for b in range(beautificators):
            self.paths[b] = []
            self.total_profits[b] = 0
            self.action_durations[b] = 0  #Reset action duration
            self.current_zones = z0.copy()  #Reset current zones

        for aco_time in range(t_max):
            for b in range(beautificators):
                if self.action_durations[b] == 0:  # Check if action is not in progress
                    action = self.choose_action(self.current_zones[b], aco_time)
                    if not action:  
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
                    self.action_durations[b] -= 1  # Decrement action duration
     

#ACO Loop
def ant_colony_optimization():
    global pheromones
    best_profits = [-float("inf")] * beautificators
    best_paths = [[] for _ in range(beautificators)]
    pheromone_increment = 0.1  #Amount of pheromone for best paths
    move_pheromone_decrement = 0.05  #Reduced pheromone for MOVE actions

    for iteration in range(num_iterations):
        #Reset scooters for each iteration
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

        #Evaporation
        for key in pheromones:
            for action_type in pheromones[key]:
                pheromones[key][action_type] *= (1 - evaporation_rate)

        #Find the best ant based on the sum of all beautificators' profits
        best_total_profit = sum(best_profits)
        best_ant = None

        for ant in ants:
            total_profit = sum(ant.total_profits)  # Sum the profits for all beautificators for this ant
            if total_profit > best_total_profit:
                best_total_profit = total_profit
                best_ant = ant
                #Once the best ant is found, assign the best paths and best profits
                best_paths = best_ant.paths
                best_profits = best_ant.total_profits


        # Tüm karıncaların toplam karlarını hesapla ve sırala
        ant_profits = [sum(ant.total_profits) for ant in ants]
        
        sorted_ants = sorted(zip(ants, ant_profits), key=lambda item: item[1], reverse=True)
        
        # En iyi karıncaya en fazla pheromone ekle, sonraki karıncalara azalan miktarda ekle
        for i, (ant, profit) in enumerate(sorted_ants):
            pheromone_amount = pheromone_increment * (profit/best_total_profit)  # Azalan pheromone miktarı
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

# Helper Function
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

#Output
#Per Beatuficator
for b in range(beautificators):
    output_file = f"output_{b+1}.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Beautificator", "Zone_From", "Time_From", "Zone_To", "Time_To", "Action"])
        for path in best_paths[b]:
            writer.writerow(path)
    print(f"Best profit for Beautificator {b+1}: {best_profits[b]}")