# CS175
CS 175 Project Proposal and Milestone 1

Members: Dongheng Li, Tiffany Ling, Jiahang Zhang

Appointment with the Instructor:
Thursday, January 30th, 1:30pm (Proposal)
Tuesday, February 11th, 1:15pm (Milestone 1)

GitHub repo: https://github.com/trling/CS175
Website: https://trling.github.io/CS175/

# Summary of the Project (Option 1)
The goal of our project is to get the agent to find the coin in a world with an uneven surface surrounded by cliffs. The agent will try to reach that coin in the lowest amount of time while losing as few health points as possible. The agent loses health points when falling from higher to lower ground and getting hit by monsters. If the agent falls off the cliff or runs out of health points, the run is unsuccessful and the agent returns to the starting point.

The inputs for our project are the environment, the location of the coin, the starting location of the agent, and the initial health points.  The outputs of our project are the series of actions the agent takes to get to the coin and the remaining health points.

A potential application of this project in the real world is to simulate the task of finding the best path to reach some goal object/location. One could model some place that exists in the real world in Minecraft. One could then run our AI on that world and find the best safe path from one point to another.

# Evaluation Plan
Quantitative evaluation

The metrics will be the time spent for the agent to reach the coin (try to minimize) and the agent's remaining health points when it reaches the coin (try to maximize). The baselines will be the time spent and remaining health points after the first few runs (maybe the average of runs 1-10). We do not know exactly by how much our approach will improve the metrics, but we hope that we can decrease the time needed to reach the coin to half and/or double the health points remaining. The data we will evaluate on will be the time/health points after the first few runs compared to that of an optimal run (we do not know if our approach will find the true optimal path, but we hope that it will come close. We will say the run is "optimal" and stop our program when the change in performance between runs is close to zero).

Qualitative evaluation

We can create sanity cases to verify our project works by first having a human manually control the agent to reach the coin. We keep track of the time needed. Most likely, this will not be the true optimal path because of the person's reaction time, time needed to think, delay in clicks, etc. However, this will provide a good benchmark of the time needed to reach the coin. If our agent is able to reach the coin in the same or lower amount of time, we can conclude that our agent is intelligent in the sense that it can replicate human intelligence to some degree. We can visualize the internals of our algorithm by watching our agent try to reach the coin over several runs. The first part of our algorithm will be to locate the coin. We can verify if this part of our algorithm works by seeing if the agent will rotate/turn until the coin is within sight. The second part of our algorithm will be to move towards the coin. We can check if this works by seeing if the agent moves in the general direction of the coin (may not be a straight line towards the coin if there are obstacles in the way). Another part of the algorithm will be to locate obstacles (e.g. trees, rocks, water, monsters) and avoid/walk around them. We can make sure our algorithm is working properly if the agent does not run directly into them (waste time/health points). Our moonshot case would be to find the true optimal path to reach the coin for any given world (which can be complicated because of getting stuck in local maxima/minima).

# Goals
1. Minimum goal
   - Milestone 1: Generate a flat world with a coin randomly placed somewhere. The agent needs to find and reach the coin.
   - Milestone 2: Generate cliffs surrounding the flat ground. The agent needs to find and reach the coin without falling off the cliff (kills the agent).
2. Realistic goal
   - Milestone 1: Make the ground uneven (i.e. not completely flat and level). The agent now needs to jump and can take fall damage from falling off hills. If the agent runs out of health points, the run is unsuccessful.
   - Milestone 2: Add trees and other objects to the world. This will make it more difficult for the agent to find a path to the coin.
3. Ambitious goal 
   - Add monsters that can attack the agent (depletes the agent's health points). If the agent runs out of health points, the run is unsuccessful. The agent needs to avoid monsters and/or kill them.

# How to run Milestone 1
1. clone or download repository from https://github.com/trling/CS175
2. open command prompt and launch Minecraft Malmo server
3. open another command prompt and go to root folder of repository
4. run python milestone1.py
5. let program run (it should show the Minecraft window with an agent trying to find a path to the golden apple and a Q-table where the green dots show the best path found so far to the golden apple). Reward results for each run will be saved in rewards.txt file


# How to run Final
1. download the cliff_walking.py and cliff_walking.xml files from https://github.com/trling/CS175
2. put the cliff_walking.xml file in the Sample_missions directory of your installation of Malmo
3. put the cliff_walking.py file in the Python_Examples directory of your installation of Malmo
4. open a terminal and run the Minecraft server
5. open a terminal, navigate to the directory of the cliff_walking.py file, and run "python3 cliff_walking.py" with flags for values of alpha, gamma, and epsilon or use the defaults. (Example: "python3 cliff_walking.py --alpha=0.1 --gamma=1.0 --epsilon=0.01")
6. let the program run (the resulting rewards for that run will be written in a file)
