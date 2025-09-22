import numpy as np
from build.simulator import StoneSimulator

from pprint import pprint
import json

stone_simulator = StoneSimulator()

with open("data.json", "r") as read_file:
    data = json.load(read_file)

team0_position: list = data["team0_positions"]
team1_position: list = data["team1_positions"]
np_team0_position = np.array(team0_position)
np_team1_position = np.array(team1_position)
shot = data["shot"]
shot_per_team = data["shot_per_team"]
team_id = data["team_id"]
x_velocities: float = data["x_velocities"]
y_velocities: float = data["y_velocities"]
angular_velocities: int = data["angular_velocities"]

simulated_stones_position, trajectory = stone_simulator.simulator(np_team0_position, np_team1_position, shot, x_velocities, y_velocities, angular_velocities, team_id, shot_per_team, 1)

# print(result)
# print(flag)
for i in range(len(simulated_stones_position)):
    pprint(simulated_stones_position[i])

# 1.9991474151611328 39.983848571777344