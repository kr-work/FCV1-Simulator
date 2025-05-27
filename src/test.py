import numpy as np
from build.simulator import StoneSimulator
import json

stone_simulator = StoneSimulator()

with open("data.json", "r") as read_file:
    data = json.load(read_file)

position: list = data["position"]
np_position = np.array(position)
shot = data["shot"]
shot_per_team = data["shot_per_team"]
x_velocity = data["x_velocities"]
y_velocity = data["y_velocities"]
angular_velocity = data["angular_velocities"]
team_id = data["team_id"]
applied_rule = data["applied_rule"]

np_x_velocity = np.float32(-0.088)
np_y_velocity = np.float32(2.438)
np_angular_velocity = int(angular_velocity)

result, trajectory = stone_simulator.simulator(np_position, shot, np_x_velocity, np_y_velocity, np_angular_velocity, team_id, shot_per_team, applied_rule)

# print(result)
print(result)
# print(trajectory)
