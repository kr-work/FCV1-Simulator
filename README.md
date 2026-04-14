# FCV1-Simulator

FCV1-Simulator is a curling stone physics simulator exposed as a Python module (`simulator`) built with pybind11 and Box2D.

This README is written for users who consume GitHub Release artifacts and want to run simulations from Python.

## Overview

- Input: current stone positions, shot context, throw velocity, spin direction, and rule mode.
- Engine: Box2D-based rigid body simulation with curling-specific motion model.
- Output:
  - final stone positions as a NumPy array
  - trajectory snapshots as a Python list (sampled during simulation)

## Supported Environment

- OS:
  - Linux (`.so` artifact)
  - Windows (`.pyd` artifact)
- Python: 3.9 to 3.12 (matching the CI/release matrix)
- Python packages:
  - `numpy < 2.0`

Install Python dependencies:

```bash
pip install "numpy<2.0"
```

## Quick Start (Using Release Artifact)

1. Download the artifact that matches your OS and Python version from Releases.
2. Create a `build` directory in your project.
3. Place the artifact in `build/` and rename it:
	- Linux: `simulator.so`
	- Windows: `simulator.pyd`
4. Run Python and import `StoneSimulator`.

Example:

```python
import numpy as np
from build.simulator import StoneSimulator

sim = StoneSimulator()

# 12 stones (mixed doubles format): shape (12, 2)
stone_positions = np.array([
	 [0.0, 34.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
	 [0.0, 0.0],  [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
], dtype=np.float64)

result, trajectory = sim.simulator(
	 stone_positions=stone_positions,
	 shot=4,
	 x_velocity=-0.09,
	 y_velocity=2.32,
	 angular_sign=1,
	 team_id=1,
	 shot_per_team=2,
	 applied_rule=2,
)

print(result.shape)   # (2, 6, 2) for 12-stone input, (2, 8, 2) for 16-stone input
print(len(trajectory))
```

## Build From Source

If you do not use release artifacts, you can build locally.

### 1) Prepare dependencies

```bash
pip install pybind11[global]
```

### 2) Prepare submodules (pinned commits used by CI)

```bash
git submodule update --init --recursive
cd extern/box2d
git checkout 9ebbbcd960ad424e03e5de6e66a40764c16f51bc
cd ../json
git checkout 11a835df85677002a8aadc5b4e945684c5b7f68b
cd ../..
```

### 3) Build Box2D

```bash
cd extern/box2d
mkdir -p build
cd build
cmake -DBOX2D_BUILD_DOCS=OFF -DBOX2D_BUILD_UNIT_TESTS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX="./" ..
cmake --build . --config Release
cmake --build . --target install --config Release
cd ../../..
```

### 4) Build nlohmann/json

```bash
cd extern/json
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX="./" ..
cmake --build .
cmake --build . --target install
cd ../../..
```

### 5) Build simulator module

```bash
cd src
mkdir -p build
cd build
cmake ..
cmake --build . --config Release
```

Build output:

- Linux: `src/build/simulator.so`
- Windows: `src/build/simulator.pyd`

## Python API Reference

### Class

- `StoneSimulator`

### Method

```python
simulator(
	 stone_positions,
	 shot,
	 x_velocity,
	 y_velocity,
	 angular_sign,
	 team_id,
	 shot_per_team,
	 applied_rule,
) -> (result, trajectory)
```

### Arguments

1. `stone_positions` (`numpy.ndarray`)
	- Accepted shapes:
	  - `1D`: length `32` (16 stones) or `24` (12 stones)
	  - `2D`: shape `(16, 2)` or `(12, 2)`
	- Team order:
	  - 16-stone mode: team0 `[0..7]`, team1 `[0..7]`
	  - 12-stone mode: team0 `[0..5]`, team1 `[0..5]`

2. `shot` (`int`)
	- Total shot count in the current end.

3. `x_velocity` (`float`)
	- Initial x velocity for the thrown stone.

4. `y_velocity` (`float`)
	- Initial y velocity for the thrown stone.

5. `angular_sign` (`int`)
	- Spin direction sign:
	  - `1`: clockwise
	  - `-1`: counterclockwise

6. `team_id` (`int`)
	- Throwing team id (`0` or `1`).

7. `shot_per_team` (`int`)
	- Shot index within a team.

8. `applied_rule` (`int`)
	- Rule mode:
	  - `0`: five rock rule
	  - `1`: no tick rule
	  - `2`: modified FGZ rule

### Returns

1. `result` (`numpy.ndarray`, 3D)
	- Shape:
	  - `(2, 8, 2)` for 16-stone input
	  - `(2, 6, 2)` for 12-stone input
	- Last axis stores `(x, y)`.

2. `trajectory` (`list`)
	- List of time steps.
	- Each step is a list of tuples: `(stone_id, x, y)`.
	- Sampling interval is every 100 simulation frames.
	- Simulation step is `0.001` sec, so trajectory sampling is approximately every `0.1` sec.

## Rule Modes

- `applied_rule = 0` (five rock rule)
  - During early shots (`shot < 5`), free-guard-zone protection logic is applied.

- `applied_rule = 1` (no tick rule)
  - During early shots (`shot < 5`), center-line no-tick checks are applied.

- `applied_rule = 2` (modified FGZ)
  - During the first 3 shots (`shot < 3`), existing in-play stones are protected.
  - Internal throw index handling differs in this mode to account for pre-placed stones.

## Input/Output Notes

- Internal simulation always uses 16 stone slots; 12-stone input is mapped internally.
- Out-of-play stones are represented as `(0.0, 0.0)`.
- Play area constraints are enforced in post-processing.

## Physical/Sheet Constants (Implemented)

- Stone radius: `0.145 m`
- House radius: `1.829 m`
- Tee line: `38.405 m`
- Play area x-range: `[-2.375, 2.375]`
- Play area y-range used in checks: approximately `[30.0, 40.234]`

## Troubleshooting

1. `ModuleNotFoundError: No module named 'build.simulator'`
	- Ensure artifact is in `build/` and named exactly `simulator.so` (Linux) or `simulator.pyd` (Windows).

2. NumPy-related import/runtime issues
	- Use `numpy < 2.0`.

3. Build cannot find Box2D or nlohmann_json
	- Confirm both dependencies were built under `extern/box2d/build` and `extern/json/build`.

4. Unexpected behavior between local build and release
	- Use the same pinned submodule commits as CI.

## Additional Files

- Example script: `src/test.py`
- Example simulator input: `src/data.json`
- Basic config example: `src/config.json`

