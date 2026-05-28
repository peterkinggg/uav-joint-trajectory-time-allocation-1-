[README.md](https://github.com/user-attachments/files/28333616/README.md)
# UAV Joint Trajectory and Time Allocation Simulation

This repository contains the Python simulation code used in the thesis:

**Trajectory Optimization and Time Allocation in a Single-UAV-Assisted IoT System**

## What this code reproduces

The script reproduces the numerical experiments reported in Chapter 4:

- baseline method
- time-allocation-only ablation
- trajectory-update-only ablation
- combined proposed method
- Scenario 1, Scenario 2, and Scenario 3 deterministic test cases
- trajectory, collected-data, distance, and iteration-history figures

## Important metric note

The reported value is named **normalized collected data** in the revised code because it is calculated as:

```text
sum_i tau_i * R_i
```

This is rate multiplied by time, so it represents a normalized amount of collected data.  
Average normalized throughput can be obtained by dividing this value by the total communication time.

## Requirements

- Python 3
- numpy
- matplotlib

Install dependencies:

```bash
pip install numpy matplotlib
```

## How to run

```bash
python uav_revised_simulation.py
```

The script prints the ablation table, the multi-scenario table, a movement feasibility check, and generates the figures.
