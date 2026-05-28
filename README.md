[Uploading README.md…]()
# UAV Joint Trajectory and Time Allocation Simulation

This repository contains the Python simulation code used in the thesis:

**Trajectory Optimization and Time Allocation in a Single-UAV-Assisted IoT System**

## Description

This code implements a simplified simulation framework for a single-UAV-assisted IoT data collection system. It compares a straight-line and equal-time baseline with an iterative alternating-update heuristic for trajectory adjustment and user time allocation.

The code is designed to reproduce the experiments reported in Chapter 4 of the thesis.

## What the Code Includes

The script includes:

- baseline method
- time-allocation-only ablation
- trajectory-update-only ablation
- combined proposed method
- deterministic test cases for Scenario 1, Scenario 2, and Scenario 3
- post-simulation movement feasibility check
- numerical result calculation
- figure generation with matching figure names and labels

## Important Metric Note

The main reported metric is **normalized collected data**.

It is calculated as:

```text
sum_i tau_i * R_i
```

This value is rate multiplied by communication time. Therefore, it represents a normalized amount of data collected over the mission, not pure instantaneous throughput.

If average normalized throughput is required, it can be calculated as:

```text
normalized_collected_data / TOTAL_TIME
```

## Requirements

- Python 3
- numpy
- matplotlib

Install dependencies with:

```bash
pip install numpy matplotlib
```

## How to Run

Run the simulation with:

```bash
python uav_revised_simulation.py
```

## Output

The script prints:

- ablation results for Scenario 1
- additional deterministic test-case results for Scenario 1, Scenario 2, and Scenario 3
- post-simulation movement feasibility check
- generated figure filenames

The script also generates figures used in the thesis, including:

- UAV trajectory comparison
- normalized collected data comparison
- average communication distance comparison
- iteration history of the proposed method

## Thesis Connection

This code supports the experiments, tables, and figures reported in Chapter 4 of the thesis. The code is intended to make the thesis results reproducible, including the baseline method, ablation study, combined proposed method, and multi-scenario experiments.

## Notes

The simulation uses deterministic user coordinates and predefined parameters. It does not use random user generation, external datasets, or machine learning training. The proposed method is a transparent heuristic rather than a globally optimal trajectory optimization solver.
