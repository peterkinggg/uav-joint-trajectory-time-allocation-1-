"""
UAV joint trajectory and time-allocation simulation.

This script implements the simulation framework used in the thesis:
- baseline method
- time-allocation-only ablation
- trajectory-update-only ablation
- combined iterative alternating-update heuristic
- deterministic multi-scenario tests
- post-simulation movement feasibility check
- figure generation

The main reported metric is normalized collected data:
    sum_i tau_i * R_i

This is rate multiplied by communication time. Average normalized throughput is also
reported as normalized_collected_data / TOTAL_TIME.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class SimulationConfig:
    start: np.ndarray
    end: np.ndarray
    num_positions: int
    altitude: float
    total_time: float
    delta_t: float
    v_max: float
    bandwidth: float
    gamma: float
    step_size: float
    iterations: int
    tau_min: float


CONFIG = SimulationConfig(
    start=np.array([0.0, 0.0]),
    end=np.array([100.0, 100.0]),
    num_positions=20,
    altitude=20.0,
    total_time=10.0,
    delta_t=0.5,
    v_max=30.0,
    bandwidth=1.0,
    gamma=4000.0,
    step_size=0.008,
    iterations=12,
    tau_min=0.5,
)


SCENARIOS: Dict[str, np.ndarray] = {
    "Scenario 1": np.array(
        [[15.0, 20.0], [25.0, 75.0], [60.0, 35.0], [82.0, 70.0]],
        dtype=float,
    ),
    "Scenario 2": np.array(
        [
            [9.60446308, 85.58444071],
            [21.06703948, 80.06020616],
            [37.15911202, 75.36708664],
            [43.73981104, 93.02021161],
        ],
        dtype=float,
    ),
    "Scenario 3": np.array(
        [
            [47.99159936, 21.07982536],
            [76.20105820, 25.93222279],
            [89.46163922, 41.44651477],
            [92.29693742, 63.46435392],
        ],
        dtype=float,
    ),
}


def build_straight_line_trajectory(config: SimulationConfig) -> np.ndarray:
    x = np.linspace(config.start[0], config.end[0], config.num_positions)
    y = np.linspace(config.start[1], config.end[1], config.num_positions)
    return np.column_stack((x, y))


def compute_distances(
    trajectory: np.ndarray, users: np.ndarray, config: SimulationConfig
) -> np.ndarray:
    dx = trajectory[None, :, 0] - users[:, None, 0]
    dy = trajectory[None, :, 1] - users[:, None, 1]
    return np.sqrt(dx**2 + dy**2 + config.altitude**2)


def compute_average_distances(
    trajectory: np.ndarray, users: np.ndarray, config: SimulationConfig
) -> np.ndarray:
    return compute_distances(trajectory, users, config).mean(axis=1)


def compute_rates(avg_distances: np.ndarray, config: SimulationConfig) -> np.ndarray:
    return config.bandwidth * np.log2(1.0 + config.gamma / (avg_distances**2))


def equal_time_allocation(n_users: int, config: SimulationConfig) -> np.ndarray:
    return np.ones(n_users) * (config.total_time / n_users)


def fairness_time_allocation(rates: np.ndarray, config: SimulationConfig) -> np.ndarray:
    n_users = len(rates)
    residual_time = config.total_time - n_users * config.tau_min
    if residual_time < 0:
        raise ValueError("tau_min is too large for the total communication time.")
    rates = np.maximum(rates, 1e-9)
    return np.ones(n_users) * config.tau_min + rates / rates.sum() * residual_time


def compute_normalized_collected_data(
    rates: np.ndarray, time_allocation: np.ndarray
) -> float:
    return float(np.sum(rates * time_allocation))


def compute_average_throughput(
    normalized_collected_data: float, config: SimulationConfig
) -> float:
    return normalized_collected_data / config.total_time


def compute_average_communication_distance(
    trajectory: np.ndarray, users: np.ndarray, config: SimulationConfig
) -> float:
    return float(compute_distances(trajectory, users, config).mean())


def update_trajectory(
    trajectory: np.ndarray,
    users: np.ndarray,
    time_allocation: np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    new_trajectory = trajectory.copy()
    weighted_center = np.average(users, axis=0, weights=time_allocation)
    new_trajectory[1:-1] = (
        new_trajectory[1:-1]
        + config.step_size * (weighted_center - new_trajectory[1:-1])
    )
    new_trajectory[0] = config.start
    new_trajectory[-1] = config.end
    return new_trajectory


def max_segment_length(trajectory: np.ndarray) -> float:
    segment_lengths = np.linalg.norm(trajectory[1:] - trajectory[:-1], axis=1)
    return float(segment_lengths.max())


def check_trajectory_feasibility(
    trajectory: np.ndarray, config: SimulationConfig
) -> Tuple[float, float, bool]:
    max_segment = max_segment_length(trajectory)
    max_allowed = config.v_max * config.delta_t
    feasible = max_segment <= max_allowed + 1e-9
    return max_segment, max_allowed, feasible


def evaluate_trajectory(
    trajectory: np.ndarray,
    users: np.ndarray,
    time_allocation: np.ndarray,
    config: SimulationConfig,
) -> Dict[str, float]:
    avg_distances = compute_average_distances(trajectory, users, config)
    rates = compute_rates(avg_distances, config)
    collected_data = compute_normalized_collected_data(rates, time_allocation)
    avg_distance = compute_average_communication_distance(trajectory, users, config)
    max_segment, max_allowed, feasible = check_trajectory_feasibility(
        trajectory, config
    )
    return {
        "normalized_collected_data": collected_data,
        "average_throughput": compute_average_throughput(collected_data, config),
        "average_distance": avg_distance,
        "max_segment": max_segment,
        "max_allowed": max_allowed,
        "feasible": feasible,
    }


def run_baseline(
    users: np.ndarray, config: SimulationConfig
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    trajectory = build_straight_line_trajectory(config)
    time_allocation = equal_time_allocation(len(users), config)
    results = evaluate_trajectory(trajectory, users, time_allocation, config)
    return trajectory, time_allocation, results


def run_time_allocation_only(
    users: np.ndarray, config: SimulationConfig
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    trajectory = build_straight_line_trajectory(config)
    avg_distances = compute_average_distances(trajectory, users, config)
    rates = compute_rates(avg_distances, config)
    time_allocation = fairness_time_allocation(rates, config)
    results = evaluate_trajectory(trajectory, users, time_allocation, config)
    return trajectory, time_allocation, results


def run_trajectory_update_only(
    users: np.ndarray, config: SimulationConfig
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    trajectory = build_straight_line_trajectory(config)
    time_allocation = equal_time_allocation(len(users), config)
    for _ in range(config.iterations):
        trajectory = update_trajectory(trajectory, users, time_allocation, config)
    results = evaluate_trajectory(trajectory, users, time_allocation, config)
    return trajectory, time_allocation, results


def run_combined_method(
    users: np.ndarray, config: SimulationConfig
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], Dict[str, List[float]]]:
    trajectory = build_straight_line_trajectory(config)
    history = {
        "normalized_collected_data": [],
        "average_throughput": [],
        "average_distance": [],
    }

    for _ in range(config.iterations):
        avg_distances = compute_average_distances(trajectory, users, config)
        rates = compute_rates(avg_distances, config)
        time_allocation = fairness_time_allocation(rates, config)

        trajectory = update_trajectory(trajectory, users, time_allocation, config)
        current_results = evaluate_trajectory(
            trajectory, users, time_allocation, config
        )

        history["normalized_collected_data"].append(
            current_results["normalized_collected_data"]
        )
        history["average_throughput"].append(current_results["average_throughput"])
        history["average_distance"].append(current_results["average_distance"])

    avg_distances = compute_average_distances(trajectory, users, config)
    rates = compute_rates(avg_distances, config)
    time_allocation = fairness_time_allocation(rates, config)
    results = evaluate_trajectory(trajectory, users, time_allocation, config)
    return trajectory, time_allocation, results, history


def percentage_gain(new_value: float, baseline_value: float) -> float:
    return (new_value - baseline_value) / baseline_value * 100.0


def percentage_reduction(new_value: float, baseline_value: float) -> float:
    return (baseline_value - new_value) / baseline_value * 100.0


def print_ablation_table(users: np.ndarray, config: SimulationConfig) -> None:
    baseline_result = run_baseline(users, config)[2]
    methods = {
        "Baseline": baseline_result,
        "Time allocation only": run_time_allocation_only(users, config)[2],
        "Trajectory update only": run_trajectory_update_only(users, config)[2],
        "Both components": run_combined_method(users, config)[2],
    }

    print("\nTable 4.2 / Ablation results for Scenario 1")
    print(
        f"{'Method':<24}{'Norm. collected data':>22}"
        f"{'Avg. distance (m)':>20}{'Data gain':>14}"
        f"{'Distance reduction':>22}"
    )
    for method, result in methods.items():
        data_gain = percentage_gain(
            result["normalized_collected_data"],
            baseline_result["normalized_collected_data"],
        )
        distance_reduction = percentage_reduction(
            result["average_distance"], baseline_result["average_distance"]
        )
        print(
            f"{method:<24}"
            f"{result['normalized_collected_data']:>22.4f}"
            f"{result['average_distance']:>20.4f}"
            f"{data_gain:>13.2f}%"
            f"{distance_reduction:>21.2f}%"
        )


def print_scenario_table(config: SimulationConfig) -> None:
    print("\nTable 4.3 / Additional deterministic test cases")
    print(
        f"{'Scenario and method':<28}{'Norm. collected data':>22}"
        f"{'Avg. distance (m)':>20}{'Data gain':>14}"
        f"{'Distance reduction':>22}"
    )

    for scenario_name, users in SCENARIOS.items():
        baseline_result = run_baseline(users, config)[2]
        combined_result = run_combined_method(users, config)[2]

        for method_name, result in [
            (f"{scenario_name} baseline", baseline_result),
            (f"{scenario_name} both", combined_result),
        ]:
            data_gain = percentage_gain(
                result["normalized_collected_data"],
                baseline_result["normalized_collected_data"],
            )
            distance_reduction = percentage_reduction(
                result["average_distance"], baseline_result["average_distance"]
            )
            print(
                f"{method_name:<28}"
                f"{result['normalized_collected_data']:>22.4f}"
                f"{result['average_distance']:>20.4f}"
                f"{data_gain:>13.2f}%"
                f"{distance_reduction:>21.2f}%"
            )


def print_feasibility_summary(config: SimulationConfig) -> None:
    print("\nPost-simulation movement feasibility check")
    max_allowed = config.v_max * config.delta_t
    print(f"Maximum allowed segment length = V_MAX * DELTA_T = {max_allowed:.4f} m")
    print(f"{'Scenario':<12}{'Method':<24}{'Max segment (m)':>18}{'Feasible':>12}")

    for scenario_name, users in SCENARIOS.items():
        baseline_traj = run_baseline(users, config)[0]
        both_traj = run_combined_method(users, config)[0]

        for method_name, trajectory in [
            ("Baseline", baseline_traj),
            ("Both components", both_traj),
        ]:
            max_segment, _, feasible = check_trajectory_feasibility(
                trajectory, config
            )
            print(
                f"{scenario_name:<12}{method_name:<24}"
                f"{max_segment:>18.4f}{str(feasible):>12}"
            )


def plot_figures_for_scenario_1(config: SimulationConfig) -> None:
    users = SCENARIOS["Scenario 1"]
    baseline_trajectory, _, baseline_results = run_baseline(users, config)
    proposed_trajectory, _, proposed_results, history = run_combined_method(
        users, config
    )

    plt.figure(figsize=(7, 5.5))
    plt.scatter(users[:, 0], users[:, 1], s=80, marker="o", label="Users")
    plt.plot(
        baseline_trajectory[:, 0],
        baseline_trajectory[:, 1],
        linestyle="--",
        marker="o",
        linewidth=2,
        markersize=4,
        label="Baseline trajectory",
    )
    plt.plot(
        proposed_trajectory[:, 0],
        proposed_trajectory[:, 1],
        linestyle="-",
        marker="s",
        linewidth=2,
        markersize=4,
        label="Proposed trajectory",
    )
    plt.scatter(config.start[0], config.start[1], s=80, marker="D", label="Start")
    plt.scatter(config.end[0], config.end[1], s=90, marker="X", label="End")
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.title("UAV Trajectory Comparison")
    plt.grid(True, alpha=0.5)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("trajectory_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    values = [
        baseline_results["normalized_collected_data"],
        proposed_results["normalized_collected_data"],
    ]
    bars = plt.bar(["Baseline", "Proposed"], values)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
        )
    plt.ylabel("Normalized collected data")
    plt.title("Normalized Collected Data Comparison")
    plt.tight_layout()
    plt.savefig("collected_data_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    values = [
        baseline_results["average_distance"],
        proposed_results["average_distance"],
    ]
    bars = plt.bar(["Baseline", "Proposed"], values)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
        )
    plt.ylabel("Average communication distance (m)")
    plt.title("Average Distance Comparison")
    plt.tight_layout()
    plt.savefig("distance_comparison.png", dpi=300)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    iterations = np.arange(1, config.iterations + 1)
    line1, = ax1.plot(
        iterations,
        history["normalized_collected_data"],
        marker="o",
        label="Collected data",
    )
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Normalized collected data")
    ax1.grid(True, alpha=0.5)

    ax2 = ax1.twinx()
    line2, = ax2.plot(
        iterations,
        history["average_distance"],
        marker="s",
        linestyle="--",
        label="Avg. distance",
    )
    ax2.set_ylabel("Average communication distance (m)")

    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best", fontsize=8)
    plt.title("Iteration History")
    fig.tight_layout()
    plt.savefig("iteration_history.png", dpi=300)
    plt.close()


def main() -> None:
    scenario_1_users = SCENARIOS["Scenario 1"]

    print_ablation_table(scenario_1_users, CONFIG)
    print_scenario_table(CONFIG)
    print_feasibility_summary(CONFIG)
    plot_figures_for_scenario_1(CONFIG)

    print("\nFigures generated:")
    print("- trajectory_comparison.png")
    print("- collected_data_comparison.png")
    print("- distance_comparison.png")
    print("- iteration_history.png")


if __name__ == "__main__":
    main()
