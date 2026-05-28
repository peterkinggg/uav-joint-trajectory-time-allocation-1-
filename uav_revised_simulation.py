"""
Complete UAV joint trajectory and time-allocation simulation.

This version addresses the supervisor's code comments:
1. V_MAX and DELTA_T are used in a trajectory feasibility check.
2. The reported metric is named "normalized collected data" because it is rate * time.
   Average throughput is also reported as normalized_collected_data / TOTAL_TIME.
3. User coordinates are passed into functions instead of being fixed as global USERS.
4. Iteration history is recorded after each trajectory update for consistency.

The script reproduces:
- baseline result
- time-allocation-only ablation
- trajectory-update-only ablation
- combined method
- Scenario 1, Scenario 2, Scenario 3 deterministic test cases
- figures used in the thesis
"""

from dataclasses import dataclass
from typing import Dict, Tuple

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
    step_size=0.01,
    iterations=12,
    tau_min=0.5,
)

# These deterministic layouts correspond to the results reported in Chapter 4.
SCENARIOS: Dict[str, np.ndarray] = {
    "Scenario 1": np.array(
        [[15.0, 20.0], [25.0, 75.0], [60.0, 35.0], [82.0, 70.0]], dtype=float
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
    """Create a straight-line trajectory with config.num_positions sampled positions."""
    x = np.linspace(config.start[0], config.end[0], config.num_positions)
    y = np.linspace(config.start[1], config.end[1], config.num_positions)
    return np.column_stack((x, y))


def compute_distances(
    trajectory: np.ndarray, users: np.ndarray, config: SimulationConfig
) -> np.ndarray:
    """Compute 3D UAV-user distances for every user and sampled trajectory position."""
    dx = trajectory[None, :, 0] - users[:, None, 0]
    dy = trajectory[None, :, 1] - users[:, None, 1]
    return np.sqrt(dx**2 + dy**2 + config.altitude**2)


def compute_average_distances(
    trajectory: np.ndarray, users: np.ndarray, config: SimulationConfig
) -> np.ndarray:
    """Compute each user's average UAV-user distance over the sampled trajectory."""
    return compute_distances(trajectory, users, config).mean(axis=1)


def compute_rates(avg_distances: np.ndarray, config: SimulationConfig) -> np.ndarray:
    """Compute simplified normalized rates from average distances."""
    return config.bandwidth * np.log2(1.0 + config.gamma / (avg_distances**2))


def fairness_time_allocation(rates: np.ndarray, config: SimulationConfig) -> np.ndarray:
    """Allocate time with a minimum service requirement for every user."""
    n_users = len(rates)
    residual_time = config.total_time - n_users * config.tau_min
    if residual_time < 0:
        raise ValueError("tau_min is too large for the total communication time.")
    rates = np.maximum(rates, 1e-9)
    return np.ones(n_users) * config.tau_min + rates / rates.sum() * residual_time


def equal_time_allocation(n_users: int, config: SimulationConfig) -> np.ndarray:
    """Allocate the same communication time to every user."""
    return np.ones(n_users) * (config.total_time / n_users)


def compute_normalized_collected_data(
    rates: np.ndarray, time_allocation: np.ndarray
) -> float:
    """
    Compute sum_i tau_i * R_i.

    This metric is rate multiplied by time, so it is a normalized amount of collected
    data, not pure throughput. Average normalized throughput can be obtained by
    dividing this value by config.total_time.
    """
    return float(np.sum(rates * time_allocation))


def compute_average_throughput(
    normalized_collected_data: float, config: SimulationConfig
) -> float:
    """Compute average normalized throughput over the total communication time."""
    return normalized_collected_data / config.total_time


def compute_average_communication_distance(
    trajectory: np.ndarray, users: np.ndarray, config: SimulationConfig
) -> float:
    """Compute average UAV-user distance across all users and sampled positions."""
    return float(compute_distances(trajectory, users, config).mean())


def update_trajectory(
    trajectory: np.ndarray,
    users: np.ndarray,
    time_allocation: np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    """
    Move interior trajectory positions slightly toward the weighted user center.

    This is a geometric path adjustment, not a global trajectory optimization solver.
    """
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
    """Return the maximum distance between adjacent sampled trajectory positions."""
    segment_lengths = np.linalg.norm(trajectory[1:] - trajectory[:-1], axis=1)
    return float(segment_lengths.max())


def check_trajectory_feasibility(
    trajectory: np.ndarray, config: SimulationConfig
) -> Tuple[float, float, bool]:
    """
    Check the simplified movement constraint.

    With T sampled positions, there are T-1 movement intervals.
    The maximum allowed interval displacement is V_MAX * DELTA_T.
    """
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
    """Evaluate one trajectory and one time-allocation vector."""
    avg_distances = compute_average_distances(trajectory, users, config)
    rates = compute_rates(avg_distances, config)
    collected_data = compute_normalized_collected_data(rates, time_allocation)
    avg_distance = compute_average_communication_distance(trajectory, users, config)
    max_segment, max_allowed, feasible = check_trajectory_feasibility(trajectory, config)
    return {
        "normalized_collected_data": collected_data,
        "average_throughput": compute_average_throughput(collected_data, config),
        "average_distance": avg_distance,
        "max_segment": max_segment,
        "max_allowed": max_allowed,
        "feasible": feasible,
    }


def run_baseline(users: np.ndarray, config: SimulationConfig) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Straight-line trajectory and equal time allocation."""
    trajectory = build_straight_line_trajectory(config)
    time_allocation = equal_time_allocation(len(users), config)
    results = evaluate_trajectory(trajectory, users, time_allocation, config)
    return trajectory, time_allocation, results


def run_time_allocation_only(
    users: np.ndarray, config: SimulationConfig
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Straight-line trajectory with the fairness-aware time allocation rule."""
    trajectory = build_straight_line_trajectory(config)
    avg_distances = compute_average_distances(trajectory, users, config)
    rates = compute_rates(avg_distances, config)
    time_allocation = fairness_time_allocation(rates, config)
    results = evaluate_trajectory(trajectory, users, time_allocation, config)
    return trajectory, time_allocation, results


def run_trajectory_update_only(
    users: np.ndarray, config: SimulationConfig
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Geometric trajectory adjustment with equal time allocation."""
    trajectory = build_straight_line_trajectory(config)
    time_allocation = equal_time_allocation(len(users), config)

    for _ in range(config.iterations):
        trajectory = update_trajectory(trajectory, users, time_allocation, config)

    results = evaluate_trajectory(trajectory, users, time_allocation, config)
    return trajectory, time_allocation, results


def run_combined_method(
    users: np.ndarray, config: SimulationConfig
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], Dict[str, list]]:
    """Alternating heuristic: update time allocation, then update trajectory."""
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

        # Record the metric after the trajectory update to match the explanation of the figure.
        trajectory = update_trajectory(trajectory, users, time_allocation, config)
        current_results = evaluate_trajectory(trajectory, users, time_allocation, config)

        history["normalized_collected_data"].append(current_results["normalized_collected_data"])
        history["average_throughput"].append(current_results["average_throughput"])
        history["average_distance"].append(current_results["average_distance"])

    avg_distances = compute_average_distances(trajectory, users, config)
    rates = compute_rates(avg_distances, config)
    time_allocation = fairness_time_allocation(rates, config)
    results = evaluate_trajectory(trajectory, users, time_allocation, config)
    return trajectory, time_allocation, results, history


def percentage_gain(new_value: float, baseline_value: float) -> float:
    """Percentage increase compared with a baseline value."""
    return (new_value - baseline_value) / baseline_value * 100.0


def percentage_reduction(new_value: float, baseline_value: float) -> float:
    """Percentage reduction compared with a baseline value."""
    return (baseline_value - new_value) / baseline_value * 100.0


def print_ablation_table(users: np.ndarray, config: SimulationConfig) -> None:
    """Print the ablation results for Scenario 1."""
    methods = {
        "Baseline": run_baseline(users, config)[2],
        "Time allocation only": run_time_allocation_only(users, config)[2],
        "Trajectory update only": run_trajectory_update_only(users, config)[2],
        "Both components": run_combined_method(users, config)[2],
    }

    baseline = methods["Baseline"]
    print("\nTable 4.2 / Ablation results for Scenario 1")
    print(
        f"{'Method':<24}{'Norm. collected data':>22}{'Avg. distance (m)':>20}"
        f"{'Data gain':>14}{'Distance reduction':>22}"
    )
    for method, result in methods.items():
        data_gain = percentage_gain(
            result["normalized_collected_data"],
            baseline["normalized_collected_data"],
        )
        distance_reduction = percentage_reduction(
            result["average_distance"], baseline["average_distance"]
        )
        print(
            f"{method:<24}"
            f"{result['normalized_collected_data']:>22.4f}"
            f"{result['average_distance']:>20.4f}"
            f"{data_gain:>13.2f}%"
            f"{distance_reduction:>21.2f}%"
        )


def print_scenario_table(config: SimulationConfig) -> None:
    """Print baseline and combined-method results for all deterministic scenarios."""
    print("\nTable 4.3 / Additional deterministic test cases")
    print(
        f"{'Scenario and method':<28}{'Norm. collected data':>22}{'Avg. distance (m)':>20}"
        f"{'Data gain':>14}{'Distance reduction':>22}"
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
    """Print the movement feasibility check for each scenario and method."""
    print("\nPost-simulation movement feasibility check")
    print(f"Maximum allowed segment length = V_MAX * DELTA_T = {config.v_max * config.delta_t:.4f} m")
    print(f"{'Scenario':<12}{'Method':<24}{'Max segment (m)':>18}{'Feasible':>12}")

    for scenario_name, users in SCENARIOS.items():
        baseline_traj = run_baseline(users, config)[0]
        both_traj = run_combined_method(users, config)[0]

        for method_name, trajectory in [
            ("Baseline", baseline_traj),
            ("Both components", both_traj),
        ]:
            max_segment, _, feasible = check_trajectory_feasibility(trajectory, config)
            print(
                f"{scenario_name:<12}{method_name:<24}{max_segment:>18.4f}{str(feasible):>12}"
            )


def plot_figures_for_scenario_1(config: SimulationConfig) -> None:
    """Generate the four figures used for Scenario 1 in the thesis."""
    users = SCENARIOS["Scenario 1"]
    baseline_trajectory, _, baseline_results = run_baseline(users, config)
    proposed_trajectory, _, proposed_results, history = run_combined_method(users, config)

    plt.figure(figsize=(7, 6))
    plt.scatter(users[:, 0], users[:, 1], s=90, marker="o", label="Ground users", zorder=5)
    plt.plot(
        baseline_trajectory[:, 0],
        baseline_trajectory[:, 1],
        linestyle="--",
        marker="o",
        linewidth=2.4,
        markersize=5,
        label="Baseline trajectory",
    )
    plt.plot(
        proposed_trajectory[:, 0],
        proposed_trajectory[:, 1],
        linestyle="-",
        marker="s",
        linewidth=3.2,
        markersize=5,
        label="Proposed trajectory",
    )
    plt.scatter(config.start[0], config.start[1], s=110, marker="D", label="Start", zorder=6)
    plt.scatter(config.end[0], config.end[1], s=130, marker="X", label="End", zorder=6)
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.title("UAV Trajectory Comparison")
    plt.grid(True, alpha=0.6)
    plt.legend(loc="lower right", fontsize=9)
    plt.xlim(-5, 105)
    plt.ylim(-5, 105)
    plt.tight_layout()
    plt.savefig("trajectory_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    bars = plt.bar(
        ["Baseline", "Proposed"],
        [
            baseline_results["normalized_collected_data"],
            proposed_results["normalized_collected_data"],
        ],
    )
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
    plt.title("Collected Data Comparison")
    plt.tight_layout()
    plt.savefig("collected_data_comparison.png", dpi=300)
    plt.close()

    # Keep the old filename too, in case the LaTeX file still uses it.
    plt.figure(figsize=(6, 4))
    bars = plt.bar(
        ["Baseline", "Proposed"],
        [
            baseline_results["normalized_collected_data"],
            proposed_results["normalized_collected_data"],
        ],
    )
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
    plt.title("Collected Data Comparison")
    plt.tight_layout()
    plt.savefig("throughput_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    bars = plt.bar(
        ["Baseline", "Proposed"],
        [baseline_results["average_distance"], proposed_results["average_distance"]],
    )
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

    plt.figure(figsize=(7, 4))
    iterations = np.arange(1, config.iterations + 1)
    plt.plot(iterations, history["normalized_collected_data"], marker="o", label="Collected data")
    plt.plot(iterations, history["average_distance"], marker="s", label="Avg. distance")
    plt.xlabel("Iteration")
    plt.ylabel("Metric value")
    plt.title("Iteration History")
    plt.grid(True, alpha=0.6)
    plt.legend()
    plt.tight_layout()
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
    print("- throughput_comparison.png")
    print("- distance_comparison.png")
    print("- iteration_history.png")


if __name__ == "__main__":
    main()
