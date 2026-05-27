"""
Revised UAV joint trajectory and time allocation simulation.

This code corresponds to the revised thesis version after supervisor comments.
It uses a fairness-aware minimum time allocation constraint and a small
trajectory update step so that the simplified mobility constraint is respected.
"""

import numpy as np
import matplotlib.pyplot as plt

USERS = np.array([[15, 20], [25, 75], [60, 35], [82, 70]], dtype=float)
START = np.array([0.0, 0.0])
END = np.array([100.0, 100.0])
NUM_SLOTS = 20
ALTITUDE = 20.0
TOTAL_TIME = 10.0
DELTA_T = 0.5
V_MAX = 30.0
BANDWIDTH = 1.0
GAMMA = 4000.0
STEP_SIZE = 0.01
ITERATIONS = 12
TAU_MIN = 0.5


def build_straight_line_trajectory(start, end, num_slots):
    x = np.linspace(start[0], end[0], num_slots)
    y = np.linspace(start[1], end[1], num_slots)
    return np.column_stack((x, y))


def compute_distances(trajectory, users, altitude):
    distances = np.zeros((users.shape[0], trajectory.shape[0]))
    for i in range(users.shape[0]):
        dx = trajectory[:, 0] - users[i, 0]
        dy = trajectory[:, 1] - users[i, 1]
        distances[i, :] = np.sqrt(dx**2 + dy**2 + altitude**2)
    return distances


def compute_average_distances(trajectory, users, altitude):
    return compute_distances(trajectory, users, altitude).mean(axis=1)


def compute_rates(avg_distances, bandwidth, gamma):
    return bandwidth * np.log2(1.0 + gamma / (avg_distances**2))


def fairness_time_allocation(rates, total_time, tau_min):
    n_users = len(rates)
    residual_time = total_time - n_users * tau_min
    if residual_time < 0:
        raise ValueError("tau_min is too large for the total communication time.")
    rates = np.maximum(rates, 1e-9)
    return np.ones(n_users) * tau_min + rates / rates.sum() * residual_time


def compute_total_throughput(rates, time_allocation):
    return float(np.sum(rates * time_allocation))


def compute_average_communication_distance(trajectory, users, altitude):
    return float(compute_distances(trajectory, users, altitude).mean())


def update_trajectory(trajectory, users, time_allocation, step_size, start, end):
    new_trajectory = trajectory.copy()
    weighted_center = np.average(users, axis=0, weights=time_allocation)
    new_trajectory[1:-1] = new_trajectory[1:-1] + step_size * (weighted_center - new_trajectory[1:-1])
    new_trajectory[0] = start
    new_trajectory[-1] = end
    return new_trajectory


def baseline_method():
    trajectory = build_straight_line_trajectory(START, END, NUM_SLOTS)
    time_allocation = np.ones(len(USERS)) * (TOTAL_TIME / len(USERS))
    avg_distances = compute_average_distances(trajectory, USERS, ALTITUDE)
    rates = compute_rates(avg_distances, BANDWIDTH, GAMMA)
    throughput = compute_total_throughput(rates, time_allocation)
    avg_distance = compute_average_communication_distance(trajectory, USERS, ALTITUDE)
    return trajectory, time_allocation, rates, throughput, avg_distance


def proposed_method():
    trajectory = build_straight_line_trajectory(START, END, NUM_SLOTS)
    throughput_history = []
    distance_history = []

    for _ in range(ITERATIONS):
        avg_distances = compute_average_distances(trajectory, USERS, ALTITUDE)
        rates = compute_rates(avg_distances, BANDWIDTH, GAMMA)
        time_allocation = fairness_time_allocation(rates, TOTAL_TIME, TAU_MIN)
        throughput_history.append(compute_total_throughput(rates, time_allocation))
        distance_history.append(compute_average_communication_distance(trajectory, USERS, ALTITUDE))
        trajectory = update_trajectory(trajectory, USERS, time_allocation, STEP_SIZE, START, END)

    avg_distances = compute_average_distances(trajectory, USERS, ALTITUDE)
    rates = compute_rates(avg_distances, BANDWIDTH, GAMMA)
    time_allocation = fairness_time_allocation(rates, TOTAL_TIME, TAU_MIN)
    throughput = compute_total_throughput(rates, time_allocation)
    avg_distance = compute_average_communication_distance(trajectory, USERS, ALTITUDE)
    return trajectory, time_allocation, rates, throughput, avg_distance, throughput_history, distance_history


def plot_figures(baseline_trajectory, proposed_trajectory, baseline_throughput, proposed_throughput, baseline_distance, proposed_distance, throughput_history, distance_history):
    plt.figure(figsize=(7, 6))
    plt.scatter(USERS[:, 0], USERS[:, 1], s=90, marker='o', label='Ground users', zorder=5)
    plt.plot(baseline_trajectory[:, 0], baseline_trajectory[:, 1], linestyle='--', marker='o', linewidth=2.4, markersize=5, label='Baseline trajectory')
    plt.plot(proposed_trajectory[:, 0], proposed_trajectory[:, 1], linestyle='-', marker='s', linewidth=3.2, markersize=5, label='Proposed trajectory')
    plt.scatter(START[0], START[1], s=110, marker='D', label='Start', zorder=6)
    plt.scatter(END[0], END[1], s=130, marker='X', label='End', zorder=6)
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.title('UAV Trajectory Comparison')
    plt.grid(True, alpha=0.6)
    plt.legend(loc='lower right', fontsize=9)
    plt.xlim(-5, 105)
    plt.ylim(-5, 105)
    plt.tight_layout()
    plt.savefig('trajectory_comparison.png', dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    bars = plt.bar(['Baseline', 'Proposed'], [baseline_throughput, proposed_throughput])
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h, f'{h:.2f}', ha='center', va='bottom')
    plt.ylabel('Normalized total throughput')
    plt.title('Total Throughput Comparison')
    plt.tight_layout()
    plt.savefig('throughput_comparison.png', dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    bars = plt.bar(['Baseline', 'Proposed'], [baseline_distance, proposed_distance])
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h, f'{h:.2f}', ha='center', va='bottom')
    plt.ylabel('Average communication distance (m)')
    plt.title('Average Distance Comparison')
    plt.tight_layout()
    plt.savefig('distance_comparison.png', dpi=300)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(7, 5))
    x = np.arange(1, len(throughput_history) + 1)
    ax1.plot(x, throughput_history, marker='o', label='Throughput')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Normalized total throughput')
    ax1.grid(True, alpha=0.6)
    ax2 = ax1.twinx()
    ax2.plot(x, distance_history, marker='s', linestyle='--', label='Average distance')
    ax2.set_ylabel('Average communication distance (m)')
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')
    plt.title('Iteration History of the Proposed Method')
    plt.tight_layout()
    plt.savefig('iteration_history.png', dpi=300)
    plt.close()


def main():
    baseline_trajectory, baseline_tau, baseline_rates, baseline_throughput, baseline_distance = baseline_method()
    proposed_trajectory, proposed_tau, proposed_rates, proposed_throughput, proposed_distance, throughput_history, distance_history = proposed_method()

    max_segment = np.linalg.norm(proposed_trajectory[1:] - proposed_trajectory[:-1], axis=1).max()

    print('=== Baseline Results ===')
    print(f'Total Throughput: {baseline_throughput:.4f}')
    print(f'Average Communication Distance: {baseline_distance:.4f} m')
    print(f'Time Allocation: {baseline_tau}')
    print()
    print('=== Proposed Results ===')
    print(f'Total Throughput: {proposed_throughput:.4f}')
    print(f'Average Communication Distance: {proposed_distance:.4f} m')
    print(f'Time Allocation: {proposed_tau}')
    print(f'Maximum trajectory segment length: {max_segment:.4f} m')
    print()
    print('=== Improvement ===')
    print(f'Throughput improvement: {(proposed_throughput-baseline_throughput)/baseline_throughput*100:.2f}%')
    print(f'Distance reduction: {(baseline_distance-proposed_distance)/baseline_distance*100:.2f}%')

    plot_figures(baseline_trajectory, proposed_trajectory, baseline_throughput, proposed_throughput, baseline_distance, proposed_distance, throughput_history, distance_history)


if __name__ == '__main__':
    main()
