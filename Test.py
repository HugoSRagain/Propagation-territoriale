import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm, colors as mcolors
import numpy as np
from scipy.interpolate import interp1d
import random

# -----------------------------------------
# 1. Create the graph with clusters and weighted edges
# -----------------------------------------
G = nx.Graph()
clusters = {
    "Cluster A": [f"A{i}" for i in range(1, 6)],
    "Cluster B": [f"B{i}" for i in range(1, 6)],
    "Cluster C": [f"C{i}" for i in range(1, 6)]
}
# Add nodes with cluster attribute
for cluster, nodes in clusters.items():
    G.add_nodes_from(nodes, module=cluster)

# Add intra-cluster edges (initial weight = 1.0)
for cluster, nodes in clusters.items():
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            G.add_edge(nodes[i], nodes[j], weight=1.0)

# Add inter-cluster edges (initial weight = 0.5)
inter_cluster_edges = [
    ("A1", "B1"), ("A2", "B2"), ("A3", "C1"), ("A4", "C2"),
    ("B3", "C3"), ("B4", "C4"), ("B5", "A5"), ("A3", "B4"), ("B5", "C5")
]
for edge in inter_cluster_edges:
    G.add_edge(edge[0], edge[1], weight=0.5)

nodes_list = list(G.nodes)
edges = list(G.edges)

# -----------------------------------------
# 2. Determine 3D positions and visual configurations
# -----------------------------------------
pos_2d = nx.spring_layout(G, seed=42, k=0.7)
# Generate 3D positions by adding a random z-coordinate
pos_3d = {node: (pos_2d[node][0], pos_2d[node][1], np.random.uniform(-0.3, 0.3))
          for node in G.nodes}

# Fixed colors per cluster for visualization
fixed_node_colors = {
    "Cluster A": "skyblue",
    "Cluster B": "lightgreen",
    "Cluster C": "lightcoral"
}
node_colors = [fixed_node_colors[G.nodes[n]["module"]] for n in nodes_list]

# -----------------------------------------
# 3. Simulation parameters and cluster activation delays
# -----------------------------------------
T = 20             # Number of discrete simulation steps
num_frames = 100   # Number of frames for smooth interpolation

# Propagation parameters (innovation is non-decreasing)
threshold = 0.2    # Minimum average influence required for spreading innovation
base_spread = 0.15 # Lowered spread factor for slower propagation

# Dynamic edge weight evolution parameters
reinforcement_threshold = 0.7  
reinforcement_factor = 0.05    
decay_weight_factor = 0.02     
min_weight = 0.3               
max_weight = 1.5               

# Define activation delays for clusters (in simulation steps)
cluster_delays = {
    "Cluster A": 0,   # Immediate activation
    "Cluster B": 5,   # Delay of 5 steps
    "Cluster C": 10   # Delay of 10 steps
}

# -----------------------------------------
# 4. Simulation loop: Innovation propagation and dynamic edge evolution
# -----------------------------------------
innovation_history = []   # Store innovation values for each node at each step
edge_weights_history = [] # Store edge weights over time

# Initialize innovation: a few nodes start with an innovation value of 1.0
innovation = {node: 0.0 for node in G.nodes}
initial_sources = ["A1", "B1", "C1"]
for source in initial_sources:
    innovation[source] = 1.0
innovation_history.append(innovation.copy())

# Initialize edge weights from the graph
edge_weights = {}
for u, v in G.edges:
    edge_weights[tuple(sorted((u, v)))] = G.edges[u, v]['weight']
edge_weights_history.append(edge_weights.copy())

# Run the simulation for T steps
for t in range(T):
    new_innovation = {}
    # Varying spread factor over time (using cosine modulation)
    spread_t = base_spread + 0.05 * np.cos(2 * np.pi * t / T)

    # Update innovation for each node (ensuring non-decreasing values)
    for node in G.nodes:
        # Get the node's cluster and check activation delay
        node_cluster = G.nodes[node]["module"]
        if t < cluster_delays[node_cluster]:
            # Do not update innovation before activation delay
            new_innovation[node] = innovation[node]
        else:
            neighbor_influence = 0.0
            total_weight = 0.0
            for neighbor in G.neighbors(node):
                edge_key = tuple(sorted((node, neighbor)))
                w = edge_weights[edge_key]
                neighbor_influence += w * innovation[neighbor]
                total_weight += w
            avg_influence = neighbor_influence / total_weight if total_weight > 0 else 0.0

            if avg_influence >= threshold:
                updated_value = innovation[node] + spread_t * avg_influence
            else:
                updated_value = innovation[node]
            new_innovation[node] = np.clip(updated_value, innovation[node], 1.0)

    # Update edge weights dynamically based on node innovation levels
    new_edge_weights = {}
    for u, v in G.edges:
        key = tuple(sorted((u, v)))
        current_weight = edge_weights[key]
        if new_innovation[u] > reinforcement_threshold and new_innovation[v] > reinforcement_threshold:
            updated_weight = current_weight + reinforcement_factor
        else:
            updated_weight = current_weight * (1 - decay_weight_factor)
        updated_weight += np.random.uniform(-0.01, 0.01)  # Add slight noise
        updated_weight = np.clip(updated_weight, min_weight, max_weight)
        new_edge_weights[key] = updated_weight

    innovation = new_innovation
    edge_weights = new_edge_weights
    innovation_history.append(innovation.copy())
    edge_weights_history.append(edge_weights.copy())

# -----------------------------------------
# 5. Interpolate history for smooth animation
# -----------------------------------------
steps_original = np.arange(len(innovation_history))
steps_interp = np.linspace(0, len(innovation_history) - 1, num_frames)

# Cubic interpolation for node innovation values
interp_innovation = {
    node: interp1d(steps_original, [state[node] for state in innovation_history], kind='cubic')(steps_interp)
    for node in G.nodes
}

# Linear interpolation for edge weights
interp_edge_weights = {}
for key in edge_weights_history[0].keys():
    weights_over_time = [state[key] for state in edge_weights_history]
    interp_edge_weights[key] = interp1d(steps_original, weights_over_time, kind='linear')(steps_interp)

# Compute global average and maximum over time for the line chart
global_avg_history = [np.mean(list(state.values())) for state in innovation_history]
global_max_history = [np.max(list(state.values())) for state in innovation_history]
interp_global_avg = interp1d(steps_original, global_avg_history, kind='linear')(steps_interp)
interp_global_max = interp1d(steps_original, global_max_history, kind='linear')(steps_interp)

# Compute average innovation per cluster over time for the bar chart
cluster_avg_history = {cluster: [] for cluster in clusters.keys()}
for state in innovation_history:
    for cluster, nodes in clusters.items():
        cluster_avg_history[cluster].append(np.mean([state[node] for node in nodes]))
interp_cluster_avg = {}
for cluster, values in cluster_avg_history.items():
    interp_cluster_avg[cluster] = interp1d(steps_original, values, kind='linear')(steps_interp)

# -----------------------------------------
# 6. Create a dashboard with four panels:
#    - Left: 3D network plot (spanning all rows)
#    - Right (4 rows): Top - Bar chart, Middle - Line chart, Bottom - Histogram
# -----------------------------------------
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(nrows=4, ncols=2, width_ratios=[2, 1])
ax3d = fig.add_subplot(gs[:, 0], projection='3d')  # 3D network plot spanning 4 rows
ax_bar = fig.add_subplot(gs[0, 1])                  # Bar chart
ax_line = fig.add_subplot(gs[1:3, 1])                 # Line chart (spanning 2 rows)
ax_hist = fig.add_subplot(gs[3, 1])                   # Histogram
gs.update(hspace=0.6)  # Increase vertical spacing on the right side

# -----------------------------------------
# 7. Update function for animation
# -----------------------------------------
def update(frame):
    # Clear all axes
    ax3d.cla()
    ax_bar.cla()
    ax_line.cla()
    ax_hist.cla()

    # Rotate the 3D view for dynamism
    ax3d.view_init(elev=30, azim=frame*3)
    ax3d.set_title(f"3D Network – Frame {frame+1}", fontsize=14)

    # Apply a slight jitter to simulate movement
    jittered_pos = {
        node: (pos_3d[node][0] + np.random.uniform(-0.02, 0.02),
               pos_3d[node][1] + np.random.uniform(-0.02, 0.02),
               pos_3d[node][2] + np.random.uniform(-0.02, 0.02))
        for node in G.nodes
    }

    # Get the interpolated innovation values for this frame
    current_innov = {node: interp_innovation[node][frame] for node in G.nodes}

    # --- 3D Network Plot ---
    xs = [jittered_pos[node][0] for node in nodes_list]
    ys = [jittered_pos[node][1] for node in nodes_list]
    zs = [jittered_pos[node][2] for node in nodes_list]
    sizes = [300 * (1 + 3 * current_innov[node]) for node in nodes_list]
    ax3d.scatter(xs, ys, zs, s=sizes, c=node_colors, edgecolors='black', linewidths=0.5, alpha=0.9)
    for i, node in enumerate(nodes_list):
        ax3d.text(xs[i], ys[i], zs[i] + 0.05, node, fontsize=8, ha='center')

    # Draw edges with thickness and color based on edge weight and innovation
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = cm.plasma
    for u, v in edges:
        key = tuple(sorted((u, v)))
        weight = interp_edge_weights[key][frame]
        avg_innov = (current_innov[u] + current_innov[v]) / 2
        color = cmap(norm(avg_innov))
        linewidth = 2 + 3 * weight
        x_line = [jittered_pos[u][0], jittered_pos[v][0]]
        y_line = [jittered_pos[u][1], jittered_pos[v][1]]
        z_line = [jittered_pos[u][2], jittered_pos[v][2]]
        ax3d.plot(x_line, y_line, z_line, color=color, linewidth=linewidth)

    ax3d.set_axis_off()
    ax3d.set_xlim([-1.2, 1.2])
    ax3d.set_ylim([-1.2, 1.2])
    ax3d.set_zlim([-1, 1])
    global_avg = np.mean(list(current_innov.values()))
    global_max = np.max(list(current_innov.values()))
    ax3d.text2D(0.05, 0.95, f"Global Avg: {global_avg:.2f}\nGlobal Max: {global_max:.2f}",
                transform=ax3d.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

    # --- Bar Chart: Average Innovation per Cluster ---
    cluster_avgs = {cluster: np.mean([current_innov[node] for node in clusters[cluster]])
                    for cluster in clusters}
    ax_bar.bar(cluster_avgs.keys(), cluster_avgs.values(),
               color=[fixed_node_colors[clust] for clust in cluster_avgs.keys()])
    ax_bar.set_ylim([0, 1])
    ax_bar.set_title("Average Innovation per Cluster")
    ax_bar.set_ylabel("Innovation")
    for i, (clust, val) in enumerate(cluster_avgs.items()):
        ax_bar.text(i, val + 0.02, f"{val:.2f}", ha='center', fontsize=9)

    # --- Line Chart: Global Evolution ---
    ax_line.plot(steps_interp[:frame+1], interp_global_avg[:frame+1], label="Global Average", color='navy')
    ax_line.plot(steps_interp[:frame+1], interp_global_max[:frame+1], label="Global Maximum", color='crimson')
    ax_line.set_ylim([0, 1])
    ax_line.set_title("Global Innovation Evolution")
    ax_line.set_xlabel("Interpolated Time")
    ax_line.set_ylabel("Innovation")
    ax_line.legend(loc='upper left', fontsize=9)
    ax_line.grid(True, linestyle='--', alpha=0.5)

    # --- Histogram: Distribution of Node Innovation ---
    innovations = list(current_innov.values())
    ax_hist.hist(innovations, bins=10, range=(0, 1), color='gray', edgecolor='black', alpha=0.7)
    ax_hist.axvline(global_avg, color='red', linestyle='--', linewidth=2, label=f"Avg = {global_avg:.2f}")
    ax_hist.set_title("Innovation Distribution")
    ax_hist.set_xlabel("Innovation")
    ax_hist.set_ylabel("Frequency")
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, linestyle='--', alpha=0.5)

# -----------------------------------------
# 8. Create and save the animation
# -----------------------------------------
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100, repeat=True)
ani.save("Enhanced_Innovation_Propagation_3D_Delayed.mp4", writer="ffmpeg")
plt.close(fig)
