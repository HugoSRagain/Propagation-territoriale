# Réimportation des bibliothèques nécessaires après réinitialisation
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib import colors as mcolors
import numpy as np
from scipy.interpolate import interp1d

# Recréation du graphe avec plus de connexions inter-clusters
G = nx.Graph()
modules = {
    "Cluster A": [f"A{i}" for i in range(1, 6)],
    "Cluster B": [f"B{i}" for i in range(1, 6)],
    "Cluster C": [f"C{i}" for i in range(1, 6)]
}
for cluster, nodes in modules.items():
    G.add_nodes_from(nodes, module=cluster)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            G.add_edge(nodes[i], nodes[j])  # connexions complètes intra-cluster

inter_cluster_edges = [
    ("A1", "B1"), ("A2", "B2"), ("A3", "C1"), ("A4", "C2"),
    ("B3", "C3"), ("B4", "C4"), ("B5", "A5")
]
G.add_edges_from(inter_cluster_edges)

nodes_list = list(G.nodes)
edges = list(G.edges())

# Positions 3D
pos_2d = nx.spring_layout(G, seed=42, k=0.7)
pos_3d = {node: (pos_2d[node][0], pos_2d[node][1], np.random.uniform(-0.3, 0.3)) for node in G.nodes}
fixed_node_colors = {
    "Cluster A": "skyblue",
    "Cluster B": "lightgreen",
    "Cluster C": "lightcoral"
}
node_colors = [fixed_node_colors[G.nodes[n]["module"]] for n in G.nodes]

# Simulation de propagation
decay = 0.85
spread = 0.35
innovation = {node: 0.0 for node in G.nodes}
innovation["A1"] = 1.0
history = [innovation.copy()]
for t in range(10):
    new_innovation = {}
    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        incoming = sum(innovation[neighbor] for neighbor in neighbors)
        new_innovation[node] = decay * innovation[node] + spread * incoming / max(len(neighbors), 1)
    innovation = new_innovation
    history.append(innovation.copy())

# Interpolation
num_frames = 30
steps_original = np.arange(len(history))
steps_interp = np.linspace(0, len(history) - 1, num_frames)
interpolated_history = []
for node in G.nodes:
    values = [state[node] for state in history]
    f_interp = interp1d(steps_original, values, kind='linear')
    interpolated_values = f_interp(steps_interp)
    for i, val in enumerate(interpolated_values):
        if len(interpolated_history) <= i:
            interpolated_history.append({})
        interpolated_history[i][node] = val

# Animation
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

def update_3d_enhanced(frame):
    ax.clear()
    ax.set_title(f"Innovation Propagation – Step {frame + 1}", fontsize=14)
    values = interpolated_history[frame]
    xs = [pos_3d[n][0] for n in nodes_list]
    ys = [pos_3d[n][1] for n in nodes_list]
    zs = [pos_3d[n][2] for n in nodes_list]
    node_cluster_color = [fixed_node_colors[G.nodes[n]["module"]] for n in nodes_list]
    ax.scatter(xs, ys, zs, s=300, c=node_cluster_color, edgecolors='black', linewidths=0.5)
    for i, node in enumerate(nodes_list):
        ax.text(xs[i], ys[i], zs[i]+0.05, node, fontsize=9, ha='center')
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = cm.plasma
    for u, v in edges:
        avg_innovation = (values[u] + values[v]) / 2
        color = cmap(norm(avg_innovation))
        x_line = [pos_3d[u][0], pos_3d[v][0]]
        y_line = [pos_3d[u][1], pos_3d[v][1]]
        z_line = [pos_3d[u][2], pos_3d[v][2]]
        ax.plot(x_line, y_line, z_line, color=color, linewidth=2)
    ax.set_axis_off()
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1, 1])

ani_enhanced = animation.FuncAnimation(fig, update_3d_enhanced, frames=num_frames, interval=500, repeat=True)

# Sauvegarde
animation_enhanced_path = "/mnt/data/Innovation_Propagation_3D_Enhanced_Fixed.mp4"
ani_enhanced.save(animation_enhanced_path, writer='ffmpeg')
plt.close()
