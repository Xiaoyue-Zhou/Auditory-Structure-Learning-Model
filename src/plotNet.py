import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def make_kk_layout(adjacency, community, target_angle=45):
    """
    Kamada-Kawai layout, rotated so that the two community centroids
    lie approximately along a diagonal.
    """
    adjacency = np.asarray(adjacency)
    community = np.asarray(community)

    G = nx.from_numpy_array(
        (adjacency > 0).astype(int)
    )

    pos = nx.kamada_kawai_layout(
        G,
        weight=None,
        scale=1.0
    )

    nodes = np.arange(adjacency.shape[0])
    xy = np.array([pos[i] for i in nodes])

    # Centre layout
    xy -= xy.mean(axis=0)

    # Rotate the line between two community centroids
    unique_comm = np.unique(community)

    if len(unique_comm) == 2:
        centre_0 = xy[community == unique_comm[0]].mean(axis=0)
        centre_1 = xy[community == unique_comm[1]].mean(axis=0)

        direction = centre_1 - centre_0
        current_angle = np.arctan2(direction[1], direction[0])
        desired_angle = np.deg2rad(target_angle)

        theta = desired_angle - current_angle

        rotation = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        xy = xy @ rotation.T

    # Normalise into a square coordinate system
    xy -= xy.mean(axis=0)
    xy /= np.max(np.abs(xy))

    return {
        int(node): xy[i]
        for i, node in enumerate(nodes)
    }

def plot_original_network(adjacency,
                          community,
                          node_deg,
                          ax=None,
                          pos=None,
                          title="Original network"):
    
    G = nx.from_numpy_array(
        (adjacency > 0).astype(int)
    )

    if pos is None:
        pos = make_kk_layout(
            adjacency,
            community,
            target_angle=45
        )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(4, 4)
        )

    # Equivalent to scale_size(range=c(13,16))
    node_sizes = np.interp(
        node_deg,
        (node_deg.min(), node_deg.max()),
        (500, 800)
    )

    # Similar to scale_color_brewer(palette="Set2")
    palette = sns.color_palette(
        "pastel",
        n_colors=len(np.unique(community))
    )

    unique_comm = np.unique(community)

    color_map = {
        comm: palette[i]
        for i, comm in enumerate(unique_comm)
    }

    node_colors = [
        color_map[c]
        for c in community
    ]

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        width=2.3,
        alpha=0.7,
        edge_color="#393939",
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="none"
    )

    ax.set_title(title)
    ax.set_axis_off()

    # Force a square plotting region
    limit = 1.18

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    ax.set_aspect(
        "equal",
        adjustable="box"
    )

    ax.set_box_aspect(1)

    return pos

def plot_learned_network(
    W,
    community,
    node_deg,
    pos,
    ax=None,
    title="Learned representation",
    threshold=0.0,
    edge_width_range=(0.3, 4.0)
):
    W = np.asarray(W, dtype=float).copy()
    community = np.asarray(community)
    node_deg = np.asarray(node_deg)

    # Directed W -> undirected visual representation
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)

    G = nx.Graph()
    G.add_nodes_from(range(W.shape[0]))

    for i in range(W.shape[0]):
        for j in range(i + 1, W.shape[1]):
            if W[i, j] > threshold:
                G.add_edge(
                    i,
                    j,
                    weight=W[i, j]
                )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(4, 4)
        )

    node_sizes = np.interp(
        node_deg,
        (node_deg.min(), node_deg.max()),
        (500, 800)
    )

    palette = sns.color_palette(
        "pastel",
        n_colors=len(np.unique(community))
    )

    unique_comm = np.unique(community)

    color_map = {
        comm: palette[i]
        for i, comm in enumerate(unique_comm)
    }

    node_colors = [
        color_map[c]
        for c in community
    ]

    edge_list = list(G.edges())

    edge_weights = np.array([
        G[u][v]["weight"]
        for u, v in edge_list
    ])

    if len(edge_weights) > 0:
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=edge_list,
            width=2.3,
            edge_color=edge_weights,
            edge_cmap=plt.cm.Greys,
            alpha=0.8
        )

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="none"
    )

    ax.set_title(title)
    ax.set_axis_off()

    limit = 1.18

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    ax.set_aspect(
        "equal",
        adjustable="box"
    )

    ax.set_box_aspect(1)

    return ax