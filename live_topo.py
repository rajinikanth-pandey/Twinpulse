import networkx as nx
import pandas as pd
import plotly.graph_objects as go

from dependency_root import build_dependency_map
from severity import severity_ranking, first_failing, impact_radius


# =====================================================
# INTERACTIVE TOPOLOGY VIEW
# =====================================================
def operator_topology_view(
    dependency_map,
    root_sensor,
    impacted_nodes,
    ranking,
    edge_weights=None,
    top_k=5,
    max_second_hop=2
):
    G = nx.DiGraph()

    if root_sensor is None:
        return {
            "root_sensor": None,
            "propagation_path": [],
            "critical_assets": [],
            "graph_summary": {},
            "figure": None
        }

    # -----------------------------------
    # ROOT + PROPAGATION
    # -----------------------------------
    G.add_node(root_sensor)

    for node in impacted_nodes:
        G.add_node(node)
        G.add_edge(root_sensor, node)

        second_hop = dependency_map.get(node, [])[:max_second_hop]

        for sub in second_hop:
            if sub != root_sensor:
                G.add_node(sub)
                G.add_edge(node, sub)

    # -----------------------------------
    # CRITICAL ASSETS
    # -----------------------------------
    if isinstance(ranking, pd.DataFrame):
        critical_assets = ranking.head(top_k).index.tolist()
    else:
        critical_assets = ranking.head(top_k).index.tolist()

    for critical in critical_assets:
        if critical not in G:
            G.add_node(critical)
            G.add_edge(root_sensor, critical)

    # -----------------------------------
    # LAYOUT
    # -----------------------------------
    pos = nx.spring_layout(
        G,
        seed=42,
        k=1.8,
        iterations=300
    )

    # -----------------------------------
    # EDGES
    # -----------------------------------
    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1.5),
        hoverinfo="none"
    )

    # -----------------------------------
    # NODES
    # -----------------------------------
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        role = "Dependent Sensor"

        if node == root_sensor:
            color = "red"
            size = 36
            role = "Root Cause"

        elif node in critical_assets:
            color = "gold"
            size = 28
            role = "Critical Asset"

        elif node in impacted_nodes:
            color = "orange"
            size = 24
            role = "Impacted Sensor"

        else:
            color = "skyblue"
            size = 20

        node_color.append(color)
        node_size.append(size)

        node_text.append(
            f"{node}<br>"
            f"Role: {role}<br>"
            f"Connections: {len(list(G.neighbors(node)))}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=list(G.nodes()),
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color="white")
        )
    )

    fig = go.Figure(
        data=[edge_trace, node_trace]
    )

    fig.update_layout(
        title="TwinPulse Interactive Topology Intelligence",
        showlegend=False,
        template="plotly_dark",
        height=700,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    graph_summary = {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "critical_assets": critical_assets,
        "max_depth": 2
    }

    return {
        "root_sensor": root_sensor,
        "propagation_path": impacted_nodes,
        "critical_assets": critical_assets,
        "graph_summary": graph_summary,
        "figure": fig
    }


# =====================================================
# LOCAL TEST
# =====================================================
if __name__ == "__main__":
    live_input = pd.read_csv("live_sensor_window.csv")

    dependency_map, corr, threshold, edge_weights = (
        build_dependency_map(live_input)
    )

    root_sensor = first_failing(live_input)

    radius_count, impacted_nodes = impact_radius(
        root_sensor,
        dependency_map
    )

    ranking = severity_ranking(live_input)

    operator_result = operator_topology_view(
        dependency_map,
        root_sensor,
        impacted_nodes,
        ranking
    )

    print(operator_result["graph_summary"])

    if operator_result["figure"] is not None:
        operator_result["figure"].show()