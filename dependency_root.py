import numpy as np
import pandas as pd


# =========================================================
# 1. BUILD INTER-SIGNAL DEPENDENCY MAP
# =========================================================
def build_dependency_map(
    live_input,
    baseline_corr=None,
    percentile=85
):
    """
    Build dynamic sensor dependency topology
    using absolute correlation graph
    """
    df = live_input.copy()
    df = df.drop(columns=["timestamp"], errors="ignore")

    corr = df.corr().abs().fillna(0)

    corr_values = corr.values[
        np.triu_indices_from(corr, k=1)
    ]

    dynamic_threshold = np.percentile(
        corr_values,
        percentile
    )

    if baseline_corr is not None:
        baseline_vals = baseline_corr.values[
            np.triu_indices_from(baseline_corr, k=1)
        ]

        dynamic_threshold = max(
            dynamic_threshold,
            np.percentile(baseline_vals, percentile)
        )

    dependency_map = {}
    edge_weights = {}

    for col in corr.columns:
        linked = corr.index[
            (corr[col] >= dynamic_threshold) &
            (corr.index != col)
        ].tolist()

        dependency_map[col] = linked

        edge_weights[col] = {
            node: round(float(corr.loc[col, node]), 3)
            for node in linked
        }

    return dependency_map, corr, dynamic_threshold, edge_weights


# =========================================================
# 2. ROOT SOURCE DETECTION (FALLBACK)
# =========================================================
def root_source_detection(live_input):
    """
    Fallback statistical root source detection
    using latest robust z-score deviation
    """
    df = live_input.copy()
    df = df.drop(columns=["timestamp"], errors="ignore")

    latest = df.iloc[-1]

    median = df.median()
    mad = (df - median).abs().median() + 1e-6

    robust_z = 0.6745 * (latest - median) / mad
    robust_z = robust_z.abs()

    return robust_z.idxmax()


# =========================================================
# 3. FUSION-ASSISTED ROOT SOURCE
# =========================================================
def fusion_assisted_root_source(fusion_result):
    """
    Primary root source from fusion detector.
    Falls back safely if unavailable.
    """
    top_signals = fusion_result.get(
        "top_contributing_signals",
        []
    )

    if len(top_signals) == 0:
        return None

    return top_signals[0]


# =========================================================
# 4. TOPOLOGY SUMMARY FOR AI CHAT
# =========================================================
def summarize_topology(dependency_map):
    """
    Generate graph summary for LLM context
    """
    total_nodes = len(dependency_map)

    edge_count = sum(
        len(v) for v in dependency_map.values()
    )

    top_hubs = sorted(
        dependency_map.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:5]

    return {
        "total_nodes": total_nodes,
        "total_edges": edge_count,
        "top_dependency_hubs": {
            k: len(v) for k, v in top_hubs
        }
    }


# =========================================================
# 5. QUICK TEST
# =========================================================
if __name__ == "__main__":
    from fusion_detector import twinpulse_live_detect

    live_input = pd.read_csv("live_sensor_window.csv")

    print("--------- INTER-SIGNAL DEPENDENCY -----------")

    dependency_map, corr, threshold, edge_weights = (
        build_dependency_map(live_input)
    )

    print("Dynamic Threshold:", round(float(threshold), 3))
    print("Dependency Hubs:", summarize_topology(dependency_map))

    print("\n--------- ROOT CAUSE DETECTION -----------")

    # fallback mode
    fallback_root = root_source_detection(live_input)
    print("Fallback Root Sensor:", fallback_root)

    # fusion-assisted mode
    fusion_result = twinpulse_live_detect(
        live_input,
        severity_k=8
    )

    fusion_root = fusion_assisted_root_source(
        fusion_result
    )

    print("Fusion Root Sensor:", fusion_root)

    print("\n--------- SAMPLE DEPENDENCY MAP -----------")
    sample_key = list(dependency_map.keys())[0]
    print(sample_key, "→", dependency_map[sample_key])

    print("\n--------- EDGE WEIGHTS -----------")
    print(sample_key, "→", edge_weights[sample_key])