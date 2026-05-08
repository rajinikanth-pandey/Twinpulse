import pandas as pd
import numpy as np
from dependency_root import build_dependency_map


# =========================================================
# SEVERITY LABELS
# =========================================================
SEVERITY_LEVELS = [
    "Healthy",
    "Info",
    "Low Risk",
    "Moderate",
    "Warning",
    "High Risk",
    "Critical",
    "Emergency Shutdown"
]


def map_score_to_severity(score, k=8):
    levels = SEVERITY_LEVELS[:k]
    score = max(0.0, min(float(score), 1.0))
    idx = min(int(score * len(levels)), len(levels) - 1)
    return levels[idx]


# =========================================================
# 1. FIRST FAILING SENSOR
# =========================================================
def first_failing(live_input, threshold=2.5):
    """
    Detect first sensor that breached anomaly threshold
    """
    df = live_input.copy()
    df = df.drop(columns=["timestamp"], errors="ignore")

    median = df.median()
    mad = (df - median).abs().median() + 1e-6

    robust_z = (
        0.6745 * (df - median) / mad
    ).abs()

    breach_times = {}

    for col in robust_z.columns:
        breach_idx = np.where(
            robust_z[col] > threshold
        )[0]

        if len(breach_idx) > 0:
            breach_times[col] = int(breach_idx[0])

    if not breach_times:
        return None

    return min(
        breach_times,
        key=breach_times.get
    )


# =========================================================
# 2. SENSOR SEVERITY RANKING
# =========================================================
def severity_ranking(live_input, k=8):
    """
    Rank sensors by robust anomaly severity
    """
    df = live_input.copy()
    df = df.drop(columns=["timestamp"], errors="ignore")

    median = df.median()
    mad = (df - median).abs().median() + 1e-6

    robust_z = (
        0.6745 * (df - median) / mad
    ).abs()

    rank_scores = robust_z.mean().sort_values(
        ascending=False
    )

    normalized = rank_scores / (rank_scores.max() + 1e-6)

    result = pd.DataFrame({
        "risk_score": normalized.round(3),
        "severity": normalized.apply(
            lambda x: map_score_to_severity(x, k)
        )
    })

    return result


# =========================================================
# 3. IMPACT RADIUS (MULTI-HOP)
# =========================================================
def impact_radius(root_sensor, dependency_map, depth=2):
    """
    Multi-hop propagation radius
    """
    if root_sensor is None:
        return 0, []

    visited = set()
    frontier = {root_sensor}

    for _ in range(depth):
        next_frontier = set()

        for node in frontier:
            linked = dependency_map.get(node, [])
            next_frontier.update(linked)

        visited.update(next_frontier)
        frontier = next_frontier

    impacted = sorted(list(visited))

    return len(impacted), impacted


# =========================================================
# 4. AI SUMMARY
# =========================================================
def summarize_risk_table(ranking_df, top_n=10):
    top = ranking_df.head(top_n)

    return top.to_dict(orient="index")


# =========================================================
# 5. QUICK TEST
# =========================================================
if __name__ == "__main__":
    live_input = pd.read_csv("live_sensor_window.csv")

    dependency_map, corr, threshold, edge_weights = (
        build_dependency_map(live_input)
    )

    first_fail = first_failing(live_input)

    ranking = severity_ranking(live_input)

    radius_count, impacted_nodes = impact_radius(
        first_fail,
        dependency_map,
        depth=2
    )

    print("🚨 First failing sensor:", first_fail)
    print("🔗 Dynamic threshold:", round(float(threshold), 3))
    print("📡 Impact radius:", radius_count)
    print("🧬 Impacted signals:", impacted_nodes)

    print("\n📊 Top severity ranking:")
    print(ranking.head(10))