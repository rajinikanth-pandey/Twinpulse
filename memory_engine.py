import pandas as pd
import os
from datetime import datetime


MEMORY_FILE = "failure_memory.csv"


# =====================================================
# 1) INITIALIZE MEMORY
# =====================================================
def initialize_memory():
    required_cols = [
        "timestamp",
        "root_sensor",
        "severity",
        "impact_radius",
        "critical_assets",
        "propagation_path",
        "top_contributing_signals",
        "resolution_action"
    ]

    if not os.path.exists(MEMORY_FILE):
        pd.DataFrame(columns=required_cols).to_csv(
            MEMORY_FILE,
            index=False
        )
    else:
        df = pd.read_csv(MEMORY_FILE)

        for col in required_cols:
            if col not in df.columns:
                df[col] = None

        df = df[required_cols]
        df.to_csv(MEMORY_FILE, index=False)


# =====================================================
# 2) STORE INCIDENT
# =====================================================
def store_failure_memory(
    root_sensor,
    severity,
    impact_radius,
    critical_assets,
    propagation_path,
    top_contributing_signals=None,
    resolution_action="Pending"
):
    initialize_memory()

    if top_contributing_signals is None:
        top_contributing_signals = [root_sensor]

    memory_df = pd.read_csv(MEMORY_FILE)

    new_row = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(),
        "root_sensor": root_sensor,
        "severity": severity,
        "impact_radius": int(impact_radius),
        "critical_assets": ",".join(map(str, critical_assets)),
        "propagation_path": ",".join(map(str, propagation_path)),
        "top_contributing_signals": ",".join(
            map(str, top_contributing_signals)
        ),
        "resolution_action": resolution_action
    }])

    memory_df = pd.concat([memory_df, new_row], ignore_index=True)
    memory_df.to_csv(MEMORY_FILE, index=False)

    return memory_df


# =====================================================
# 3) FIND SIMILAR INCIDENTS
# =====================================================
def find_similar_incidents(
    root_sensor,
    severity=None,
    top_signal=None
):
    initialize_memory()

    memory_df = pd.read_csv(MEMORY_FILE)

    similar = memory_df[
        memory_df["root_sensor"] == root_sensor
    ]

    if severity is not None:
        similar = similar[
            similar["severity"] == severity
        ]

    if top_signal is not None:
        similar = similar[
            memory_df["top_contributing_signals"]
            .fillna("")
            .str.contains(top_signal)
        ]

    return similar


# =====================================================
# 4) GET LAST RESOLUTION
# =====================================================
def get_resolution_memory(root_sensor):
    similar = find_similar_incidents(root_sensor)

    if len(similar) == 0:
        return "No historical resolution found"

    return str(similar.iloc[-1]["resolution_action"])


# =====================================================
# 5) FUTURE CASCADE PREDICTION
# =====================================================
def future_cascade_prediction(
    root_sensor,
    dependency_map,
    depth=3
):
    visited = set()
    frontier = [root_sensor]
    cascade = []

    for _ in range(depth):
        next_frontier = []

        for node in frontier:
            children = dependency_map.get(node, [])

            for child in children:
                if child not in visited and child != root_sensor:
                    visited.add(child)
                    cascade.append(child)
                    next_frontier.append(child)

        frontier = next_frontier

        if not frontier:
            break

    return cascade


# =====================================================
# 6) RISK IF IGNORED
# =====================================================
def risk_if_ignored(severity, future_nodes):
    count = len(future_nodes)
    severity = str(severity)

    if severity in ["Emergency Shutdown", "Critical"]:
        return (
            f"Extreme risk of subsystem-wide cascade across "
            f"{count} downstream sensors"
        )

    elif severity in ["High Risk", "Warning"]:
        return (
            f"High probability of propagation to "
            f"{count} additional sensors"
        )

    elif severity in ["Moderate", "Low Risk"]:
        return (
            f"Limited but growing risk across "
            f"{count} future nodes"
        )

    return "Low short-term risk"


# =====================================================
# 7) ALTERNATIVE RECOVERY PATH
# =====================================================
def alternative_recovery_path(
    root_sensor,
    severity="Warning"
):
    severity = str(severity)

    base_path = [
        f"Isolate {root_sensor} locally",
        "Shift load to redundant subsystem",
        "Run fallback cooling loop",
        "Continue degraded-safe mode"
    ]

    if severity in ["Critical", "Emergency Shutdown"]:
        base_path.append("Prepare emergency shutdown fallback")

    return base_path


# =====================================================
# 8) MEMORY SUMMARY FOR AI CHAT
# =====================================================
def memory_summary(root_sensor):
    similar = find_similar_incidents(root_sensor)

    return {
        "incident_count": len(similar),
        "last_resolution": get_resolution_memory(root_sensor)
    }