import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend-safe for FastAPI
import matplotlib.pyplot as plt
from datetime import datetime


DRIFT_FILE = "fusion_score_history.csv"


# =====================================================
# 1) INITIALIZE DRIFT HISTORY
# =====================================================
def initialize_drift_history():
    if not os.path.exists(DRIFT_FILE):
        df = pd.DataFrame(columns=[
            "timestamp",
            "fusion_score"
        ])
        df.to_csv(DRIFT_FILE, index=False)


# =====================================================
# 2) LOG CURRENT SCORE
# =====================================================
def log_fusion_score(fusion_score):
    initialize_drift_history()

    df = pd.read_csv(DRIFT_FILE)

    # duplicate prevention
    if len(df) > 0:
        last_score = float(df.iloc[-1]["fusion_score"])
        if abs(last_score - float(fusion_score)) < 1e-6:
            return df

    new_row = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(),
        "fusion_score": float(fusion_score)
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(DRIFT_FILE, index=False)

    return df


# =====================================================
# 3) DETECT CONCEPT DRIFT
# =====================================================
def detect_concept_drift(window=10, drift_threshold=0.15):
    """
    Compare rolling recent mean vs baseline
    """
    initialize_drift_history()

    df = pd.read_csv(DRIFT_FILE)

    if len(df) < window * 2:
        return {
            "drift_detected": False,
            "drift_status": "Insufficient History",
            "reason": "Not enough history yet"
        }

    recent = df["fusion_score"].tail(window).mean()
    baseline = df["fusion_score"].head(window).mean()

    drift_value = abs(recent - baseline)

    # trend direction
    if len(df) >= window + 1:
        previous = df["fusion_score"].tail(window + 1).head(window).mean()
        drift_velocity = round(float(recent - previous), 3)
    else:
        drift_velocity = 0.0

    if drift_value >= 0.25:
        drift_status = "Critical Drift"
    elif drift_value >= 0.15:
        drift_status = "High Drift"
    elif drift_value >= drift_threshold:
        drift_status = "Moderate Drift"
    else:
        drift_status = "Stable"

    return {
        "drift_detected": drift_value > drift_threshold,
        "drift_status": drift_status,
        "drift_value": round(float(drift_value), 3),
        "drift_velocity": drift_velocity,
        "recent_mean": round(float(recent), 3),
        "baseline_mean": round(float(baseline), 3)
    }


# =====================================================
# 4) DRIFT VISUALIZATION
# =====================================================
def plot_drift_trend(window=10):
    initialize_drift_history()

    df = pd.read_csv(DRIFT_FILE)

    if len(df) < window:
        return None

    df["rolling_mean"] = (
        df["fusion_score"]
        .rolling(window)
        .mean()
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        df["fusion_score"],
        label="Fusion Score",
        linewidth=1.8
    )

    ax.plot(
        df["rolling_mean"],
        label=f"Rolling Mean ({window})",
        linewidth=2.5
    )

    ax.axhline(
        df["fusion_score"].head(window).mean(),
        linestyle="--",
        linewidth=1.5,
        label="Baseline Mean"
    )

    ax.set_title("TwinPulse Concept Drift Trend")
    ax.set_xlabel("Run Index")
    ax.set_ylabel("Fusion Score")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    return fig


# =====================================================
# 5) DRIFT SUMMARY FOR AI CHAT
# =====================================================
def drift_summary():
    result = detect_concept_drift()

    return {
        "status": result.get("drift_status"),
        "value": result.get("drift_value"),
        "velocity": result.get("drift_velocity")
    }


# =====================================================
# 6) LOCAL TEST
# =====================================================
if __name__ == "__main__":
    log_fusion_score(0.31)
    log_fusion_score(0.35)
    log_fusion_score(0.41)
    log_fusion_score(0.49)
    log_fusion_score(0.58)
    log_fusion_score(0.61)

    print(detect_concept_drift())

    fig = plot_drift_trend()
    if fig:
        fig.savefig("drift_test.png")
        print("Saved drift_test.png")