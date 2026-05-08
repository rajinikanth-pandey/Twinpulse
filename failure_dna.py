import pandas as pd
import numpy as np
import os
import hashlib
from datetime import datetime


DNA_FILE = "failure_dna.csv"
SELF_GROW_FILE = "self_growing_dataset.csv"


# =====================================================
# 1) INITIALIZE STORAGE
# =====================================================
def initialize_dna_storage():
    required_dna_cols = [
        "signature_id",
        "fingerprint",
        "timestamp",
        "root_sensor",
        "fusion_score",
        "severity",
        "impact_radius",
        "critical_assets",
        "propagation_path",
        "top_contributing_signals",
        "propagation_complexity",
        "novelty_score"
    ]

    required_self_cols = [
        "signature_id",
        "root_sensor",
        "fusion_score",
        "severity",
        "impact_radius",
        "retrain_reason",
        "priority"
    ]

    if not os.path.exists(DNA_FILE):
        pd.DataFrame(columns=required_dna_cols).to_csv(
            DNA_FILE,
            index=False
        )
    else:
        df = pd.read_csv(DNA_FILE)
        for col in required_dna_cols:
            if col not in df.columns:
                df[col] = None
        df = df[required_dna_cols]
        df.to_csv(DNA_FILE, index=False)

    if not os.path.exists(SELF_GROW_FILE):
        pd.DataFrame(columns=required_self_cols).to_csv(
            SELF_GROW_FILE,
            index=False
        )
    else:
        df = pd.read_csv(SELF_GROW_FILE)
        for col in required_self_cols:
            if col not in df.columns:
                df[col] = None
        df = df[required_self_cols]
        df.to_csv(SELF_GROW_FILE, index=False)


# =====================================================
# 2) BUILD FAILURE DNA SIGNATURE
# =====================================================
def build_failure_dna(
    root_sensor,
    fusion_score,
    severity,
    impact_radius,
    critical_assets,
    propagation_path,
    top_contributing_signals=None
):
    initialize_dna_storage()

    if top_contributing_signals is None:
        top_contributing_signals = [root_sensor]

    propagation_complexity = len(set(propagation_path))

    top_signal_string = "-".join(
        sorted(map(str, top_contributing_signals[:5]))
    )

    raw_fingerprint = (
        f"{root_sensor}|{severity}|"
        f"{impact_radius}|{propagation_complexity}|"
        f"{top_signal_string}"
    )

    fingerprint = hashlib.md5(
        raw_fingerprint.encode()
    ).hexdigest()[:12]

    signature = {
        "signature_id": (
            f"{root_sensor}_"
            f"{severity}_"
            f"{impact_radius}_"
            f"{propagation_complexity}"
        ),
        "fingerprint": fingerprint,
        "timestamp": datetime.now().isoformat(),
        "root_sensor": root_sensor,
        "fusion_score": round(float(fusion_score), 3),
        "severity": severity,
        "impact_radius": int(impact_radius),
        "critical_assets": ",".join(sorted(map(str, critical_assets))),
        "propagation_path": ",".join(sorted(map(str, propagation_path))),
        "top_contributing_signals": ",".join(
            sorted(map(str, top_contributing_signals))
        ),
        "propagation_complexity": propagation_complexity
    }

    return signature


# =====================================================
# 3) NOVELTY DETECTION
# =====================================================
def evaluate_signature_novelty(signature):
    initialize_dna_storage()

    dna_df = pd.read_csv(DNA_FILE)

    if len(dna_df) == 0:
        return True, 1.0

    if signature["fingerprint"] in dna_df["fingerprint"].values:
        return False, 0.0

    same_root = dna_df[
        dna_df["root_sensor"] == signature["root_sensor"]
    ]

    if len(same_root) == 0:
        return True, 1.0

    score_dist = abs(
        same_root["fusion_score"] - signature["fusion_score"]
    ).min()

    radius_dist = (
        abs(
            same_root["impact_radius"]
            - signature["impact_radius"]
        ).min() / 10
    )

    complexity_dist = (
        abs(
            same_root["propagation_complexity"]
            - signature["propagation_complexity"]
        ).min() / 10
    )

    severity_match = (
        same_root["severity"] == signature["severity"]
    ).any()

    severity_penalty = 0 if severity_match else 0.25

    novelty_score = round(
        float(
            score_dist +
            radius_dist +
            complexity_dist +
            severity_penalty
        ),
        3
    )

    is_new = novelty_score > 0.18

    return is_new, novelty_score


# =====================================================
# 4) SAVE FAILURE DNA
# =====================================================
def save_failure_dna(signature, novelty_score):
    initialize_dna_storage()

    df = pd.read_csv(DNA_FILE)

    if signature["fingerprint"] in df["fingerprint"].values:
        return df

    signature["novelty_score"] = novelty_score

    new_row = pd.DataFrame([signature])

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(DNA_FILE, index=False)

    return df


# =====================================================
# 5) SELF-GROWING DATASET
# =====================================================
def add_to_self_growing_dataset(
    signature,
    retrain_reason="New fusion anomaly DNA discovered"
):
    initialize_dna_storage()

    df = pd.read_csv(SELF_GROW_FILE)

    if signature["signature_id"] in df["signature_id"].values:
        return df

    priority = (
        "High"
        if signature["severity"] in [
            "Critical",
            "Emergency Shutdown"
        ]
        else "Medium"
    )

    new_row = pd.DataFrame([{
        "signature_id": signature["signature_id"],
        "root_sensor": signature["root_sensor"],
        "fusion_score": signature["fusion_score"],
        "severity": signature["severity"],
        "impact_radius": signature["impact_radius"],
        "retrain_reason": retrain_reason,
        "priority": priority
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(SELF_GROW_FILE, index=False)

    return df


# =====================================================
# 6) DNA SUMMARY FOR AI CHAT
# =====================================================
def dna_summary(root_sensor):
    initialize_dna_storage()

    df = pd.read_csv(DNA_FILE)

    subset = df[df["root_sensor"] == root_sensor]

    return {
        "historical_signatures": len(subset),
        "avg_novelty": round(
            float(subset["novelty_score"].mean()),
            3
        ) if len(subset) else 0.0
    }