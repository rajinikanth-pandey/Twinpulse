import pandas as pd
import os
import json
from datetime import datetime


SELF_GROW_FILE = "self_growing_dataset.csv"
MODEL_REGISTRY = "model_registry.json"
STAT_CONFIG_FILE = "statistical_config.json"


# =====================================================
# 1) INITIALIZE MODEL REGISTRY
# =====================================================
def initialize_model_registry():
    if not os.path.exists(MODEL_REGISTRY):
        registry = {
            "current_version": "v1",
            "last_retrain_time": None,
            "best_validation_score": 0.90,
            "deployment_status": "active",
            "main_branch": "Statistical",
            "rollback_available": False
        }

        with open(MODEL_REGISTRY, "w") as f:
            json.dump(registry, f, indent=4)


# =====================================================
# 2) CHECK RETRAIN REQUIREMENT
# =====================================================
def check_retrain_trigger(
    drift_result,
    min_new_samples=7
):
    initialize_model_registry()

    if not os.path.exists(SELF_GROW_FILE):
        return {
            "retrain_required": False,
            "reason": "No self-growing dataset found"
        }

    df = pd.read_csv(SELF_GROW_FILE)

    # sample threshold logic
    new_sample_count = len(df)
    enough_new_data = new_sample_count >= min_new_samples

    # drift logic
    drift_detected = bool(
        drift_result.get("drift_detected", False)
    )
    drift_status = drift_result.get(
        "drift_status", "Stable"
    )

    # critical anomaly pressure
    critical_count = (
        df["severity"].isin(
            ["Critical", "Emergency Shutdown"]
        )
    ).sum()

    critical_pressure = critical_count >= 2

    retrain_required = (
        enough_new_data or
        drift_detected or
        critical_pressure
    )

    reason = []

    if enough_new_data:
        reason.append(
            f"{new_sample_count} new signatures reached"
        )

    if drift_detected:
        reason.append(
            f"concept drift detected ({drift_status})"
        )

    if critical_pressure:
        reason.append(
            f"{critical_count} critical signatures accumulated"
        )

    if not reason:
        reason.append(
            f"waiting until {min_new_samples} samples"
        )

    return {
        "retrain_required": retrain_required,
        "new_samples": new_sample_count,
        "threshold_required": min_new_samples,
        "critical_signatures": int(critical_count),
        "reason": ", ".join(reason)
    }


# =====================================================
# 3) MAIN STATISTICAL BRANCH RETRAINING
# =====================================================
def retrain_main_statistical_branch():
    if not os.path.exists(SELF_GROW_FILE):
        return {
            "status": "no_dataset"
        }

    df = pd.read_csv(SELF_GROW_FILE)

    zscore_threshold = round(
        max(2.0, min(3.5, df["fusion_score"].mean() * 5)),
        3
    )

    corr_percentile = int(
        min(97, max(80, df["impact_radius"].mean() * 8 + 80))
    )

    return {
        "main_retrained_branch": "Statistical",
        "zscore_threshold": zscore_threshold,
        "corr_percentile": corr_percentile,
        "status": "recalibrated"
    }


# =====================================================
# 4) VALIDATION ENGINE
# =====================================================
def validate_new_model(
    old_score=0.90,
    simulated_new_score=0.94
):
    improved = simulated_new_score > old_score

    delta = round(
        float(simulated_new_score - old_score),
        3
    )

    return {
        "validated": improved,
        "old_score": old_score,
        "new_score": simulated_new_score,
        "delta": delta
    }


# =====================================================
# 5) AUTO REDEPLOY
# =====================================================
def auto_redeploy(validation_result, retrain_assets=None):
    initialize_model_registry()

    if not validation_result["validated"]:
        return {
            "redeployed": False,
            "reason": "new calibration underperformed"
        }

    with open(MODEL_REGISTRY, "r") as f:
        registry = json.load(f)

    current_version = registry["current_version"]
    version_num = int(current_version.replace("v", "")) + 1
    new_version = f"v{version_num}"

    registry["rollback_available"] = True
    registry["current_version"] = new_version
    registry["last_retrain_time"] = datetime.now().isoformat()
    registry["best_validation_score"] = validation_result["new_score"]
    registry["deployment_status"] = "active"

    with open(MODEL_REGISTRY, "w") as f:
        json.dump(registry, f, indent=4)

    # save latest statistical config
    if retrain_assets is not None:
        with open(STAT_CONFIG_FILE, "w") as f:
            json.dump(
                {
                    "version": new_version,
                    "zscore_threshold": retrain_assets["zscore_threshold"],
                    "corr_percentile": retrain_assets["corr_percentile"]
                },
                f,
                indent=4
            )

    return {
        "redeployed": True,
        "new_version": new_version,
        "validation_score": validation_result["new_score"]
    }


# =====================================================
# 6) MODEL LIFECYCLE SUMMARY
# =====================================================
def lifecycle_summary():
    initialize_model_registry()

    with open(MODEL_REGISTRY, "r") as f:
        registry = json.load(f)

    return registry


# =====================================================
# 7) LOCAL TEST
# =====================================================
if __name__ == "__main__":
    drift_result = {
        "drift_detected": False,
        "drift_status": "Stable"
    }

    retrain_status = check_retrain_trigger(drift_result)
    print("RETRAIN CHECK:", retrain_status)

    if retrain_status["retrain_required"]:
        retrain_assets = retrain_main_statistical_branch()
        print("RETRAIN:", retrain_assets)

        validation = validate_new_model()
        print("VALIDATION:", validation)

        redeploy = auto_redeploy(validation, retrain_assets)
        print("REDEPLOY:", redeploy)

    print("LIFECYCLE:", lifecycle_summary())