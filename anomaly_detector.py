import pandas as pd
import os
from datetime import datetime


FEEDBACK_FILE = "operator_feedback.csv"


# =====================================================
# 1) INITIALIZE FEEDBACK FILE
# =====================================================
def initialize_feedback():
    if not os.path.exists(FEEDBACK_FILE):
        df = pd.DataFrame(columns=[
            "timestamp",
            "root_sensor",
            "predicted_severity",
            "feedback_label",
            "action_taken",
            "operator_note"
        ])
        df.to_csv(FEEDBACK_FILE, index=False)


# =====================================================
# 2) LOG FEEDBACK WITH DUPLICATE PREVENTION
# =====================================================
def log_operator_feedback(
    root_sensor,
    predicted_severity,
    feedback_label,
    action_taken,
    operator_note=""
):
    """
    feedback_label:
        True Anomaly / False Positive
    """
    initialize_feedback()

    df = pd.read_csv(FEEDBACK_FILE)

    # -----------------------------
    # DUPLICATE PREVENTION
    # -----------------------------
    if len(df) > 0:
        last = df.iloc[-1]

        duplicate = (
            last["root_sensor"] == root_sensor
            and last["predicted_severity"] == predicted_severity
            and last["feedback_label"] == feedback_label
            and last["action_taken"] == action_taken
        )

        if duplicate:
            return df

    new_row = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(),
        "root_sensor": root_sensor,
        "predicted_severity": predicted_severity,
        "feedback_label": feedback_label,
        "action_taken": action_taken,
        "operator_note": operator_note
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(FEEDBACK_FILE, index=False)

    return df


# =====================================================
# 3) VIEW FEEDBACK HISTORY
# =====================================================
def get_feedback_history():
    initialize_feedback()
    return pd.read_csv(FEEDBACK_FILE)


# =====================================================
# 4) SENSOR TRUST SCORE
# =====================================================
def sensor_trust_score(root_sensor):
    df = get_feedback_history()

    sensor_df = df[df["root_sensor"] == root_sensor]

    if len(sensor_df) == 0:
        return 1.0

    true_count = (
        sensor_df["feedback_label"] == "True Anomaly"
    ).sum()

    return round(true_count / len(sensor_df), 3)


# =====================================================
# 5) GLOBAL FALSE POSITIVE RATE
# =====================================================
def false_positive_rate():
    df = get_feedback_history()

    if len(df) == 0:
        return 0.0

    fp = (
        df["feedback_label"] == "False Positive"
    ).sum()

    return round(fp / len(df), 3)


# =====================================================
# 6) FEEDBACK SUMMARY FOR AI CHAT
# =====================================================
def feedback_summary(last_n=10):
    df = get_feedback_history()

    recent = df.tail(last_n)

    return {
        "recent_feedback_count": len(recent),
        "false_positive_rate": false_positive_rate(),
        "top_flagged_sensors": recent["root_sensor"]
        .value_counts()
        .head(5)
        .to_dict()
    }


# =====================================================
# 7) RETRAIN READINESS SIGNAL
# =====================================================
def retrain_feedback_signal(threshold=0.35):
    fpr = false_positive_rate()

    return {
        "false_positive_rate": fpr,
        "retrain_recommended": fpr >= threshold
    }


# =====================================================
# 8) LOCAL TEST
# =====================================================
if __name__ == "__main__":
    log_operator_feedback(
        root_sensor="sensor_12",
        predicted_severity="High Risk",
        feedback_label="True Anomaly",
        action_taken="Bearing replaced",
        operator_note="Vibration exceeded baseline"
    )

    print(get_feedback_history().tail())
    print(sensor_trust_score("sensor_12"))
    print(false_positive_rate())
    print(feedback_summary())
    print(retrain_feedback_signal())