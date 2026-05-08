import os
from groq import Groq


# =========================================================
# 1. SAFE CLIENT FACTORY
# =========================================================
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        api_key = "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"

    if not api_key:
        return None

    return Groq(api_key=api_key)


# =========================================================
# 2. NEXT BEST ACTION
# =========================================================
def next_best_action(severity, radius):
    if severity in ["Critical", "Emergency Shutdown"]:
        return "Immediate subsystem isolation recommended"

    if severity in ["Warning", "High Risk"]:
        return "Schedule preventive maintenance in next shift"

    return "Continue observation and monitor trend"


# =========================================================
# 3. OPERATOR COPILOT
# =========================================================
def twinpulse_operator_copilot(
    severity,
    root_sensor,
    impacted_nodes,
    ranking,
    graph_summary=None
):
    top_risks = ranking.head(5).to_dict()

    return {
        "root_cause_explanation": (
            f"{root_sensor} is the highest fusion anomaly contributor. "
            f"The anomaly may propagate to {len(impacted_nodes)} dependent signals."
        ),
        "top_contributing_signals": list(top_risks.keys()),
        "signal_ranking": top_risks,
        "graph_summary": graph_summary,
        "next_best_action": next_best_action(
            severity,
            len(impacted_nodes)
        )
    }


# =========================================================
# 4. CHAT COPILOT (FINAL JUDGE VERSION)
# =========================================================
def chat_with_copilot(user_message, latest_result, page="general"):
    client = get_groq_client()

    if client is None:
        return "• AI unavailable\n• Check GROQ API key"

    fusion = latest_result.get("fusion_result", {})
    diagnosis = latest_result.get("diagnosis", {})
    drift = latest_result.get("concept_drift", {})
    memory = latest_result.get("future_memory", {})
    alerts = latest_result.get("alerts", {})
    failure_dna = latest_result.get("failure_dna", {})
    lifecycle = latest_result.get("model_lifecycle", {})
    feedback = latest_result.get("adaptive_feedback", [])

    # last 4 anomalies
    last_4 = feedback[-4:] if isinstance(feedback, list) else []

    page_context = {
        "dashboard": "Focus on live alerts, fusion severity, upload analytics and KPI summary.",
        "topology": "Focus on root cause sensor, blast radius, cascading risk and shutdown dependencies.",
        "memory": "Focus on failure DNA, novelty score, historical similarity and recovery path.",
        "adaptive": "Focus on drift, false positives, retraining decision and model lifecycle.",
        "copilot": "Provide complete system-level industrial reasoning."
    }.get(page, "Provide system-wide explanation.")

    prompt = f"""
You are TwinPulse Industrial AI Judge Copilot.

STRICT RESPONSE FORMAT:
- Answer in crisp bullet points
- Maximum 6 bullets
- Very simple non-technical English
- Directly answer the question
- Use section headers only if needed
- Mention business impact where useful

SUPPORTED QUESTIONS INCLUDE:
- root cause
- next cascading failure
- historical repeat
- maintenance action
- blast radius
- ignored for 10 minutes
- shutdown risk
- recovery steps
- compare with failure DNA
- retraining due to drift
- last 4 anomalies

PAGE MODE:
{page_context}

LIVE SYSTEM CONTEXT:
• Severity: {fusion.get("severity")}
• Fusion Score: {fusion.get("fusion_score")}
• Root Sensor: {diagnosis.get("root_sensor")}
• Impact Radius: {diagnosis.get("impact_radius")}
• Impacted Nodes: {diagnosis.get("impacted_nodes")}
• Drift Status: {drift.get("drift_status")}
• Drift Detected: {drift.get("drift_detected")}
• Alert: {alerts.get("message")}
• Future Cascade: {memory.get("future_cascade_prediction")}
• Recovery Path: {memory.get("alternative_recovery_path")}
• Failure DNA: {failure_dna}
• Model Version: {lifecycle.get("current_version")}
• Last 4 anomalies: {last_4}

USER QUESTION:
{user_message}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You explain industrial AI outputs for non-technical judges "
                    "using bullet points, direct actions, and clear business language."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.4,
        max_tokens=450
    )

    return response.choices[0].message.content