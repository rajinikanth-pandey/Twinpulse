import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from main import twinpulse_main


# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="TwinPulse Live Monitoring",
    page_icon="🚀",
    layout="wide",
)

st.title("🚀 TwinPulse Live Sensor Copilot")
st.markdown(
    "Upload live sensor CSV data and run the full TwinPulse orchestration pipeline."
)


# =====================================
# SIDEBAR
# =====================================
st.sidebar.header("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader(
    "📂 Upload live sensor CSV",
    type=["csv"],
)


# =====================================
# HELPERS
# =====================================
def render_any_figure(fig_obj):
    """Render matplotlib/plotly figures safely in Streamlit."""
    if fig_obj is None:
        st.info("📉 Drift trend plot will appear after enough pipeline runs are collected (default: 20 runs).")
        return

    # matplotlib figure
    if hasattr(fig_obj, "savefig"):
        st.pyplot(fig_obj, use_container_width=True)
        return

    # matplotlib axes
    if hasattr(fig_obj, "figure"):
        st.pyplot(fig_obj.figure, use_container_width=True)
        return

    # plotly figure
    if hasattr(fig_obj, "to_plotly_json"):
        st.plotly_chart(fig_obj, use_container_width=True)
        return

    st.warning("Unsupported figure format returned by pipeline.")


def clean_export_payload(result: dict) -> dict:
    """Remove non-serializable figure objects before JSON export."""
    export_result = result.copy()

    topology = export_result.get("topology", {}).copy()
    topology.pop("figure", None)
    export_result["topology"] = topology

    export_result.pop("concept_drift_plot", None)

    return export_result


# =====================================
# MAIN FLOW
# =====================================
if uploaded_file is not None:
    temp_csv_path = None

    try:
        # -----------------------------
        # Preview uploaded data
        # -----------------------------
        df_preview = pd.read_csv(uploaded_file)

        st.subheader("📄 Live Sensor Preview")
        st.dataframe(df_preview.head(20), use_container_width=True)

        # -----------------------------
        # Safe numeric trend chart
        # -----------------------------
        numeric_df = df_preview.select_dtypes(include=["number"])

        if not numeric_df.empty:
            st.subheader("📈 Live Sensor Trends")
            st.line_chart(numeric_df.tail(100), use_container_width=True)
        else:
            st.warning(
                "No numeric sensor columns available for chart rendering."
            )

        # -----------------------------
        # Run pipeline button
        # -----------------------------
        if st.sidebar.button(
            "🚀 Run TwinPulse Pipeline",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Running TwinPulse orchestration..."):
                # Save uploaded CSV temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".csv",
                ) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    temp_csv_path = tmp.name

                # Run master pipeline
                result = twinpulse_main(temp_csv_path)

            st.success("✅ TwinPulse pipeline executed successfully!")

            # =====================================
            # KPI ROW
            # =====================================
            diagnosis = result["diagnosis"]
            fusion = result["fusion_result"]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("🚨 Root Sensor", diagnosis["root_sensor"])

            with col2:
                st.metric("📡 Impact Radius", diagnosis["impact_radius"])

            with col3:
                st.metric("⚠️ Severity", fusion["severity"])

            # =====================================
            # MAIN RESULT PANELS
            # =====================================
            left, right = st.columns(2)

            with left:
                st.subheader("📊 Fusion Result")
                st.json(result["fusion_result"])

                st.subheader("📉 Concept Drift")
                st.json(result.get("concept_drift", {}))

                drift_fig = result.get("concept_drift_plot")
                render_any_figure(drift_fig)

                st.subheader("🧠 Diagnosis")
                st.json(result["diagnosis"])

            with right:
                st.subheader("👨‍🏭 Operator Copilot")
                st.json(result["copilot"])

                st.subheader("🔮 Future Memory")
                st.json(result["future_memory"])

                st.subheader("🔁 Adaptive Feedback")
                st.json(result["adaptive_feedback"])

            # =====================================
            # TOPOLOGY GRAPH
            # =====================================
            st.subheader("🗺️ Operator Intelligence Topology")

            topology_fig = result.get("topology", {}).get("figure")

            render_any_figure(topology_fig)

            # =====================================
            # TOPOLOGY SUMMARY
            # =====================================
            st.subheader("🧭 Topology Summary")
            st.json(
                {
                    "root_sensor": result["topology"]["root_sensor"],
                    "propagation_path": result["topology"][
                        "propagation_path"
                    ],
                    "critical_assets": result["topology"][
                        "critical_assets"
                    ],
                }
            )

            # =====================================
            # EXPORT JSON
            # =====================================
            st.subheader("💾 Export Full Output")

            export_result = clean_export_payload(result)

            st.download_button(
                label="⬇️ Download JSON",
                data=json.dumps(export_result, indent=2, default=str),
                file_name="twinpulse_output.json",
                mime="application/json",
                use_container_width=True,
            )

    except Exception as e:
        st.error(f"❌ Pipeline failed: {str(e)}")

    finally:
        # Optional temp file cleanup safety
        if temp_csv_path and Path(temp_csv_path).exists():
            try:
                Path(temp_csv_path).unlink()
            except Exception:
                pass

else:
    st.info("⬅️ Upload a CSV file from the sidebar to begin.")
