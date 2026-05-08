import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="TwinPulse Digital Twin API")

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
templates.env.cache = None

CURRENT_CSV_FILE = None
LATEST_RESULT = {}
LATEST_RAW_INPUT = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fusion_detector import twinpulse_live_detect
from dependency_root import build_dependency_map, fusion_assisted_root_source
from severity import severity_ranking, impact_radius
from live_topo import operator_topology_view
from reason import twinpulse_operator_copilot, chat_with_copilot
from memory_engine import (
    future_cascade_prediction,
    risk_if_ignored,
    alternative_recovery_path,
)
from anomaly_detector import get_feedback_history
from drift_engine import (
    log_fusion_score,
    detect_concept_drift,
    plot_drift_trend,
)
from failure_dna import (
    build_failure_dna,
    evaluate_signature_novelty,
    save_failure_dna,
    add_to_self_growing_dataset,
)
from redeploy import lifecycle_summary


class ChatRequest(BaseModel):
    message: str
    page: str = "general"


def sanitize(obj):
    import math

    if obj is None:
        return None
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if math.isnan(val) or math.isinf(val) else val
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [sanitize(v) for v in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return sanitize(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return sanitize(obj.to_dict())
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize(v) for v in obj]
    return obj


def twinpulse_main(csv_path: str):
    global LATEST_RAW_INPUT

    live_input = pd.read_csv(BASE_DIR / csv_path)
    LATEST_RAW_INPUT = live_input.copy()

    fusion_result = twinpulse_live_detect(live_input, severity_k=8)
    top_signals = fusion_result.get("top_contributing_signals", [])

    root_sensor = fusion_assisted_root_source(fusion_result) or live_input.columns[1]

    severity = fusion_result["severity"]
    fusion_score = fusion_result["fusion_score"]

    log_fusion_score(fusion_score)
    drift_result = detect_concept_drift()

    dependency_map, _, threshold, edge_weights = build_dependency_map(live_input)
    ranking = severity_ranking(live_input)

    radius_count, impacted_nodes = impact_radius(
        root_sensor, dependency_map, depth=2
    )

    operator_result = operator_topology_view(
        dependency_map,
        root_sensor,
        impacted_nodes,
        ranking,
        edge_weights=edge_weights,
    )

    topology_safe = dict(operator_result)
    topology_safe.pop("figure", None)

    copilot_result = twinpulse_operator_copilot(
        severity,
        root_sensor,
        impacted_nodes,
        ranking,
        operator_result.get("graph_summary", {}),
    )

    signature = build_failure_dna(
        root_sensor=root_sensor,
        fusion_score=fusion_score,
        severity=severity,
        impact_radius=radius_count,
        critical_assets=operator_result["critical_assets"],
        propagation_path=impacted_nodes,
        top_contributing_signals=top_signals,
    )

    is_new, novelty_score = evaluate_signature_novelty(signature)

    if is_new:
        save_failure_dna(signature, float(novelty_score))
        add_to_self_growing_dataset(signature)

    feedback_history = get_feedback_history()

    future_nodes = future_cascade_prediction(
        root_sensor, dependency_map, depth=3
    )

    future_risk = risk_if_ignored(severity, future_nodes)
    recovery_path = alternative_recovery_path(root_sensor, severity)

    return sanitize({
        "fusion_result": fusion_result,
        "concept_drift": drift_result,
        "failure_dna": {
            "signature": signature,
            "novelty_score": novelty_score,
        },
        "diagnosis": {
            "dynamic_corr_threshold": round(float(threshold), 3),
            "root_sensor": root_sensor,
            "impact_radius": int(radius_count),
            "impacted_nodes": impacted_nodes,
            "top_risks": ranking.head(5).to_dict(orient="index"),
        },
        "topology": topology_safe,
        "copilot": copilot_result,
        "future_memory": {
            "future_cascade_prediction": future_nodes,
            "risk_if_ignored": future_risk,
            "alternative_recovery_path": recovery_path,
        },
        "adaptive_feedback": feedback_history.tail(10).to_dict(orient="records"),
        "model_lifecycle": lifecycle_summary(),
    })


# =========================
# HTML ROUTES (FINAL FOR YOUR VERSION)
# =========================
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"request": request},
    )


@app.get("/memory", response_class=HTMLResponse)
async def memory_page(request: Request):
    return templates.TemplateResponse(
        request,
        "memory.html",
        {"request": request},
    )


@app.get("/topology", response_class=HTMLResponse)
async def topology_page(request: Request):
    return templates.TemplateResponse(
        request,
        "topology.html",
        {"request": request},
    )


@app.get("/copilot", response_class=HTMLResponse)
async def copilot_page(request: Request):
    return templates.TemplateResponse(
        request,
        "copilot.html",
        {"request": request},
    )


@app.get("/adaptive", response_class=HTMLResponse)
async def adaptive_page(request: Request):
    return templates.TemplateResponse(
        request,
        "adaptive.html",
        {"request": request},
    )

# =========================
# APIs
# =========================
@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    global CURRENT_CSV_FILE, LATEST_RESULT

    upload_path = BASE_DIR / file.filename
    with open(upload_path, "wb") as f:
        f.write(await file.read())

    CURRENT_CSV_FILE = file.filename
    LATEST_RESULT = twinpulse_main(CURRENT_CSV_FILE)

    return LATEST_RESULT


@app.get("/api/live-demo")
async def live_demo():
    global LATEST_RESULT
    live_file = "live_sensor_stream_100.csv"

    if not (BASE_DIR / live_file).exists():
        return JSONResponse({
            "status": "error",
            "message": "Run live_sensor_stream.py first",
        })

    LATEST_RESULT = twinpulse_main(live_file)

    return {
        "status": "live data processed",
        "severity": LATEST_RESULT["fusion_result"]["severity"],
        "root_sensor": LATEST_RESULT["diagnosis"]["root_sensor"],
    }


@app.get("/api/memory-data")
async def memory_data():
    if not LATEST_RESULT:
        return {
            "fingerprint": "No signature",
            "novelty_score": None,
            "future_cascade": [],
            "recovery_path": [],
        }

    failure_dna = LATEST_RESULT.get("failure_dna", {})
    future_memory = LATEST_RESULT.get("future_memory", {})
    signature = failure_dna.get("signature", {})

    return {
        "fingerprint": signature.get("signature_id", "No signature"),
        "novelty_score": failure_dna.get("novelty_score"),
        "future_cascade": future_memory.get("future_cascade_prediction", []),
        "recovery_path": future_memory.get("alternative_recovery_path", []),
    }


@app.get("/api/historical-cascade")
async def historical_cascade():
    """
    Memory-specific cascade replay graph.
    Shows only historical root + future predicted cascade.
    """
    if not LATEST_RESULT:
        return HTMLResponse(
            "<h3 style='color:white;padding:20px'>Run Live Demo first</h3>"
        )

    future_nodes = (
        LATEST_RESULT
        .get("future_memory", {})
        .get("future_cascade_prediction", [])
    )

    root_sensor = (
        LATEST_RESULT
        .get("diagnosis", {})
        .get("root_sensor", "unknown")
    )

    import plotly.graph_objects as go
    import math

    nodes = [root_sensor] + future_nodes[:8]

    x_vals = [0]
    y_vals = [0]

    for i in range(len(future_nodes[:8])):
        angle = 2 * math.pi * i / max(1, len(future_nodes[:8]))
        x_vals.append(math.cos(angle))
        y_vals.append(math.sin(angle))

    edge_x = []
    edge_y = []

    for i in range(1, len(nodes)):
        edge_x += [x_vals[0], x_vals[i], None]
        edge_y += [y_vals[0], y_vals[i], None]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=2, color="red"),
        hoverinfo="none",
    ))

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="markers+text",
        text=nodes,
        textposition="top center",
        marker=dict(
            size=[30] + [18] * (len(nodes) - 1),
            color=["red"] + ["white"] * (len(nodes) - 1),
        ),
    ))

    fig.update_layout(
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )

    return HTMLResponse(
        fig.to_html(
            full_html=False,
            include_plotlyjs="cdn",
        )
    )

@app.post("/api/copilot-chat")
async def copilot_chat(req: ChatRequest):
    global LATEST_RESULT

    if not LATEST_RESULT:
        return {
            "reply": (
                "No live anomaly context available yet. "
                "Please upload CSV or run live demo first."
            )
        }

    reply = chat_with_copilot(
        req.message,
        LATEST_RESULT,
        req.page
    )

    return {"reply": reply}

@app.get("/api/topology-graph")
async def topology_graph():
    """
    Live topology dependency graph for topology page.
    """
    if LATEST_RAW_INPUT is None or not LATEST_RESULT:
        return HTMLResponse(
            "<h3 style='color:white;padding:20px'>Run Live Demo or Upload CSV first</h3>"
        )

    dependency_map, _, _, edge_weights = build_dependency_map(
        LATEST_RAW_INPUT
    )

    root_sensor = LATEST_RESULT["diagnosis"]["root_sensor"]
    impacted_nodes = LATEST_RESULT["diagnosis"]["impacted_nodes"]

    ranking = severity_ranking(LATEST_RAW_INPUT)

    operator_result = operator_topology_view(
        dependency_map,
        root_sensor,
        impacted_nodes,
        ranking,
        edge_weights=edge_weights,
    )

    return HTMLResponse(
        operator_result["figure"].to_html(
            full_html=False,
            include_plotlyjs="cdn",
        )
    )

@app.get("/api/drift-graph")
async def drift_graph():
    fig = plot_drift_trend()

    if fig is None:
        return HTMLResponse("<h3>Not enough drift history</h3>")

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")