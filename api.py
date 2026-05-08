from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from main import twinpulse_main

app = FastAPI(title="TwinPulse API")

# allow frontend html access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/twinpulse")
def get_twinpulse():
    result = twinpulse_main("live_sensor_window_45_sensors.csv")
    return result