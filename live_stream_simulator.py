import pandas as pd
import numpy as np
import time
from datetime import datetime

CSV_FILE = "live_sensor_stream_100.csv"
NUM_ROWS = 200


# =====================================================
# 100 INDUSTRIAL SENSOR TAGS
# =====================================================
SENSOR_TAGS = [
    "reactor_temp", "coolant_temp", "bearing_temp", "oil_temp",
    "ambient_temp", "exhaust_temp", "pump_pressure", "line_pressure",
    "tank_pressure", "hydraulic_pressure", "steam_pressure",
    "flow_rate_main", "flow_rate_aux", "oil_flow", "coolant_flow",
    "gas_flow", "motor_current", "motor_voltage", "power_draw",
    "torque_load", "shaft_speed", "rpm_main", "rpm_aux",
    "bearing_vibration_x", "bearing_vibration_y", "bearing_vibration_z",
    "pump_vibration", "fan_vibration", "compressor_vibration",
    "gearbox_vibration", "valve_position", "control_signal",
    "actuator_load", "cooling_fan_speed", "pump_speed",
    "compressor_speed", "fuel_level", "oil_level", "water_level",
    "tank_level", "humidity", "gas_leak_ppm", "smoke_density",
    "air_quality_index", "ph_level", "conductivity",
    "viscosity", "density", "torque_feedback", "strain_gauge",
    "load_cell", "belt_tension", "chain_tension",
    "voltage_phase_a", "voltage_phase_b", "voltage_phase_c",
    "current_phase_a", "current_phase_b", "current_phase_c",
    "frequency", "power_factor", "cooling_efficiency",
    "heat_exchange_rate", "compressor_efficiency",
    "filter_clog_index", "lubrication_index",
    "machine_health_score", "rul_indicator", "failure_probability",

    # 🔥 extra 30 realistic plant signals
    "boiler_temp", "boiler_pressure", "steam_flow_rate",
    "condensate_level", "cooling_tower_temp",
    "cooling_tower_flow", "turbine_rpm",
    "generator_voltage", "generator_current",
    "generator_frequency", "grid_sync_phase",
    "transformer_temp", "transformer_oil_level",
    "switchgear_temp", "breaker_status",
    "compressor_inlet_temp", "compressor_outlet_temp",
    "compressor_discharge_pressure",
    "pump_suction_pressure", "pump_discharge_pressure",
    "seal_leak_rate", "bearing_load",
    "shaft_alignment_error", "valve_leak_rate",
    "pipe_wall_temp", "pipe_corrosion_index",
    "lubrication_flow_rate", "fan_current",
    "air_intake_temp", "vibration_envelope"
]


# =====================================================
# DATA GENERATOR
# =====================================================
def generate_live_sensor_data():
    data = {}

    data["timestamp"] = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for _ in range(NUM_ROWS)
    ]

    for tag in SENSOR_TAGS:
        base = np.random.normal(50, 5, NUM_ROWS)

        # realistic anomaly injection
        if np.random.rand() < 0.35:
            anomaly_idx = np.random.randint(120, 199)
            base[anomaly_idx:] += np.random.uniform(10, 25)

        # degradation trend for health / RUL
        if "health" in tag or "rul" in tag:
            base = np.linspace(95, 50, NUM_ROWS) + np.random.normal(0, 1, NUM_ROWS)

        # failure probability trend
        if "failure_probability" in tag:
            base = np.linspace(0.05, 0.98, NUM_ROWS) + np.random.normal(0, 0.03, NUM_ROWS)

        # vibration tags more noisy
        if "vibration" in tag:
            base += np.random.normal(0, 2, NUM_ROWS)

        # pressure tags slow rise
        if "pressure" in tag:
            base += np.linspace(0, 10, NUM_ROWS)

        # temperature drift
        if "temp" in tag:
            base += np.linspace(0, 6, NUM_ROWS)

        data[tag] = base.round(3)

    return pd.DataFrame(data)


# =====================================================
# LIVE STREAM LOOP
# =====================================================
def run_live_stream():
    print("🚀 Industrial live 100-sensor stream started (200 rows)...")

    while True:
        df = generate_live_sensor_data()
        df.to_csv(CSV_FILE, index=False)

        print(
            f"✅ Updated {CSV_FILE} "
            f"with {NUM_ROWS} rows × {len(SENSOR_TAGS)} industrial tags"
        )

        time.sleep(20)


if __name__ == "__main__":
    run_live_stream()