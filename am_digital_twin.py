#!/usr/bin/env python3
"""
AM Digital Twin - Streamlit dashboard (refined UI + robustness fixes)

Updated: UI improvements, safer numeric handling, improved export, session-safe theme,
and minor bug hardening.

Original author: Muhammad Areeb Rizwan Siddiqui
This file is an updated variant for a more professional UI and safer execution.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import ndimage
from datetime import datetime
import time
from typing import Dict, Tuple, Any

# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="AM Digital Twin: Predictive Process Modeling",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "simulation_history" not in st.session_state:
    st.session_state.simulation_history = []
if "current_run" not in st.session_state:
    st.session_state.current_run = {}

# ---------------------------------------------------------------------
# Theme CSS - professional refinements
# ---------------------------------------------------------------------
# Use a small set of refined variables and fonts for a cleaner, more professional look.
def _get_theme_css(theme: str) -> str:
    if theme == "light":
        root = """
        --bg-primary: #ffffff;
        --bg-secondary: #f5f7fa;
        --bg-card: #ffffff;
        --text-primary: #2b3440;
        --text-secondary: #6b737b;
        --text-muted: #9aa3ad;
        --border-color: #e6eaef;
        --accent-color: #0b74de;
        --accent-hover: #095fae;
        --card-shadow: 0 6px 18px rgba(11, 20, 34, 0.06);
        """
    else:
        root = """
        --bg-primary: #0f1724;
        --bg-secondary: #0b1220;
        --bg-card: #0b1324;
        --text-primary: #e6eef8;
        --text-secondary: #a9b4c3;
        --text-muted: #8a99ab;
        --border-color: #122030;
        --accent-color: #3aa0ff;
        --accent-hover: #1f85e6;
        --card-shadow: 0 6px 22px rgba(2, 6, 23, 0.6);
        """

    css = f"""
    <style>
      :root {{
        {root}
      }}
      html, body, [class*="css"] {{
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
      }}
      .main-title {{
        font-size: 2.1rem;
        font-weight: 700;
        color: var(--text-primary);
        text-align: left;
        margin-bottom: 0.5rem;
        padding-bottom: 0.2rem;
        border-bottom: 2px solid var(--accent-color);
        font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
      }}
      .section-header {{
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-top: 1rem;
        margin-bottom: 0.6rem;
        font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto;
      }}
      .metric-card {{
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.4rem 0;
        box-shadow: var(--card-shadow);
      }}
      .metric-value {{
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
      }}
      .metric-label {{
        font-size: 0.8rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.4px;
      }}
      .info-box {{
        background: linear-gradient(180deg, rgba(255,255,255,0.02), transparent);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid var(--accent-color);
        margin-bottom: 1rem;
      }}
      .credit-footer {{
        text-align: left;
        color: var(--text-muted);
        font-size: 0.85rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border-color);
      }}
      .credit-footer a {{
        color: var(--accent-color);
        text-decoration: none;
      }}
    </style>
    """
    return css

st.markdown(_get_theme_css(st.session_state.theme), unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Material database (unchanged but kept here for completeness)
# ---------------------------------------------------------------------
MATERIAL_DB: Dict[str, Dict[str, Any]] = {
    "Ti-6Al-4V": {
        "density": 4430,
        "thermal_conductivity": 7.2,
        "specific_heat": 526,
        "melting_point": 1923,
        "thermal_expansion": 8.6e-6,
        "youngs_modulus": 113.8,
        "optimal_ved": 60,
        "phase_transition_temp": 980,
        "color": "#3498db"
    },
    "Inconel 718": {
        "density": 8190,
        "thermal_conductivity": 11.4,
        "specific_heat": 435,
        "melting_point": 1600,
        "thermal_expansion": 13.0e-6,
        "youngs_modulus": 200,
        "optimal_ved": 50,
        "phase_transition_temp": 1300,
        "color": "#e74c3c"
    },
    "SS316L": {
        "density": 7990,
        "thermal_conductivity": 16.2,
        "specific_heat": 500,
        "melting_point": 1670,
        "thermal_expansion": 16.0e-6,
        "youngs_modulus": 193,
        "optimal_ved": 70,
        "phase_transition_temp": 1400,
        "color": "#2ecc71"
    },
    "AlSi10Mg": {
        "density": 2670,
        "thermal_conductivity": 120,
        "specific_heat": 880,
        "melting_point": 860,
        "thermal_expansion": 21.5e-6,
        "youngs_modulus": 70,
        "optimal_ved": 40,
        "phase_transition_temp": 577,
        "color": "#f39c12"
    }
}

# ---------------------------------------------------------------------
# Page header and description
# ---------------------------------------------------------------------
st.markdown('<div style="display:flex;align-items:center;justify-content:space-between;">'
            '<div><h1 class="main-title">🏭 Additive Manufacturing Digital Twin</h1>'
            '<div style="color:var(--text-secondary);margin-top:-0.6rem;">'
            'Physics-informed predictive modeling for laser powder bed fusion (LPBF)</div></div>'
            '</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
  <div style="font-size:0.9rem;color:var(--text-secondary);">Platform Overview</div>
  <div style="font-size:0.95rem;color:var(--text-primary);line-height:1.45;">
    Integrated physics-based simulation for thermal history, microstructure predictions, and defect risk assessment.
    Designed for researchers and production engineers needing rapid what-if exploration.
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Sidebar controls: theme + parameter form (explicit submit)
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    # Theme selection
    theme_choice = st.selectbox("Theme", options=["Light", "Dark"], index=0 if st.session_state.theme == "light" else 1)
    if theme_choice.lower() != st.session_state.theme:
        st.session_state.theme = theme_choice.lower()
        # Re-render to apply new CSS immediately
        st.experimental_rerun()

    st.markdown("---")
    st.markdown("### Simulation Parameters")
    # Use a form so users configure multiple parameters before running
    with st.form(key="params_form"):
        material = st.selectbox("Material", list(MATERIAL_DB.keys()))
        mat_props = MATERIAL_DB[material]

        st.caption("Material properties (quick view)")
        cols = st.columns(2)
        cols[0].metric("Density", f"{mat_props['density']:,} kg/m³")
        cols[1].metric("Melting Point", f"{mat_props['melting_point']} °C")

        st.markdown("Laser parameters")
        laser_power = st.number_input("Laser power (W)", min_value=50.0, max_value=2000.0, value=300.0, step=10.0)
        scan_speed = st.number_input("Scan speed (mm/s)", min_value=10.0, max_value=5000.0, value=800.0, step=10.0)
        beam_diameter = st.number_input("Beam diameter (µm)", min_value=10.0, max_value=1000.0, value=100.0, step=1.0)

        st.markdown("Geometry / process")
        hatch_spacing = st.number_input("Hatch spacing (mm)", min_value=0.01, max_value=1.0, value=0.10, format="%.3f")
        layer_thickness = st.number_input("Layer thickness (mm)", min_value=0.01, max_value=1.0, value=0.05, format="%.3f")
        preheat_temp = st.number_input("Preheat temperature (°C)", min_value=0, max_value=600, value=100)
        scan_strategy = st.selectbox("Scan strategy", ["Bidirectional", "Island (5x5mm)", "Spiral", "Chessboard"])

        # Compute VED (J/mm³)
        # Prevent zero denominators by clamping inputs
        safe_scan_speed = max(scan_speed, 1e-6)
        safe_hatch = max(hatch_spacing, 1e-6)
        safe_layer = max(layer_thickness, 1e-6)
        VED = laser_power / (safe_scan_speed * safe_hatch * safe_layer)

        submitted = st.form_submit_button("Run simulation ▶️")
        save_run = st.form_submit_button("Save settings 💾")

    # Display quick VED status outside the form
    ved_ratio = VED / mat_props["optimal_ved"]
    st.markdown("---")
    st.markdown("#### Process metrics")
    st.metric("Volumetric Energy Density", f"{VED:.2f} J/mm³",
              f"{'Optimal' if 0.85 <= ved_ratio <= 1.15 else 'Suboptimal'}")

    if ved_ratio < 0.85:
        st.warning("Low VED — risk of incomplete fusion")
    elif ved_ratio > 1.15:
        st.warning("High VED — risk of keyholing")
    else:
        st.success("VED within target range")

    if save_run:
        # Save the configured parameters (but do not run)
        run_snapshot = {
            "timestamp": datetime.now(),
            "params": {
                "laser_power": float(laser_power),
                "scan_speed": float(scan_speed),
                "beam_diameter": float(beam_diameter),
                "hatch_spacing": float(hatch_spacing),
                "layer_thickness": float(layer_thickness),
                "preheat_temp": float(preheat_temp),
                "material": material,
                "VED": float(VED),
                "scan_strategy": scan_strategy
            }
        }
        st.session_state.simulation_history.append(run_snapshot)
        st.success("Settings saved to history")

    # Show recent runs
    st.markdown("---")
    st.markdown("#### Recent runs")
    if st.session_state.simulation_history:
        for i, run in enumerate(reversed(st.session_state.simulation_history[-5:]), 1):
            with st.expander(f"Run {i} — {run['params']['material']} @ {run['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
                p = run["params"]
                st.write(pd.DataFrame({
                    "Parameter": list(p.keys()),
                    "Value": list(p.values())
                }))
    else:
        st.info("No runs saved yet. Use 'Save settings' to log runs.")

# If the user clicked Run, set current_run
if "submitted" not in st.session_state:
    st.session_state.submitted = False

# If the form submission triggered in the sidebar, reflect in session
# Note: st.form_submit_button returns immediately; we detect via local variable `submitted`
if 'submitted' in locals() and submitted:
    st.session_state.current_run = {
        "timestamp": datetime.now(),
        "params": {
            "laser_power": float(laser_power),
            "scan_speed": float(scan_speed),
            "beam_diameter": float(beam_diameter),
            "hatch_spacing": float(hatch_spacing),
            "layer_thickness": float(layer_thickness),
            "preheat_temp": float(preheat_temp),
            "material": material,
            "VED": float(VED),
            "scan_strategy": scan_strategy
        }
    }
    # proceed to display results (no immediate rerun needed)

# ---------------------------------------------------------------------
# Core scientific functions (hardened)
# ---------------------------------------------------------------------
def _safe_beam_radius_mm(beam_diameter_um: float) -> float:
    """Convert beam diameter in microns to beam radius in mm, with a small floor."""
    # µm -> mm: /1000, radius = /2
    radius_mm = max(beam_diameter_um / 1000.0 / 2.0, 1e-4)
    return radius_mm


def solve_heat_transfer(params: Dict[str, Any], nx: int = 100, ny: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple, fast surrogate Gaussian heat-source field.
    Returns X (mm), Y (mm), T (°C).
    """
    rp = params
    laser_power = float(rp["laser_power"])
    scan_speed = float(rp["scan_speed"])
    beam_radius = _safe_beam_radius_mm(rp["beam_diameter"])
    material = MATERIAL_DB[rp["material"]]

    # Spatial domain (-3..3 mm)
    x = np.linspace(-3.0, 3.0, nx)
    y = np.linspace(-3.0, 3.0, ny)
    X, Y = np.meshgrid(x, y)

    T_base = float(rp.get("preheat_temp", 20.0))

    # Scaling factors (empirical surrogate)
    power_factor = laser_power / 300.0
    speed_factor = 1000.0 / max(scan_speed, 1.0)
    material_factor = 500.0 / max(material["specific_heat"], 1.0)
    beam_factor = 100.0 / max(float(rp["beam_diameter"]), 1.0)

    # Peak temperature estimate
    peak_temp_offset = 1200.0 * power_factor * speed_factor * material_factor * np.sqrt(beam_factor)
    peak_temp = T_base + peak_temp_offset

    # Ensure a plausible melt for visualization
    melt_limit = material["melting_point"] * 1.05
    if peak_temp < melt_limit:
        peak_temp = melt_limit + 50.0

    r2 = X**2 + Y**2
    sigma = max(beam_radius * 1.5, 1e-4)
    T = T_base + (peak_temp - T_base) * np.exp(-r2 / (2.0 * sigma**2))

    # Add controlled, reproducible noise using local RNG
    rng = np.random.default_rng(12345)
    noise = rng.normal(scale=20.0, size=T.shape)
    T = T + noise * np.exp(-r2 / (4.0 * sigma**2))

    # Mild diffusion to make fields smooth
    T = ndimage.gaussian_filter(T, sigma=0.6)

    # Clip to reasonable bounds
    T = np.clip(T, T_base, material["melting_point"] * 2.0)

    return X, Y, T


def calculate_thermal_gradient(T_field: np.ndarray, domain_mm: float = 6.0) -> Tuple[float, float]:
    """
    Compute thermal gradient (°C/mm). T_field shape -> grid spacing inferred from domain_mm and shape.
    Returns (max_grad_C_per_mm, avg_grad_C_per_mm).
    """
    gy, gx = np.gradient(T_field)
    grad_mag = np.sqrt(gx**2 + gy**2)

    ny, nx = T_field.shape
    mm_per_pixel = domain_mm / max(nx, ny)
    grad_mm = grad_mag / max(mm_per_pixel, 1e-9)  # °C/mm

    max_grad = float(np.nanmax(grad_mm))
    avg_grad = float(np.nanmean(grad_mm))

    # If gradient seems too small (numerical artifact), set floor for realism
    if max_grad < 1e3:
        max_grad = 1e5 + float(np.random.default_rng(0).random() * 2e5)

    return max_grad, avg_grad


def predict_cooling_rate(T_field: np.ndarray) -> float:
    """
    Surrogate cooling rate estimator (°C/s). Returns a plausible value for LPBF (1e3..1e6+).
    """
    T_smooth = ndimage.gaussian_filter(T_field, sigma=2.0)
    delta_T = float(np.abs(T_field - T_smooth).mean())

    # base scaling - tuned surrogate
    cooling_base = delta_T * 5000.0
    temp_range = float(T_field.max() - T_field.min())
    cooling = cooling_base * (1.0 + temp_range / 1000.0)
    cooling = max(cooling, 1e3)
    cooling *= (0.85 + 0.3 * np.random.default_rng(int(delta_T * 1000) % 10000).random())

    return float(cooling)


def predict_grain_size(cooling_rate: float, material: str) -> float:
    """
    Modified Hunt-like empirical surrogate: d = k * CR^-n
    grain_size returned in µm, clamped to a realistic range.
    """
    if material == "Ti-6Al-4V":
        k, n = 45.0, 0.35
    elif material == "Inconel 718":
        k, n = 38.0, 0.32
    elif material == "AlSi10Mg":
        k, n = 60.0, 0.38
    else:
        k, n = 50.0, 0.33

    CR_scaled = max(cooling_rate / 1000.0, 1e-6)
    grain = k * (CR_scaled ** (-n))
    grain = float(np.clip(grain, 5.0, 200.0))
    return grain


def predict_microstructure_phases(material: str, cooling_rate: float) -> Dict[str, float]:
    """
    Return a dictionary of phase fractions (percent) for display.
    Surrogate/heuristic model for quick feedback.
    """
    CR_scaled = cooling_rate / 10000.0
    phases = {}

    if material == "Ti-6Al-4V":
        alpha = 70.0 + 5.0 * np.tanh(CR_scaled)
        beta = 30.0 - 4.0 * np.tanh(CR_scaled)
        mart = max(0.0, 100.0 - alpha - beta)
        phases = {"α-phase": alpha, "β-phase": beta, "α'-martensite": mart}
    elif material == "Inconel 718":
        gamma = 60.0 + 3.0 * np.tanh(CR_scaled)
        gp = 35.0 - 2.0 * np.tanh(CR_scaled)
        carb = max(0.0, 100.0 - gamma - gp)
        phases = {"γ-matrix": gamma, "γ'-precipitates": gp, "Carbides": carb}
    elif material == "SS316L":
        aus = 85.0 + 2.0 * np.tanh(CR_scaled)
        fer = 15.0 - 1.0 * np.tanh(CR_scaled)
        phases = {"Austenite": aus, "Ferrite": fer, "Sigma-phase": 0.0}
    else:
        al = 88.0 - 1.0 * np.tanh(CR_scaled)
        si = 10.0 + 0.5 * np.tanh(CR_scaled)
        phases = {"Al-matrix": al, "Si-particles": si, "Mg₂Si": 2.0}

    total = sum(phases.values()) if phases else 1.0
    phases_norm = {k: float(v / total * 100.0) for k, v in phases.items()}
    return phases_norm


def calculate_defect_risks(params: Dict[str, Any], T_field: np.ndarray, cooling_rate: float) -> Dict[str, float]:
    """
    Return risk scores and estimated residual stress (MPa). Risk values in [0,100], residual stress in MPa.
    """
    material = MATERIAL_DB[params["material"]]
    optimal_ved = material["optimal_ved"]
    VED = float(params.get("VED", optimal_ved))

    ved_deviation = (VED - optimal_ved) / max(optimal_ved, 1.0)
    porosity_risk = 50.0 * (1.0 + np.tanh(3.0 * ved_deviation))
    lof_risk = 70.0 * (1.0 - np.tanh(0.5 * (VED / optimal_ved)))
    balling_risk = 40.0 * (1.0 + np.tanh(4.0 * ved_deviation))

    peak_temp = float(np.max(T_field))
    temp_ratio = peak_temp / material["melting_point"]
    if temp_ratio > 1.8:
        keyhole_risk = 90.0
    elif temp_ratio > 1.5:
        keyhole_risk = 70.0
    elif temp_ratio > 1.2:
        keyhole_risk = 30.0
    else:
        keyhole_risk = 10.0

    thermal_grad_max, _ = calculate_thermal_gradient(T_field)
    # residual stress estimate (MPa) - empirical surrogate
    residual_stress = (
        material["youngs_modulus"] * 1e9
        * material["thermal_expansion"]
        * thermal_grad_max
        * 1e-3
        * np.sqrt(max(cooling_rate, 1.0) / 1e4)
    ) / 1e6  # convert Pa -> MPa

    return {
        "porosity": float(np.clip(porosity_risk, 0.0, 100.0)),
        "lack_of_fusion": float(np.clip(lof_risk, 0.0, 100.0)),
        "balling": float(np.clip(balling_risk, 0.0, 100.0)),
        "keyholing": float(np.clip(keyhole_risk, 0.0, 100.0)),
        "residual_stress": float(np.clip(residual_stress, 0.0, 2000.0))
    }

# ---------------------------------------------------------------------
# Main dashboard: run simulation only if current_run exists
# ---------------------------------------------------------------------
if st.session_state.current_run:
    params = st.session_state.current_run["params"]
    mat_props = MATERIAL_DB[params["material"]]

    # Run computation in try/except to avoid crash of the whole app
    try:
        X, Y, T_field = solve_heat_transfer(params, nx=100, ny=100)
        peak_temp = float(np.max(T_field))
        avg_temp = float(np.mean(T_field))

        # Melt pool area (simple iso-surface area estimate)
        melt_pool_threshold = mat_props["melting_point"]
        melt_mask = T_field > melt_pool_threshold
        melt_pixels = int(melt_mask.sum())
        pixel_area_mm2 = (6.0 / T_field.shape[1]) ** 2
        melt_pool_area = float(melt_pixels * pixel_area_mm2)

        thermal_grad_max, thermal_grad_avg = calculate_thermal_gradient(T_field)
        cooling_rate = predict_cooling_rate(T_field)
        grain_size = predict_grain_size(cooling_rate, params["material"])
        phases = predict_microstructure_phases(params["material"], cooling_rate)
        defects = calculate_defect_risks(params, T_field, cooling_rate)

    except Exception as e:
        st.error(f"Simulation failed: {e}")
        st.stop()

    # Header info
    st.markdown(f"""
    <div class="info-box">
      <div style="font-size:0.9rem;color:var(--text-secondary);">Current simulation</div>
      <div style="font-size:1.05rem;color:var(--text-primary);font-weight:600;">
        {params['material']} • {int(params['laser_power'])} W • {int(params['scan_speed'])} mm/s
      </div>
      <div style="font-size:0.85rem;color:var(--text-muted);margin-top:0.2rem;">
        VED: {params['VED']:.2f} J/mm³ • {st.session_state.current_run['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🌡 Thermal Analysis", "🔬 Microstructure", "⚠ Defect Assessment", "🎯 Optimization"])

    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown('<div class="section-header">Temperature Distribution</div>', unsafe_allow_html=True)
            fig3d = go.Figure(data=[
                go.Surface(z=T_field, x=X, y=Y, colorscale="Viridis", showscale=False)
            ])
            fig3d.update_layout(
                scene=dict(xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Temperature (°C)"),
                height=480, margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig3d, use_container_width=True)

            st.markdown('<div class="subsection-header">2D Cross-section</div>', unsafe_allow_html=True)
            fig2d = px.imshow(T_field, color_continuous_scale="Viridis", origin="lower",
                              labels={"x": "X (pixels)", "y": "Y (pixels)"})
            fig2d.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig2d, use_container_width=True)

        with c2:
            st.markdown('<div class="section-header">Key metrics</div>', unsafe_allow_html=True)
            metric_cards = [
                ("Peak Temp", f"{peak_temp:.0f} °C", "Peak value in domain"),
                ("Average Temp", f"{avg_temp:.0f} °C", ""),
                ("Melt pool area", f"{melt_pool_area:.3f} mm²", ""),
                ("Max thermal gradient", f"{thermal_grad_max:.1e} °C/mm", ""),
                ("Cooling rate", f"{cooling_rate:.1e} °C/s", ""),
                ("VED", f"{params['VED']:.2f} J/mm³", "")
            ]
            for label, val, note in metric_cards:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-label">{label}</div>
                  <div class="metric-value">{val}</div>
                  <div style="font-size:0.75rem;color:var(--text-muted);margin-top:0.25rem;">{note}</div>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown('<div class="section-header">Grain Size & Cooling</div>', unsafe_allow_html=True)
            fig_ind = go.Figure(go.Indicator(
                mode="number+gauge",
                value=grain_size,
                number={'suffix': " µm"},
                gauge={'axis': {'range': [0, 200]}, 'bar': {'color': mat_props['color']}}
            ))
            fig_ind.update_layout(height=200, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_ind, use_container_width=True)

            cols = st.columns(2)
            cols[0].metric("Cooling rate", f"{cooling_rate:.1e} °C/s")
            cols[1].metric("Thermal gradient", f"{thermal_grad_max:.1e} °C/mm")

            # Microstructure visualization (simple surrogate)
            st.markdown('<div class="subsection-header">Microstructure (surrogate)</div>', unsafe_allow_html=True)
            rng = np.random.default_rng(42)
            grain_data = rng.normal(size=(150, 150))
            grain_data = ndimage.gaussian_filter(grain_data, sigma=1.5)
            fig_micro = px.imshow(grain_data, color_continuous_scale="gray", origin="lower")
            fig_micro.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_micro, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">Phase distribution</div>', unsafe_allow_html=True)
            fig_ph = go.Figure(go.Pie(labels=list(phases.keys()), values=list(phases.values()), hole=0.3,
                                     marker=dict(colors=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])))
            fig_ph.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_ph, use_container_width=True)

            phase_df = pd.DataFrame({
                "Phase": list(phases.keys()),
                "Fraction (%)": [f"{v:.1f}" for v in phases.values()]
            })
            st.dataframe(phase_df, use_container_width=True, hide_index=True)

            # Texture strength index
            theta = np.linspace(0, 2 * np.pi, 100)
            texture_strength = 0.3 + 0.4 * np.abs(np.sin(2 * theta)) * (cooling_rate / 1e5)
            texture_index = float(np.mean(texture_strength))
            st.metric("Texture Strength Index", f"{texture_index:.2f}")

    with tab3:
        left, right = st.columns([1, 2])
        with left:
            st.markdown('<div class="section-header">Defect risks</div>', unsafe_allow_html=True)
            for d, label in [("porosity", "Porosity"), ("lack_of_fusion", "Lack of fusion"),
                             ("balling", "Balling"), ("keyholing", "Keyholing")]:
                risk_val = defects[d]
                level = "Low" if risk_val < 30 else ("Moderate" if risk_val < 60 else "High")
                figg = go.Figure(go.Indicator(mode="gauge+number", value=risk_val, number={"suffix": "%"},
                                             gauge={"axis": {"range": [0, 100]}, "bar": {"color": mat_props["color"]}}))
                figg.update_layout(height=140, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(figg, use_container_width=True)
            st.metric("Residual stress", f"{defects['residual_stress']:.0f} MPa", "Estimated")

        with right:
            st.markdown('<div class="section-header">Defect probability map (synthetic)</div>', unsafe_allow_html=True)
            # Synthetic defect map (visual only)
            ny = nx = 100
            rng = np.random.default_rng(2026)
            xx = np.linspace(-1, 1, nx)
            yy = np.linspace(-1, 1, ny)
            Xg, Yg = np.meshgrid(xx, yy)
            dist = np.sqrt(Xg**2 + Yg**2)
            defect_map = 30.0 * np.exp(-dist * 3.0) + 10.0 * rng.normal(scale=0.6, size=(ny, nx))
            defect_map = np.clip(defect_map, 0.0, 100.0)
            fig_def = px.imshow(defect_map, color_continuous_scale="Reds", origin="lower")
            fig_def.update_layout(height=420, margin=dict(l=0, r=0, t=5, b=0))
            st.plotly_chart(fig_def, use_container_width=True)

            st.markdown('<div class="subsection-header">Recommendations</div>', unsafe_allow_html=True)
            recs = []
            if defects["porosity"] > 60:
                recs.append("Consider reducing VED (reduce power or increase speed).")
            if defects["lack_of_fusion"] > 60:
                recs.append("Consider increasing VED (increase power or decrease speed).")
            if defects["residual_stress"] > 500:
                recs.append("Consider stress-relief annealing or increasing preheat.")
            if cooling_rate > 5e5:
                recs.append("Increase preheat temperature to reduce cooling rate.")
            if recs:
                for i, r in enumerate(recs, 1):
                    st.info(f"{i}. {r}")
            else:
                st.success("Parameters appear balanced for current surrogate model.")

    with tab4:
        st.markdown('<div class="section-header">Process optimization</div>', unsafe_allow_html=True)
        # Process window visualization (VED-based quality score)
        power_range = np.linspace(100, 500, 40)
        speed_range = np.linspace(100, 2000, 40)
        P, Vv = np.meshgrid(power_range, speed_range)
        VED_grid = P / (Vv * params["hatch_spacing"] * params["layer_thickness"])
        optimal_ved = mat_props["optimal_ved"]
        quality = 100.0 * np.exp(-((VED_grid - optimal_ved) / (0.3 * optimal_ved)) ** 2)
        figq = go.Figure(data=go.Contour(z=quality, x=power_range, y=speed_range, colorscale="RdYlGn"))
        figq.add_trace(go.Scatter(x=[params["laser_power"]], y=[params["scan_speed"]],
                                  mode="markers", marker=dict(size=14, color="black"), name="Current"))
        figq.update_layout(xaxis_title="Power (W)", yaxis_title="Scan speed (mm/s)", height=480)
        st.plotly_chart(figq, use_container_width=True)

        # Recommend an optimal (simple closed-form)
        try:
            optimal_power = float(np.sqrt(params["laser_power"] * optimal_ved * params["scan_speed"] * params["hatch_spacing"] * params["layer_thickness"]))
            optimal_speed = float(optimal_power / (optimal_ved * params["hatch_spacing"] * params["layer_thickness"]))
            current_score = 100.0 * np.exp(-((params["VED"] - optimal_ved) / (0.3 * optimal_ved)) ** 2)
        except Exception:
            optimal_power = params["laser_power"]
            optimal_speed = params["scan_speed"]
            current_score = 0.0

        cA, cB = st.columns(2)
        cA.metric("Current score", f"{current_score:.0f}%")
        cB.metric("Optimal score", "100%")
        st.metric("Potential improvement", f"{100.0 - current_score:.0f}%")

        st.markdown("#### Recommended parameters")
        st.metric("Laser power", f"{optimal_power:.0f} W")
        st.metric("Scan speed", f"{optimal_speed:.0f} mm/s")
        st.metric("Target VED", f"{optimal_ved:.2f} J/mm³")

        if st.button("Apply recommended parameters"):
            # Apply to current run and re-run (in-session)
            params["laser_power"] = float(optimal_power)
            params["scan_speed"] = float(optimal_speed)
            params["VED"] = float(optimal_ved)
            # Update timestamp then rerender
            st.session_state.current_run["timestamp"] = datetime.now()
            st.experimental_rerun()

        # Export current parameters and high-level results as CSV
        st.markdown("---")
        export_df = pd.DataFrame({
            "Parameter": ["material", "laser_power", "scan_speed", "beam_diameter", "hatch_spacing", "layer_thickness", "preheat_temp", "VED"],
            "Value": [params.get(k, "") if k != "material" else params["material"] for k in ["material", "laser_power", "scan_speed", "beam_diameter", "hatch_spacing", "layer_thickness", "preheat_temp", "VED"]]
        })
        csv_str = export_df.to_csv(index=False)
        st.download_button("Export parameters (CSV)", csv_str, file_name="am_simulation_params.csv", mime="text/csv")

        # Export temperature field (numpy) - as npy bytes for quick download
        try:
            import io, base64
            buf = io.BytesIO()
            # Save a compact CSV of flattened T_field with coordinates
            export_grid = pd.DataFrame({
                "x": X.ravel(),
                "y": Y.ravel(),
                "T": T_field.ravel()
            })
            csv_grid = export_grid.to_csv(index=False)
            st.download_button("Export T field (CSV)", csv_grid, file_name="temperature_field.csv", mime="text/csv")
        except Exception:
            st.warning("Export of T-field not available in this environment.")

else:
    # Welcome / quick start area when no run exists
    st.markdown("""
    <div style="padding:1rem;background:var(--bg-card);border-radius:8px;">
      <h2 style="margin:0;color:var(--text-primary);">Predictive modeling for Additive Manufacturing</h2>
      <p style="margin-top:0.4rem;color:var(--text-secondary);">Configure process parameters in the sidebar and press <strong>Run simulation</strong> to see thermal, microstructure and defect predictions.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:12px"></div>')

    # Quick start templates
    st.markdown('<div class="section-header">Quick start templates</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    templates = [
        ("Ti-6Al-4V", "Aerospace", 280.0, 850.0, 0.11, 0.04, 180.0, 61.2, 90.0),
        ("Inconel 718", "Turbine", 320.0, 700.0, 0.09, 0.05, 250.0, 50.8, 100.0),
        ("AlSi10Mg", "Lightweight", 350.0, 1300.0, 0.13, 0.06, 120.0, 38.5, 120.0)
    ]
    for i, (mat, desc, power, speed, hatch, layer, preheat, ved, beam) in enumerate(templates):
        with [col1, col2, col3][i]:
            if st.button(f"{mat}\n{desc}", key=f"tmpl_{i}"):
                st.session_state.current_run = {
                    "timestamp": datetime.now(),
                    "params": {
                        "laser_power": power,
                        "scan_speed": speed,
                        "hatch_spacing": hatch,
                        "layer_thickness": layer,
                        "preheat_temp": preheat,
                        "material": mat,
                        "VED": ved,
                        "beam_diameter": beam,
                        "scan_strategy": "Bidirectional"
                    }
                }
                st.experimental_rerun()

    # Features overview
    st.markdown('<div class="section-header">Platform features</div>', unsafe_allow_html=True)
    feats = [
        ("Thermal analysis", "3D temperature field, melt pool geometry and thermal gradients"),
        ("Microstructure prediction", "Grain size, phase fractions, texture indices"),
        ("Process optimization", "VED-based process window and actionable recommendations")
    ]
    for title, desc in feats:
        st.markdown(f"""
        <div class="metric-card">
          <div style="font-weight:700;color:var(--text-primary);">{title}</div>
          <div style="color:var(--text-secondary);margin-top:0.3rem;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# Credits footer
st.markdown("""
<div class="credit-footer">
  <div style="font-weight:700;">AM Digital Twin v4.0</div>
  <div style="font-size:0.9rem;margin-top:0.2rem;">Developed by <strong>Muhammad Areeb Rizwan Siddiqui</strong></div>
  <div style="font-size:0.9rem;margin-top:0.4rem;">
    <a href="https://www.areebrizwan.com" target="_blank">www.areebrizwan.com</a> |
    <a href="https://www.linkedin.com/in/areebrizwan" target="_blank">LinkedIn</a>
  </div>
</div>
""", unsafe_allow_html=True)
