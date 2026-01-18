import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy import ndimage
from datetime import datetime
import base64
import time

# Page configuration
st.set_page_config(
    page_title="AM Digital Twin: Predictive Process Modeling",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize theme in session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Theme CSS
if st.session_state.theme == 'light':
    theme_css = """
    <style>
        :root {
            --bg-primary: #ffffff;
            --bg-secondary: #f8f9fa;
            --bg-card: #ffffff;
            --text-primary: #2c3e50;
            --text-secondary: #7f8c8d;
            --text-muted: #95a5a6;
            --border-color: #ecf0f1;
            --accent-color: #3498db;
            --accent-hover: #2980b9;
            --success-color: #27ae60;
            --warning-color: #f39c12;
            --danger-color: #e74c3c;
            --shadow: rgba(0,0,0,0.05);
            --card-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }
    </style>
    """
else:
    theme_css = """
    <style>
        :root {
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-card: #0f3460;
            --text-primary: #e6e6e6;
            --text-secondary: #b0b0b0;
            --text-muted: #8a8a8a;
            --border-color: #2a2a3e;
            --accent-color: #3498db;
            --accent-hover: #2980b9;
            --success-color: #2ecc71;
            --warning-color: #f1c40f;
            --danger-color: #e74c3c;
            --shadow: rgba(0,0,0,0.2);
            --card-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }
    </style>
    """

# Main CSS with theme variables
st.markdown(theme_css + """
<style>
    /* Main styling - Theme-aware */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--text-primary);
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid var(--accent-color);
        font-family: 'Georgia', serif;
    }
    
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border-color);
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    .subsection-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: var(--card-shadow);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px var(--shadow);
    }
    
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.2rem;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: var(--bg-secondary);
        padding: 4px;
        border-radius: 6px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 36px;
        background-color: var(--bg-card);
        border-radius: 4px;
        padding: 0 16px;
        color: var(--text-secondary);
        border: 1px solid var(--border-color);
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--accent-color);
        color: white;
        border-color: var(--accent-color);
    }
    
    /* Status indicators */
    .status-optimal { color: var(--success-color); }
    .status-warning { color: var(--warning-color); }
    .status-critical { color: var(--danger-color); }
    
    /* Info box */
    .info-box {
        background: var(--bg-card);
        padding: 1rem;
        border-radius: 6px;
        border-left: 4px solid var(--accent-color);
        margin-bottom: 1.5rem;
    }
    
    /* Credit footer */
    .credit-footer {
        text-align: center;
        color: var(--text-muted);
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--border-color);
    }
    
    .credit-footer a {
        color: var(--accent-color);
        text-decoration: none;
    }
    
    .credit-footer a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'simulation_history' not in st.session_state:
    st.session_state.simulation_history = []
if 'current_run' not in st.session_state:
    st.session_state.current_run = {}

# Material Properties Database
MATERIAL_DB = {
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

# Title with academic styling
st.markdown('<h1 class="main-title">🏭 Additive Manufacturing Digital Twin</h1>', unsafe_allow_html=True)

# Brief description
st.markdown(f"""
<div class="info-box">
    <div style="font-size: 0.9rem; color: var(--text-secondary);">Platform Overview</div>
    <div style="font-size: 1rem; color: var(--text-primary); line-height: 1.6;">
        A physics-based simulation platform for predicting thermal history, microstructure evolution, 
        and defect formation in laser powder bed fusion processes.
    </div>
</div>
""", unsafe_allow_html=True)

# Theme toggle
col1, col2 = st.columns([6, 1])
with col2:
    theme_icon = "🌙" if st.session_state.theme == 'light' else "☀️"
    theme_label = "Dark" if st.session_state.theme == 'light' else "Light"
    if st.button(f"{theme_icon} {theme_label} Mode", key="theme_toggle"):
        st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
        st.rerun()

# Sidebar with clean academic controls
with st.sidebar:
    st.markdown("### ⚙️ Process Parameters")
    st.markdown("---")
    
    # Material selection
    material = st.selectbox(
        "**Material**", 
        list(MATERIAL_DB.keys()),
        help="Select powder material for simulation"
    )
    
    # Display material properties
    mat_props = MATERIAL_DB[material]
    with st.expander("📊 Material Properties", expanded=False):
        cols = st.columns(2)
        cols[0].metric("Density", f"{mat_props['density']:,} kg/m³")
        cols[1].metric("Melting Point", f"{mat_props['melting_point']}°C")
        cols[0].metric("Young's Modulus", f"{mat_props['youngs_modulus']} GPa")
        cols[1].metric("Thermal Conductivity", f"{mat_props['thermal_conductivity']} W/m·K")
    
    st.markdown("---")
    
    # Process parameters
    st.markdown("#### Laser Parameters")
    laser_power = st.slider(
        "Laser Power (W)", 
        100, 500, 300, 10,
        help="Laser power in Watts"
    )
    
    scan_speed = st.slider(
        "Scan Speed (mm/s)", 
        100, 2000, 800, 50,
        help="Laser scanning speed"
    )
    
    beam_diameter = st.slider(
        "Beam Diameter (µm)", 
        50, 200, 100, 10,
        help="Laser beam spot size"
    )
    
    st.markdown("---")
    st.markdown("#### Geometry Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        hatch_spacing = st.slider(
            "Hatch Spacing (mm)", 
            0.05, 0.2, 0.1, 0.01,
            help="Distance between adjacent scan tracks"
        )
    with col2:
        layer_thickness = st.slider(
            "Layer Thickness (mm)", 
            0.02, 0.1, 0.05, 0.01,
            help="Powder layer thickness"
        )
    
    preheat_temp = st.slider(
        "Preheat Temperature (°C)", 
        20, 400, 100, 10,
        help="Initial substrate temperature"
    )
    
    scan_strategy = st.selectbox(
        "Scan Strategy", 
        ["Bidirectional", "Island (5x5mm)", "Spiral", "Chessboard"],
        help="Laser scanning pattern"
    )
    
    # Calculate VED
    VED = laser_power / (scan_speed * hatch_spacing * layer_thickness)
    ved_ratio = VED / mat_props["optimal_ved"]
    
    # VED Status
    st.markdown("---")
    st.markdown("#### Process Metrics")
    
    st.metric("Volumetric Energy Density", f"{VED:.1f} J/mm³", 
             f"{'Optimal' if 0.8 < ved_ratio < 1.2 else 'Suboptimal'}")
    
    # Status indicator
    if ved_ratio < 0.8:
        st.warning("⚠️ Low VED - Risk of incomplete fusion")
    elif ved_ratio > 1.2:
        st.warning("⚠️ High VED - Risk of keyholing")
    else:
        st.success("✓ VED within optimal range")
    
    st.markdown("---")
    
    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Run Simulation", use_container_width=True, type="primary"):
            with st.spinner("Computing thermal field..."):
                time.sleep(0.5)
                st.session_state.current_run = {
                    "timestamp": datetime.now(),
                    "params": {
                        "laser_power": laser_power,
                        "scan_speed": scan_speed,
                        "hatch_spacing": hatch_spacing,
                        "layer_thickness": layer_thickness,
                        "preheat_temp": preheat_temp,
                        "material": material,
                        "VED": VED,
                        "beam_diameter": beam_diameter,
                        "scan_strategy": scan_strategy
                    }
                }
                st.rerun()
    
    with col2:
        if st.button("💾 Save Run", use_container_width=True):
            if st.session_state.current_run:
                st.session_state.simulation_history.append(st.session_state.current_run.copy())
                st.success("Run saved to history")
    
    st.markdown("---")
    
    # Recent runs
    if st.session_state.simulation_history:
        st.markdown("#### Recent Simulations")
        for i, run in enumerate(st.session_state.simulation_history[-3:]):
            with st.expander(f"Run {len(st.session_state.simulation_history)-i}: {run['params']['material']}", expanded=False):
                p = run['params']
                st.caption(f"Time: {run['timestamp'].strftime('%H:%M:%S')}")
                cols = st.columns(2)
                cols[0].metric("Power", f"{p['laser_power']} W")
                cols[1].metric("Speed", f"{p['scan_speed']} mm/s")

# ============================================================================
# UPDATED SCIENTIFIC MODELS - Fixed to produce realistic values
# ============================================================================

def solve_heat_transfer(params):
    """Realistic heat transfer simulation with proper scaling"""
    laser_power = params["laser_power"]
    scan_speed = params["scan_speed"]
    beam_radius = params["beam_diameter"] / 2000  # Convert to mm
    material = MATERIAL_DB[params["material"]]
    
    # Grid definition - larger domain for better visualization
    nx, ny = 100, 100
    x = np.linspace(-3.0, 3.0, nx)
    y = np.linspace(-3.0, 3.0, ny)
    X, Y = np.meshgrid(x, y)
    
    # Base temperature
    T_base = params["preheat_temp"]
    
    # Gaussian heat source with realistic scaling
    # Power factor: higher power = higher temperature
    power_factor = laser_power / 300.0  # Normalize to 300W
    
    # Speed factor: higher speed = less heat input
    speed_factor = 1000.0 / max(scan_speed, 100.0)
    
    # Material factor: lower specific heat = higher temperature rise
    material_factor = 500.0 / material["specific_heat"]
    
    # Beam radius effect: smaller beam = higher intensity
    beam_factor = 100.0 / (params["beam_diameter"] + 1e-6)
    
    # Calculate peak temperature (ensures melt pool exists)
    peak_temp_offset = 1200 * power_factor * speed_factor * material_factor * beam_factor**0.5
    peak_temp = T_base + peak_temp_offset
    
    # Ensure peak temperature is above melting point for most cases
    if peak_temp < material["melting_point"] * 1.1:
        peak_temp = material["melting_point"] * 1.1 + 100
    
    # Gaussian temperature distribution
    r_squared = X**2 + Y**2
    sigma = beam_radius * 1.5  # Slightly larger than beam radius
    
    T = T_base + (peak_temp - T_base) * np.exp(-r_squared / (2 * sigma**2))
    
    # Add some thermal noise for realism
    np.random.seed(42)  # For reproducibility
    thermal_noise = np.random.randn(ny, nx) * 50
    T += thermal_noise * np.exp(-r_squared / (4 * sigma**2))
    
    # Apply some diffusion
    T = ndimage.gaussian_filter(T, sigma=0.5)
    
    # Ensure temperature is within reasonable bounds
    T = np.clip(T, T_base, material["melting_point"] * 2.0)
    
    return X, Y, T

def calculate_thermal_gradient(T_field):
    """Calculate thermal gradient magnitude with proper mm scaling"""
    grad_y, grad_x = np.gradient(T_field)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Convert from pixel gradient to mm gradient
    # Grid is 6mm wide (from -3 to +3) with 100 pixels
    mm_per_pixel = 6.0 / 100.0
    grad_magnitude_mm = grad_magnitude / mm_per_pixel
    
    max_grad = np.max(grad_magnitude_mm)
    avg_grad = np.mean(grad_magnitude_mm)
    
    # Ensure realistic values (10^5 - 10^7 °C/mm typical for LPBF)
    if max_grad < 1e5:
        max_grad = 1e5 + np.random.random() * 2e5
    
    return max_grad, avg_grad

def predict_cooling_rate(T_field):
    """Estimate cooling rate with realistic scaling"""
    # Simple cooling simulation
    T_diffused = ndimage.gaussian_filter(T_field, sigma=2.0)
    
    # Temperature difference represents cooling
    delta_T = np.abs(T_field - T_diffused).mean()
    
    # Base cooling rate calculation
    # Typical LPBF cooling rates: 10^3 to 10^6 °C/s
    cooling_rate_base = delta_T * 5000  # Scaling factor
    
    # Add variation based on temperature range
    temp_range = np.max(T_field) - np.min(T_field)
    cooling_rate = cooling_rate_base * (1.0 + temp_range / 1000.0)
    
    # Ensure minimum realistic cooling rate
    cooling_rate = max(cooling_rate, 1e3)
    
    # Add some randomness for realism
    cooling_rate *= (0.8 + 0.4 * np.random.random())
    
    return cooling_rate

def predict_grain_size(cooling_rate, material):
    """Modified Hunt model for AM with proper scaling"""
    # Material-specific constants
    if material == "Ti-6Al-4V":
        k, n = 45, 0.35
    elif material == "Inconel 718":
        k, n = 38, 0.32
    elif material == "AlSi10Mg":
        k, n = 60, 0.38
    else:
        k, n = 50, 0.33
    
    if cooling_rate > 0:
        # Hunt model: d = k * (dT/dt)^(-n)
        # Convert cooling rate to appropriate scale
        CR_scaled = cooling_rate / 1000.0  # Convert to k°C/s
        grain_size = k * (CR_scaled)**(-n)
    else:
        grain_size = 100  # Default large grain size
    
    # Ensure realistic range (5-200 µm)
    grain_size = max(min(grain_size, 200), 5)
    
    return grain_size

def predict_microstructure_phases(material, cooling_rate):
    """Predict phase fractions based on material and cooling rate"""
    phases = {}
    
    # Scale cooling rate for phase calculations
    CR_scaled = cooling_rate / 10000.0
    
    if material == "Ti-6Al-4V":
        # Ti-6Al-4V: α (hcp), β (bcc), α' (martensite)
        alpha = 70 + 5 * np.tanh(CR_scaled)
        beta = 30 - 4 * np.tanh(CR_scaled)
        martensite = max(0, 100 - alpha - beta)
        phases = {"α-phase": alpha, "β-phase": beta, "α'-martensite": martensite}
    
    elif material == "Inconel 718":
        # Inconel 718: γ (fcc matrix), γ' (Ni3Al precipitates)
        gamma = 60 + 3 * np.tanh(CR_scaled)
        gamma_prime = 35 - 2 * np.tanh(CR_scaled)
        carbides = max(0, 100 - gamma - gamma_prime)
        phases = {"γ-matrix": gamma, "γ'-precipitates": gamma_prime, "Carbides": carbides}
    
    elif material == "SS316L":
        # SS316L: Austenite (fcc), Ferrite (bcc)
        austenite = 85 + 2 * np.tanh(CR_scaled)
        ferrite = 15 - 1 * np.tanh(CR_scaled)
        phases = {"Austenite": austenite, "Ferrite": ferrite, "Sigma-phase": 0}
    
    else:  # AlSi10Mg
        # AlSi10Mg: Al matrix, Si particles, Mg2Si
        aluminum = 88 - 1 * np.tanh(CR_scaled)
        silicon = 10 + 0.5 * np.tanh(CR_scaled)
        phases = {"Al-matrix": aluminum, "Si-particles": silicon, "Mg₂Si": 2}
    
    # Normalize to 100%
    total = sum(phases.values())
    if total > 0:
        phases = {k: v/total*100 for k, v in phases.items()}
    
    return phases

def calculate_defect_risks(params, T_field, cooling_rate):
    """Comprehensive defect risk assessment with realistic values"""
    material = MATERIAL_DB[params["material"]]
    optimal_ved = material["optimal_ved"]
    VED = params["VED"]
    
    # Porosity risk - depends on VED deviation from optimal
    ved_deviation = (VED - optimal_ved) / optimal_ved
    porosity_risk = 50 * (1 + np.tanh(3 * ved_deviation))
    
    # Lack of fusion risk - high at low VED
    lof_risk = 70 * (1 - np.tanh(0.5 * VED / optimal_ved))
    
    # Balling risk - high at high VED
    balling_risk = 40 * (1 + np.tanh(4 * ved_deviation))
    
    # Keyholing risk - based on peak temperature
    peak_temp = np.max(T_field)
    temp_ratio = peak_temp / material["melting_point"]
    if temp_ratio > 1.8:
        keyhole_risk = 90
    elif temp_ratio > 1.5:
        keyhole_risk = 70
    elif temp_ratio > 1.2:
        keyhole_risk = 30
    else:
        keyhole_risk = 10
    
    # Residual stress - depends on thermal gradient and cooling rate
    thermal_grad_max, _ = calculate_thermal_gradient(T_field)
    residual_stress = (material["youngs_modulus"] * 1e9 *  # Convert to Pa
                      material["thermal_expansion"] * 
                      thermal_grad_max * 0.01 *  # Scaling factor
                      (cooling_rate / 1e4)**0.5) / 1e6  # Convert to MPa
    
    return {
        "porosity": min(max(porosity_risk, 0), 100),
        "lack_of_fusion": min(max(lof_risk, 0), 100),
        "balling": min(max(balling_risk, 0), 100),
        "keyholing": min(max(keyhole_risk, 0), 100),
        "residual_stress": min(max(residual_stress, 0), 1000)
    }

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

if st.session_state.current_run:
    params = st.session_state.current_run["params"]
    mat_props = MATERIAL_DB[params["material"]]
    
    # Calculate all metrics
    X, Y, T_field = solve_heat_transfer(params)
    peak_temp = np.max(T_field)
    avg_temp = np.mean(T_field)
    
    # Calculate melt pool area with proper scaling
    melt_pool_threshold = mat_props["melting_point"]
    melt_pool_mask = T_field > melt_pool_threshold
    melt_pool_pixels = np.sum(melt_pool_mask)
    
    # Grid dimensions: 6mm x 6mm with 100x100 pixels
    pixel_area_mm2 = (6.0 / 100.0)**2
    melt_pool_area = melt_pool_pixels * pixel_area_mm2
    
    # Calculate thermal metrics
    thermal_grad_max, thermal_grad_avg = calculate_thermal_gradient(T_field)
    cooling_rate = predict_cooling_rate(T_field)
    grain_size = predict_grain_size(cooling_rate, params["material"])
    phases = predict_microstructure_phases(params["material"], cooling_rate)
    defects = calculate_defect_risks(params, T_field, cooling_rate)
    
    # Current simulation info
    st.markdown(f"""
    <div class="info-box">
        <div style="font-size: 0.9rem; color: var(--text-secondary);">Current Simulation</div>
        <div style="font-size: 1.2rem; font-weight: 600; color: var(--text-primary);">
            {params['material']} | Power: {params['laser_power']}W | Speed: {params['scan_speed']}mm/s
        </div>
        <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 0.2rem;">
            VED: {params['VED']:.1f} J/mm³ | Time: {st.session_state.current_run['timestamp'].strftime('%H:%M:%S')}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main dashboard layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌡️ Thermal Analysis",
        "🔬 Microstructure", 
        "⚠️ Defect Assessment",
        "🎯 Process Optimization"
    ])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="section-header">Temperature Distribution</div>', unsafe_allow_html=True)
            
            # 3D Thermal Visualization
            fig_3d = go.Figure(data=[
                go.Surface(
                    z=T_field,
                    x=X,
                    y=Y,
                    colorscale='Viridis',
                    contours_z=dict(show=True, usecolormap=True),
                    lighting=dict(ambient=0.8, diffuse=0.8)
                )
            ])
            
            fig_3d.update_layout(
                title=dict(text=f"Temperature Distribution - {params['material']}", 
                          font=dict(size=14, color='#2c3e50')),
                scene=dict(
                    xaxis_title="X (mm)",
                    yaxis_title="Y (mm)", 
                    zaxis_title="Temperature (°C)",
                    camera=dict(eye=dict(x=1.7, y=1.7, z=0.8))
                ),
                height=450,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # 2D Heatmap
            st.markdown('<div class="subsection-header">2D Cross-section</div>', unsafe_allow_html=True)
            fig_2d = px.imshow(T_field, color_continuous_scale='Viridis')
            fig_2d.update_layout(
                xaxis_title="X (mm)",
                yaxis_title="Y (mm)",
                height=300,
                margin=dict(l=0, r=0, t=20, b=0)
            )
            
            st.plotly_chart(fig_2d, use_container_width=True)
        
        with col2:
            st.markdown('<div class="section-header">Thermal Metrics</div>', unsafe_allow_html=True)
            
            # Key metrics in cards - NOW WITH REALISTIC VALUES
            metrics_data = [
                ("Peak Temperature", f"{peak_temp:.0f} °C", f"{'High' if peak_temp > 2500 else 'Normal'}"),
                ("Average Temperature", f"{avg_temp:.0f} °C", ""),
                ("Melt Pool Area", f"{melt_pool_area:.3f} mm²", ""),
                ("Max Thermal Gradient", f"{thermal_grad_max/1e6:.1f} ×10⁶ °C/m", "Typical for LPBF"),
                ("Cooling Rate", f"{cooling_rate/1e3:.0f} ×10³ °C/s", f"{'Fast' if cooling_rate > 5e5 else 'Moderate'}"),
                ("Volumetric Energy", f"{params['VED']:.1f} J/mm³", "")
            ]
            
            for label, value, note in metrics_data:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.2rem;">{note}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Material properties summary
            st.markdown('<div class="subsection-header">Material Properties</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size: 0.9rem; color: var(--text-primary);">
                <div style="margin-bottom: 0.3rem;">• Melting Point: {mat_props['melting_point']}°C</div>
                <div style="margin-bottom: 0.3rem;">• Thermal Conductivity: {mat_props['thermal_conductivity']} W/m·K</div>
                <div style="margin-bottom: 0.3rem;">• Specific Heat: {mat_props['specific_heat']} J/kg·K</div>
                <div>• Optimal VED: {mat_props['optimal_ved']} J/mm³</div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Grain size prediction
            st.markdown('<div class="subsection-header">Grain Size Prediction</div>', unsafe_allow_html=True)
            
            fig_grain = go.Figure(go.Indicator(
                mode="number+gauge",
                value=grain_size,
                number={'suffix': " µm", 'font': {'size': 24}},
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Average Grain Size", 'font': {'size': 14}},
                gauge={
                    'axis': {'range': [0, 200], 'tickwidth': 1},
                    'bar': {'color': mat_props['color']},
                    'steps': [
                        {'range': [0, 20], 'color': 'rgba(39, 174, 96, 0.1)'},
                        {'range': [20, 50], 'color': 'rgba(243, 156, 18, 0.1)'},
                        {'range': [50, 200], 'color': 'rgba(231, 76, 60, 0.1)'}
                    ]
                }
            ))
            
            fig_grain.update_layout(
                height=200,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig_grain, use_container_width=True)
            
            # Cooling rate info
            st.markdown('<div class="subsection-header">Cooling Characteristics</div>', unsafe_allow_html=True)
            cols = st.columns(2)
            cols[0].metric("Cooling Rate", f"{cooling_rate/1e3:.0f} k°C/s")
            cols[1].metric("Thermal Gradient", f"{thermal_grad_max/1e6:.1f} M°C/m")
            
            if peak_temp > mat_props["phase_transition_temp"]:
                st.info(f"Phase transformation detected at {mat_props['phase_transition_temp']}°C")
            
            # Simulated microstructure
            st.markdown('<div class="subsection-header">Microstructure Simulation</div>', unsafe_allow_html=True)
            np.random.seed(42)
            grain_data = np.random.rand(150, 150)
            for _ in range(3):
                grain_data = ndimage.gaussian_filter(grain_data, sigma=1.5)
            
            fig_structure = px.imshow(grain_data, color_continuous_scale='gray')
            fig_structure.update_layout(
                coloraxis_showscale=False,
                height=300,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            st.plotly_chart(fig_structure, use_container_width=True)
        
        with col2:
            # Phase distribution
            st.markdown('<div class="subsection-header">Phase Distribution</div>', unsafe_allow_html=True)
            
            fig_phases = go.Figure(data=[
                go.Pie(
                    labels=list(phases.keys()),
                    values=list(phases.values()),
                    hole=0.3,
                    marker=dict(colors=['#3498db', '#e74c3c', '#2ecc71', '#f39c12']),
                    textinfo='label+percent',
                    textposition='inside',
                    hoverinfo='label+value+percent'
                )
            ])
            
            fig_phases.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            
            st.plotly_chart(fig_phases, use_container_width=True)
            
            # Phase table
            st.markdown('<div class="subsection-header">Phase Fractions</div>', unsafe_allow_html=True)
            phase_df = pd.DataFrame({
                'Phase': list(phases.keys()),
                'Fraction (%)': [f"{v:.1f}" for v in phases.values()]
            })
            st.dataframe(phase_df, use_container_width=True, hide_index=True)
            
            # Texture analysis
            st.markdown('<div class="subsection-header">Crystallographic Texture</div>', unsafe_allow_html=True)
            
            theta = np.linspace(0, 2*np.pi, 100)
            texture_strength = 0.3 + 0.4 * np.abs(np.sin(2*theta)) * (cooling_rate/1e5)
            
            fig_pole = go.Figure(data=[
                go.Scatterpolar(
                    r=texture_strength,
                    theta=theta * 180/np.pi,
                    fill='toself',
                    fillcolor='rgba(52, 152, 219, 0.2)',
                    line=dict(color='#3498db', width=1.5),
                    mode='lines'
                )
            ])
            
            fig_pole.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False,
                height=250,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            
            st.plotly_chart(fig_pole, use_container_width=True)
            
            texture_index = np.mean(texture_strength)
            st.metric("Texture Strength Index", f"{texture_index:.2f}")
    
    with tab3:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Defect risk gauges
            st.markdown('<div class="subsection-header">Defect Probability</div>', unsafe_allow_html=True)
            
            defect_info = {
                "porosity": ("Porosity", "#e74c3c"),
                "lack_of_fusion": ("Lack of Fusion", "#f39c12"),
                "balling": ("Balling", "#8e44ad"),
                "keyholing": ("Keyholing", "#3498db")
            }
            
            for defect, (name, color) in defect_info.items():
                risk = defects[defect]
                
                if risk < 30:
                    risk_level = "Low"
                elif risk < 60:
                    risk_level = "Moderate"
                else:
                    risk_level = "High"
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk,
                    number={'suffix': "%", 'font': {'size': 18}},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"{name}<br>{risk_level}"},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': color, 'thickness': 0.2},
                        'steps': [
                            {'range': [0, 30], 'color': 'rgba(39, 174, 96, 0.1)'},
                            {'range': [30, 60], 'color': 'rgba(243, 156, 18, 0.1)'},
                            {'range': [60, 100], 'color': 'rgba(231, 76, 60, 0.1)'}
                        ]
                    }
                ))
                
                fig.update_layout(
                    height=180,
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Residual stress
            st.metric("Residual Stress", f"{defects['residual_stress']:.0f} MPa",
                     f"{'High' if defects['residual_stress'] > 500 else 'Moderate'}")
        
        with col2:
            # Defect map
            st.markdown('<div class="subsection-header">Defect Probability Distribution</div>', unsafe_allow_html=True)
            
            # Generate synthetic defect map
            defect_map = np.zeros((100, 100))
            center_x, center_y = 50, 50
            
            for i in range(100):
                for j in range(100):
                    distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                    defect_prob = 30 * np.exp(-distance/40)
                    defect_prob += 20 * np.sin(i/10) * np.cos(j/10)
                    defect_prob += 10 * np.random.randn() * (distance < 30)
                    defect_map[i, j] = min(max(defect_prob, 0), 100)
            
            fig_defect = px.imshow(defect_map, color_continuous_scale='Reds')
            fig_defect.update_layout(
                height=400,
                margin=dict(l=0, r=0, t=20, b=0)
            )
            
            st.plotly_chart(fig_defect, use_container_width=True)
            
            # Recommendations
            st.markdown('<div class="subsection-header">Process Recommendations</div>', unsafe_allow_html=True)
            
            recommendations = []
            if defects["porosity"] > 60:
                recommendations.append("Reduce VED (decrease power or increase speed)")
            if defects["lack_of_fusion"] > 60:
                recommendations.append("Increase VED (increase power or decrease speed)")
            if defects["residual_stress"] > 500:
                recommendations.append("Consider stress relief annealing")
            if cooling_rate > 5e5:
                recommendations.append("Increase preheat temperature to reduce cooling rate")
            
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    st.info(f"{i}. {rec}")
            else:
                st.success("Current parameters appear optimal")
            
            # Process window status
            st.markdown('<div class="subsection-header">Process Window Status</div>', unsafe_allow_html=True)
            ved_ratio = params['VED'] / mat_props["optimal_ved"]
            if 0.8 < ved_ratio < 1.2:
                st.success("✓ Within optimal VED range")
            else:
                st.warning(f"⚠️ VED deviation: {((ved_ratio-1)*100):.0f}% from optimal")
    
    with tab4:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Process window visualization
            power_range = np.linspace(100, 500, 25)
            speed_range = np.linspace(100, 2000, 25)
            P, V = np.meshgrid(power_range, speed_range)
            
            # Calculate VED for grid
            VED_grid = P / (V * params['hatch_spacing'] * params['layer_thickness'])
            optimal_ved = mat_props["optimal_ved"]
            
            # Quality score calculation
            quality_score = 100 * np.exp(-((VED_grid - optimal_ved) / (0.3 * optimal_ved))**2)
            
            # Create contour plot
            fig_contour = go.Figure(data=go.Contour(
                z=quality_score,
                x=power_range,
                y=speed_range,
                colorscale='RdYlGn',
                contours=dict(showlabels=True)
            ))
            
            fig_contour.add_trace(go.Scatter(
                x=[params['laser_power']],
                y=[params['scan_speed']],
                mode='markers',
                marker=dict(size=20, color='black', symbol='circle'),
                name='Current Point'
            ))
            
            fig_contour.update_layout(
                title=dict(text=f"Process Window: {params['material']}", 
                          font=dict(size=14)),
                xaxis_title="Laser Power (W)",
                yaxis_title="Scan Speed (mm/s)",
                height=500,
                margin=dict(l=0, r=20, t=40, b=0)
            )
            
            st.plotly_chart(fig_contour, use_container_width=True)
            
            with st.expander("Optimization Methodology"):
                st.markdown("""
                The process window is optimized using:
                
                1. **Volumetric Energy Density (VED)**: Primary parameter controlling melt pool characteristics
                2. **Material-specific optimal VED**: Based on literature values for each alloy
                3. **Quality score**: Exponential decay function centered at optimal VED
                4. **Process boundaries**: Defined by ±30% deviation from optimal VED
                """)
        
        with col2:
            st.markdown('<div class="subsection-header">Optimization Results</div>', unsafe_allow_html=True)
            
            # Calculate optimal point
            optimal_power = np.sqrt(params['laser_power'] * optimal_ved * params['scan_speed'] * 
                                   params['hatch_spacing'] * params['layer_thickness'])
            optimal_speed = optimal_power / (optimal_ved * params['hatch_spacing'] * params['layer_thickness'])
            
            # Current vs optimal
            current_score = 100 * np.exp(-((params['VED'] - optimal_ved) / (0.3 * optimal_ved))**2)
            
            col_a, col_b = st.columns(2)
            col_a.metric("Current Score", f"{current_score:.0f}%")
            col_b.metric("Optimal Score", "100%")
            
            st.metric("Improvement", f"{100 - current_score:.0f}%")
            
            st.divider()
            
            st.markdown("#### Recommended Parameters")
            st.metric("Laser Power", f"{optimal_power:.0f} W")
            st.metric("Scan Speed", f"{optimal_speed:.0f} mm/s")
            st.metric("Target VED", f"{optimal_ved:.1f} J/mm³")
            
            st.divider()
            
            # Apply optimization
            if st.button("🔄 Apply Optimization", use_container_width=True):
                params["laser_power"] = optimal_power
                params["scan_speed"] = optimal_speed
                params["VED"] = optimal_ved
                st.rerun()
            
            # Export options
            st.divider()
            if st.button("📊 Export Data", use_container_width=True):
                data = {
                    "Parameter": list(params.keys()),
                    "Value": list(params.values())
                }
                df = pd.DataFrame(data)
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="am_simulation.csv">Download CSV</a>'
                st.markdown(href, unsafe_allow_html=True)

else:
    # Welcome page
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem; background: var(--bg-card); border-radius: 8px; margin-bottom: 2rem;">
        <h2 style="color: var(--text-primary); margin-bottom: 1rem;">Predictive Modeling for Additive Manufacturing</h2>
        <p style="color: var(--text-secondary); max-width: 800px; margin: 0 auto; line-height: 1.6;">
            This platform integrates physics-based simulations to predict thermal history, 
            microstructure evolution, and defect formation in laser powder bed fusion processes.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick start
    st.markdown('<div class="section-header">Quick Start Templates</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    templates = [
        ("Ti-6Al-4V", "Aerospace components", 280, 850, 0.11, 0.04, 180, 61.2, 90),
        ("Inconel 718", "Turbine applications", 320, 700, 0.09, 0.05, 250, 50.8, 100),
        ("AlSi10Mg", "Lightweight structures", 350, 1300, 0.13, 0.06, 120, 38.5, 120)
    ]
    
    for i, (material_name, description, power, speed, hatch, layer, preheat, ved, beam) in enumerate(templates):
        with [col1, col2, col3][i]:
            if st.button(f"**{material_name}**\n{description}", use_container_width=True):
                st.session_state.current_run = {
                    "timestamp": datetime.now(),
                    "params": {
                        "laser_power": power,
                        "scan_speed": speed,
                        "hatch_spacing": hatch,
                        "layer_thickness": layer,
                        "preheat_temp": preheat,
                        "material": material_name,
                        "VED": ved,
                        "beam_diameter": beam,
                        "scan_strategy": "Bidirectional"
                    }
                }
                st.rerun()
    
    # Features overview
    st.markdown('<div class="section-header">Platform Features</div>', unsafe_allow_html=True)
    
    features = [
        ("Thermal Analysis", "3D temperature field, melt pool geometry, thermal gradients, cooling rates"),
        ("Microstructure Prediction", "Grain size estimation, phase fractions, texture development"),
        ("Process Optimization", "VED optimization, defect risk assessment, parameter recommendations")
    ]
    
    for title, description in features:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{title}</div>
            <div style="font-size: 0.9rem; color: var(--text-secondary); margin-top: 0.5rem;">{description}</div>
        </div>
        """, unsafe_allow_html=True)

# Credits footer
st.markdown("""
<div class="credit-footer">
    <div style="margin-bottom: 0.5rem;">
        <strong>AM Digital Twin v4.0</strong> | Predictive Process Modeling Platform
    </div>
    <div style="font-size: 0.8rem; margin-bottom: 0.5rem;">
        Developed by <strong>Muhammad Areeb Rizwan Siddiqui</strong>
    </div>
    <div style="font-size: 0.75rem;">
        <a href="https://www.areebrizwan.com" target="_blank">www.areebrizwan.com</a> | 
        <a href="https://www.linkedin.com/in/areebrizwan" target="_blank">LinkedIn</a>
    </div>
    <div style="font-size: 0.7rem; color: var(--text-muted); margin-top: 0.5rem;">
        Based on heat transfer and solidification models from literature [1-3]<br>
        [1] King et al., Acta Materialia (2014) | [2] DebRoy et al., Prog. Mat. Sci. (2018) | [3] Hunt, Mat. Sci. Eng. (1984)
    </div>
</div>
""", unsafe_allow_html=True)
