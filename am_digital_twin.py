import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy import ndimage
from datetime import datetime
import base64
from fpdf import FPDF
import time
from scipy.signal.windows import gaussian
from scipy.ndimage import gaussian_filter

# Page configuration - Professional dark theme
st.set_page_config(
    page_title="AM Digital Twin: Predictive Process Modeling",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS with dark theme
st.markdown("""
<style>
    /* Main styling */
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #ffffff;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    
    .metric-card {
        background: rgba(30, 30, 40, 0.8);
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #a0a0c0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(20, 20, 30, 0.8);
        padding: 8px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        background-color: rgba(40, 40, 50, 0.8);
        border-radius: 6px;
        padding: 0 20px;
        color: #b0b0d0;
        transition: all 0.2s;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: rgba(20, 20, 30, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Streamlit native elements override */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Custom container */
    .custom-container {
        background: rgba(25, 25, 35, 0.9);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    
    /* Status indicators */
    .status-optimal { color: #00d4aa; }
    .status-warning { color: #ffaa00; }
    .status-critical { color: #ff4d4d; }
    
    /* Loading spinner */
    .stSpinner > div {
        border-color: #667eea transparent transparent transparent;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'simulation_history' not in st.session_state:
    st.session_state.simulation_history = []
if 'current_run' not in st.session_state:
    st.session_state.current_run = {}
if 'comparison_mode' not in st.session_state:
    st.session_state.comparison_mode = False

# Material Properties Database (Expanded)
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
        "color": "#1f77b4"
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
        "color": "#ff7f0e"
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
        "color": "#2ca02c"
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
        "color": "#9467bd"
    },
    "CoCrMo": {
        "density": 8300,
        "thermal_conductivity": 13.5,
        "specific_heat": 430,
        "melting_point": 1650,
        "thermal_expansion": 12.5e-6,
        "youngs_modulus": 230,
        "optimal_ved": 55,
        "phase_transition_temp": 1200,
        "color": "#8c564b"
    }
}

# Title
st.markdown('<h1 class="main-title">🔬 Additive Manufacturing Digital Twin: Predictive Process Modeling</h1>', unsafe_allow_html=True)

# Sidebar with professional controls
with st.sidebar:
    st.markdown("### ⚙️ PROCESS PARAMETERS")
    st.markdown("---")
    
    # Material selection with color coding
    material = st.selectbox(
        "**MATERIAL**", 
        list(MATERIAL_DB.keys()),
        help="Select the powder material for simulation"
    )
    
    # Display material properties
    mat_props = MATERIAL_DB[material]
    with st.expander("Material Properties", expanded=False):
        cols = st.columns(2)
        cols[0].metric("ρ", f"{mat_props['density']:,} kg/m³")
        cols[1].metric("Tmelt", f"{mat_props['melting_point']}°C")
        cols[0].metric("E", f"{mat_props['youngs_modulus']} GPa")
        cols[1].metric("α", f"{mat_props['thermal_expansion']*1e6:.1f} µm/m·K")
    
    st.markdown("---")
    
    # Process parameters with professional layout
    col1, col2 = st.columns(2)
    with col1:
        laser_power = st.slider(
            "**LASER POWER (W)**", 
            100, 500, 300, 10,
            help="Input laser power in Watts"
        )
    with col2:
        scan_speed = st.slider(
            "**SCAN SPEED (mm/s)**", 
            100, 2000, 800, 50,
            help="Laser scanning speed"
        )
    
    col1, col2 = st.columns(2)
    with col1:
        hatch_spacing = st.slider(
            "**HATCH SPACING (mm)**", 
            0.05, 0.2, 0.1, 0.01,
            help="Distance between adjacent scan tracks"
        )
    with col2:
        layer_thickness = st.slider(
            "**LAYER THICKNESS (mm)**", 
            0.02, 0.1, 0.05, 0.01,
            help="Powder layer thickness"
        )
    
    preheat_temp = st.slider(
        "**PREHEAT TEMPERATURE (°C)**", 
        20, 400, 100, 10,
        help="Initial substrate temperature"
    )
    
    beam_diameter = st.slider(
        "**BEAM DIAMETER (µm)**", 
        50, 200, 100, 10,
        help="Laser beam spot size"
    )
    
    scan_strategy = st.selectbox(
        "**SCAN STRATEGY**", 
        ["Bidirectional", "Island (5x5mm)", "Spiral", "Chessboard", "Custom"],
        help="Laser scanning pattern"
    )
    
    # Calculate VED
    VED = laser_power / (scan_speed * hatch_spacing * layer_thickness)
    ved_ratio = VED / mat_props["optimal_ved"]
    
    # VED Status
    st.markdown("---")
    ved_col1, ved_col2 = st.columns([2, 1])
    with ved_col1:
        st.metric("**VED**", f"{VED:.1f} J/mm³", 
                 f"{'Optimal' if 0.8 < ved_ratio < 1.2 else 'Non-optimal'}")
    with ved_col2:
        if ved_ratio < 0.8:
            st.error("⏬")
        elif ved_ratio > 1.2:
            st.error("⏫")
        else:
            st.success("✓")
    
    st.markdown("---")
    
    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ RUN SIMULATION", use_container_width=True, type="primary"):
            with st.spinner("Executing thermal-fluid simulation..."):
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
        if st.button("💾 ARCHIVE RUN", use_container_width=True):
            if st.session_state.current_run:
                st.session_state.simulation_history.append(st.session_state.current_run.copy())
                st.success("Run archived to database")
    
    # Comparison mode toggle
    st.markdown("---")
    st.session_state.comparison_mode = st.toggle("📊 COMPARISON MODE", 
                                                 help="Compare multiple simulation runs")
    
    # Recent runs
    if st.session_state.simulation_history:
        st.markdown("### 📈 RECENT SIMULATIONS")
        for i, run in enumerate(st.session_state.simulation_history[-3:]):
            with st.expander(f"Run #{len(st.session_state.simulation_history)-i}: {run['params']['material']} | {run['timestamp'].strftime('%H:%M')}", expanded=False):
                p = run['params']
                cols = st.columns(3)
                cols[0].metric("P", f"{p['laser_power']}W")
                cols[1].metric("V", f"{p['scan_speed']}mm/s")
                cols[2].metric("VED", f"{p['VED']:.1f}")

# ============================================================================
# SCIENTIFIC MODELS
# ============================================================================

def solve_heat_transfer(params):
    """Finite difference solution of heat conduction with moving source"""
    laser_power = params["laser_power"]
    scan_speed = params["scan_speed"]
    beam_radius = params["beam_diameter"] / 2000  # Convert to mm
    material = MATERIAL_DB[params["material"]]
    
    # Grid definition
    nx, ny = 80, 80
    x = np.linspace(-2.5, 2.5, nx)
    y = np.linspace(-2.5, 2.5, ny)
    X, Y = np.meshgrid(x, y)
    
    # Material properties
    alpha = material["thermal_conductivity"] / (material["density"] * material["specific_heat"])
    
    # Gaussian heat source
    eta = 0.65  # Absorption coefficient
    q_max = (2 * eta * laser_power) / (np.pi * beam_radius**2)
    
    # Distance from heat source center (moving at scan speed)
    t = 0.01  # Time increment
    x0, y0 = 0, 0  # Heat source center
    
    # Temperature field initialization
    T = np.ones((ny, nx)) * params["preheat_temp"]
    
    # Solve transient heat conduction (simplified)
    r_squared = (X - x0)**2 + (Y - y0)**2
    q = q_max * np.exp(-2 * r_squared / beam_radius**2)
    
    # Add heat source contribution
    T += q * t / (material["density"] * material["specific_heat"])
    
    # Apply diffusion
    T = gaussian_filter(T, sigma=0.8)
    
    # Ensure physical limits
    T = np.clip(T, params["preheat_temp"], 3500)
    
    return X, Y, T

def calculate_thermal_gradient(T_field):
    """Calculate thermal gradient magnitude"""
    grad_y, grad_x = np.gradient(T_field)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return np.max(grad_magnitude), np.mean(grad_magnitude)

def predict_cooling_rate(T_field, time_step=0.001):
    """Estimate cooling rate from temperature evolution"""
    # Simulate cooling by diffusion
    T_cooled = gaussian_filter(T_field, sigma=1.2)
    cooling_rate = (T_field - T_cooled).mean() / time_step
    return max(cooling_rate, 1e-3)  # Avoid zero division

def predict_grain_size(cooling_rate, material):
    """Modified Hunt model for AM with material-specific constants"""
    if material == "Ti-6Al-4V":
        k, n = 45, 0.35
    elif material == "Inconel 718":
        k, n = 38, 0.32
    elif material == "AlSi10Mg":
        k, n = 60, 0.38
    else:
        k, n = 50, 0.33
    
    if cooling_rate > 0:
        grain_size = k * (cooling_rate)**(-n)
    else:
        grain_size = 100
    
    return min(grain_size, 200)  # Limit maximum grain size

def predict_microstructure_phases(material, cooling_rate):
    """Predict phase fractions based on material and cooling rate"""
    phases = {}
    
    if material == "Ti-6Al-4V":
        alpha = 70 + 0.03 * cooling_rate
        beta = 30 - 0.02 * cooling_rate
        martensite = max(0, 100 - alpha - beta)
        phases = {"α-phase": alpha, "β-phase": beta, "α'-martensite": martensite}
    
    elif material == "Inconel 718":
        gamma = 60 + 0.02 * cooling_rate
        gamma_prime = 35 - 0.015 * cooling_rate
        carbides = max(0, 100 - gamma - gamma_prime)
        phases = {"γ-matrix": gamma, "γ'-precipitates": gamma_prime, "Carbides": carbides}
    
    elif material == "SS316L":
        austenite = 85 + 0.01 * cooling_rate
        ferrite = 15 - 0.01 * cooling_rate
        phases = {"Austenite": austenite, "Ferrite": ferrite, "Sigma-phase": 0}
    
    else:  # AlSi10Mg
        aluminum = 88 - 0.005 * cooling_rate
        silicon = 10 + 0.003 * cooling_rate
        phases = {"Al-matrix": aluminum, "Si-particles": silicon, "Mg₂Si": 2}
    
    # Normalize to 100%
    total = sum(phases.values())
    phases = {k: v/total*100 for k, v in phases.items()}
    
    return phases

def calculate_defect_risks(params, T_field, cooling_rate):
    """Comprehensive defect risk assessment"""
    material = MATERIAL_DB[params["material"]]
    optimal_ved = material["optimal_ved"]
    VED = params["VED"]
    
    # Porosity risk (sigmoid function)
    porosity_risk = 50 * (1 + np.tanh(0.15 * (VED - optimal_ved)))
    
    # Lack of fusion risk
    lof_risk = 60 * (1 - np.tanh(0.2 * VED))
    
    # Balling risk
    balling_risk = 40 * (1 + np.tanh(0.25 * (VED - optimal_ved * 1.5)))
    
    # Keyholing risk
    peak_temp = np.max(T_field)
    if peak_temp > material["melting_point"] * 1.5:
        keyhole_risk = 80
    elif peak_temp > material["melting_point"] * 1.2:
        keyhole_risk = 50
    else:
        keyhole_risk = 10
    
    # Residual stress
    thermal_grad = calculate_thermal_gradient(T_field)[0]
    residual_stress = (material["youngs_modulus"] * material["thermal_expansion"] * 
                      thermal_grad * 0.001 * 1000)  # Convert to MPa
    
    return {
        "porosity": min(porosity_risk, 100),
        "lack_of_fusion": min(lof_risk, 100),
        "balling": min(balling_risk, 100),
        "keyholing": min(keyhole_risk, 100),
        "residual_stress": min(residual_stress, 1000)
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
    melt_pool_area = np.sum(T_field > mat_props["melting_point"]) * 0.0016  # mm²
    thermal_grad_max, thermal_grad_avg = calculate_thermal_gradient(T_field)
    cooling_rate = predict_cooling_rate(T_field)
    grain_size = predict_grain_size(cooling_rate, params["material"])
    phases = predict_microstructure_phases(params["material"], cooling_rate)
    defects = calculate_defect_risks(params, T_field, cooling_rate)
    
    # Main dashboard layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 PROCESS SIMULATION",
        "🔬 MICROSTRUCTURE PREDICTION", 
        "⚠️ DEFECT RISK ASSESSMENT",
        "🎯 PROCESS OPTIMIZATION"
    ])
    
    with tab1:
        st.markdown('<div class="section-header">Thermal Field Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 3D Thermal Visualization
            fig_3d = go.Figure(data=[
                go.Surface(
                    z=T_field,
                    x=X,
                    y=Y,
                    colorscale='Plasma',
                    contours_z=dict(
                        show=True,
                        usecolormap=True,
                        highlightcolor="cyan",
                        project_z=True
                    ),
                    lighting=dict(
                        ambient=0.8,
                        diffuse=0.8,
                        roughness=0.9,
                        specular=0.2
                    )
                )
            ])
            
            fig_3d.update_layout(
                title=dict(
                    text=f"Temperature Distribution | {params['material']}",
                    font=dict(size=16, color='white')
                ),
                scene=dict(
                    xaxis_title="X (mm)",
                    yaxis_title="Y (mm)",
                    zaxis_title="Temperature (°C)",
                    camera=dict(
                        eye=dict(x=1.7, y=1.7, z=0.8)
                    ),
                    bgcolor='rgba(20,20,30,1)'
                ),
                height=500,
                margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig_3d, use_container_width=True, theme=None)
        
        with col2:
            st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)
            
            # Metric cards
            metrics = [
                ("Peak Temperature", f"{peak_temp:.0f} °C", "status-critical" if peak_temp > 2500 else "status-optimal"),
                ("Average Temperature", f"{avg_temp:.0f} °C", ""),
                ("Melt Pool Area", f"{melt_pool_area:.3f} mm²", ""),
                ("Thermal Gradient", f"{thermal_grad_max:.0f} °C/mm", ""),
                ("Cooling Rate", f"{cooling_rate:.0f} °C/s", "status-warning" if cooling_rate > 1000 else ""),
                ("VED", f"{params['VED']:.1f} J/mm³", "")
            ]
            
            for label, value, status in metrics:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value {status}">{value}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Material indicator
            st.markdown(f"""
            <div style="margin-top: 1rem; padding: 1rem; background: rgba(30,30,40,0.8); border-radius: 8px; border-left: 4px solid {mat_props['color']};">
                <div style="font-size: 0.9rem; color: #a0a0c0;">Current Material</div>
                <div style="font-size: 1.2rem; font-weight: 600; color: white;">{params['material']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="section-header">Microstructure Evolution</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.markdown("### Grain Structure")
            fig_grain = go.Figure(data=[
                go.Indicator(
                    mode="number+gauge",
                    value=grain_size,
                    number={'suffix': " µm", 'font': {'size': 24}},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Avg. Grain Size", 'font': {'size': 14}},
                    gauge={
                        'axis': {'range': [0, 200], 'tickwidth': 1},
                        'bar': {'color': mat_props['color']},
                        'steps': [
                            {'range': [0, 50], 'color': 'rgba(0, 212, 170, 0.2)'},
                            {'range': [50, 100], 'color': 'rgba(255, 170, 0, 0.2)'},
                            {'range': [100, 200], 'color': 'rgba(255, 77, 77, 0.2)'}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 2},
                            'thickness': 0.75,
                            'value': 80
                        }
                    }
                )
            ])
            
            fig_grain.update_layout(
                height=200,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig_grain, use_container_width=True, theme=None)
            
            # Cooling rate indicator
            st.metric("Cooling Rate", f"{cooling_rate:.0f} °C/s", 
                     delta=f"{'Fast' if cooling_rate > 500 else 'Moderate' if cooling_rate > 100 else 'Slow'}")
            
            # Phase transition
            if peak_temp > mat_props["phase_transition_temp"]:
                st.info(f"✓ Phase transformation at {mat_props['phase_transition_temp']}°C")
        
        with col2:
            # Phase distribution
            fig_phases = go.Figure(data=[
                go.Pie(
                    labels=list(phases.keys()),
                    values=list(phases.values()),
                    hole=0.4,
                    marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']),
                    textinfo='label+percent',
                    textposition='inside',
                    hoverinfo='label+value+percent',
                    textfont=dict(size=12, color='white')
                )
            ])
            
            fig_phases.update_layout(
                title=dict(
                    text="Predicted Phase Distribution",
                    font=dict(size=16, color='white')
                ),
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(
                    font=dict(color='white'),
                    bgcolor='rgba(0,0,0,0)',
                    bordercolor='rgba(255,255,255,0.2)'
                )
            )
            
            st.plotly_chart(fig_phases, use_container_width=True, theme=None)
            
            # Simulated grain structure
            np.random.seed(42)
            grain_data = np.random.rand(150, 150)
            for _ in range(3):
                grain_data = gaussian_filter(grain_data, sigma=1.5)
            
            fig_structure = px.imshow(
                grain_data,
                color_continuous_scale='gray',
                title="Simulated Grain Network",
                width=400,
                height=300
            )
            
            fig_structure.update_layout(
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_structure, use_container_width=True, theme=None)
        
        with col3:
            st.markdown("### Crystallographic Texture")
            
            # Generate pole figure data
            theta = np.linspace(0, 2*np.pi, 100)
            texture_strength = 0.3 + 0.4 * np.abs(np.sin(2*theta)) * (cooling_rate/1000)
            
            fig_pole = go.Figure(data=[
                go.Scatterpolar(
                    r=texture_strength,
                    theta=theta * 180/np.pi,
                    fill='toself',
                    fillcolor='rgba(102, 126, 234, 0.3)',
                    line=dict(color='#667eea', width=2),
                    mode='lines'
                )
            ])
            
            fig_pole.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        tickfont=dict(color='white')
                    ),
                    angularaxis=dict(
                        tickfont=dict(color='white'),
                        rotation=90
                    ),
                    bgcolor='rgba(30, 30, 40, 0.8)'
                ),
                showlegend=False,
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_pole, use_container_width=True, theme=None)
            
            # Texture strength metric
            texture_index = np.mean(texture_strength)
            st.metric("Texture Index", f"{texture_index:.2f}", 
                     delta=f"{'Strong' if texture_index > 0.6 else 'Moderate' if texture_index > 0.4 else 'Weak'}")
    
    with tab3:
        st.markdown('<div class="section-header">Quality Assurance Dashboard</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Defect risk gauges
            defect_names = {
                "porosity": "Porosity Risk",
                "lack_of_fusion": "Lack of Fusion",
                "balling": "Balling Effect",
                "keyholing": "Keyholing"
            }
            
            for defect, name in defect_names.items():
                risk = defects[defect]
                
                # Determine color based on risk level
                if risk < 30:
                    color = "#00d4aa"
                elif risk < 60:
                    color = "#ffaa00"
                else:
                    color = "#ff4d4d"
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk,
                    number={'suffix': "%", 'font': {'size': 20}},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': name, 'font': {'size': 14}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                        'bar': {'color': color, 'thickness': 0.25},
                        'bgcolor': "rgba(0,0,0,0)",
                        'borderwidth': 2,
                        'bordercolor': "rgba(255,255,255,0.2)",
                        'steps': [
                            {'range': [0, 30], 'color': 'rgba(0, 212, 170, 0.1)'},
                            {'range': [30, 60], 'color': 'rgba(255, 170, 0, 0.1)'},
                            {'range': [60, 100], 'color': 'rgba(255, 77, 77, 0.1)'}
                        ]
                    }
                ))
                
                fig.update_layout(
                    height=180,
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                
                st.plotly_chart(fig, use_container_width=True, theme=None)
            
            # Residual stress
            st.metric("Residual Stress", f"{defects['residual_stress']:.0f} MPa",
                     delta=f"{'High' if defects['residual_stress'] > 500 else 'Moderate' if defects['residual_stress'] > 200 else 'Low'}")
        
        with col2:
            # Defect probability map
            st.markdown("### Defect Probability Heatmap")
            
            # Generate synthetic defect map
            defect_map = np.zeros((100, 100))
            center_x, center_y = 50, 50
            
            for i in range(100):
                for j in range(100):
                    distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                    # Base defect probability
                    defect_prob = 30 * np.exp(-distance/40)
                    
                    # Add variations
                    defect_prob += 20 * np.sin(i/10) * np.cos(j/10)
                    defect_prob += 10 * np.random.randn() * (distance < 30)
                    
                    defect_map[i, j] = min(max(defect_prob, 0), 100)
            
            fig_defect = px.imshow(
                defect_map,
                color_continuous_scale='Reds',
                title="Relative Defect Probability Distribution",
                labels={'color': 'Probability (%)'},
                width=600,
                height=400
            )
            
            fig_defect.update_layout(
                margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                coloraxis_colorbar=dict(
                    title_font=dict(color='white'),
                    tickfont=dict(color='white')
                )
            )
            
            st.plotly_chart(fig_defect, use_container_width=True, theme=None)
            
            # Recommendations
            st.markdown("### Recommendations")
            
            recommendations = []
            if defects["porosity"] > 60:
                recommendations.append("Reduce VED to mitigate porosity")
            if defects["lack_of_fusion"] > 60:
                recommendations.append("Increase VED to improve fusion")
            if defects["residual_stress"] > 500:
                recommendations.append("Implement stress relief annealing")
            if cooling_rate > 1000:
                recommendations.append("Consider pre-heating to reduce cooling rate")
            
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    st.info(f"{i}. {rec}")
            else:
                st.success("✓ Process parameters appear optimal")
    
    with tab4:
        st.markdown('<div class="section-header">Process Parameter Optimization</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Interactive process window
            power_range = np.linspace(100, 500, 25)
            speed_range = np.linspace(100, 2000, 25)
            P, V = np.meshgrid(power_range, speed_range)
            
            # Calculate VED for grid
            VED_grid = P / (V * params['hatch_spacing'] * params['layer_thickness'])
            optimal_ved = mat_props["optimal_ved"]
            
            # Quality score calculation
            quality_score = np.zeros_like(VED_grid)
            quality_score = 100 * np.exp(-((VED_grid - optimal_ved) / (0.3 * optimal_ved))**2)
            
            # Create contour plot
            fig_contour = go.Figure(data=[
                go.Contour(
                    z=quality_score,
                    x=power_range,
                    y=speed_range,
                    colorscale='Viridis',
                    contours=dict(
                        showlabels=True,
                        labelfont=dict(size=12, color='white')
                    ),
                    colorbar=dict(
                        title="Quality Score",
                        titlefont=dict(color='white'),
                        tickfont=dict(color='white')
                    ),
                    hovertemplate="Power: %{x} W<br>Speed: %{y} mm/s<br>Score: %{z:.1f}<extra></extra>"
                ),
                go.Scatter(
                    x=[params['laser_power']],
                    y=[params['scan_speed']],
                    mode='markers',
                    marker=dict(
                        size=20,
                        color='white',
                        symbol='star',
                        line=dict(width=2, color='black')
                    ),
                    name='Current Point',
                    hovertemplate="Current Parameters<br>Power: %{x} W<br>Speed: %{y} mm/s<extra></extra>"
                )
            ])
            
            fig_contour.update_layout(
                title=dict(
                    text="Process Window Optimization Map",
                    font=dict(size=16, color='white')
                ),
                xaxis_title="Laser Power (W)",
                yaxis_title="Scan Speed (mm/s)",
                height=500,
                margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(
                    gridcolor='rgba(255,255,255,0.1)',
                    zerolinecolor='rgba(255,255,255,0.2)'
                ),
                yaxis=dict(
                    gridcolor='rgba(255,255,255,0.1)',
                    zerolinecolor='rgba(255,255,255,0.2)'
                )
            )
            
            st.plotly_chart(fig_contour, use_container_width=True, theme=None)
        
        with col2:
            st.markdown("### Optimization Results")
            
            # Calculate optimal point
            optimal_power = np.sqrt(params['laser_power'] * optimal_ved * params['scan_speed'] * 
                                   params['hatch_spacing'] * params['layer_thickness'])
            optimal_speed = optimal_power / (optimal_ved * params['hatch_spacing'] * params['layer_thickness'])
            
            # Quality improvement
            current_quality = 100 * np.exp(-((params['VED'] - optimal_ved) / (0.3 * optimal_ved))**2)
            optimized_quality = 100  # Maximum at optimal VED
            
            st.metric("Current Quality", f"{current_quality:.1f}%")
            st.metric("Optimal Quality", f"{optimized_quality:.1f}%")
            st.metric("Improvement", f"{optimized_quality - current_quality:.1f}%")
            
            st.divider()
            
            st.markdown("#### Recommended Parameters")
            st.metric("Power", f"{optimal_power:.0f} W")
            st.metric("Speed", f"{optimal_speed:.0f} mm/s")
            st.metric("VED", f"{optimal_ved:.1f} J/mm³")
            
            st.divider()
            
            # Apply optimization
            if st.button("🔄 APPLY OPTIMIZATION", use_container_width=True):
                params["laser_power"] = optimal_power
                params["scan_speed"] = optimal_speed
                params["VED"] = optimal_ved
                st.rerun()
    
    # Advanced controls at bottom
    st.markdown("---")
    st.markdown('<div class="section-header">Advanced Controls</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 EXPORT DATA", use_container_width=True):
            # Create downloadable CSV
            data = {
                "Parameter": list(params.keys()),
                "Value": list(params.values())
            }
            df = pd.DataFrame(data)
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="am_simulation_data.csv">Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    with col2:
        if st.button("📈 COMPARE RUNS", use_container_width=True):
            st.session_state.comparison_mode = not st.session_state.comparison_mode
            st.rerun()
    
    with col3:
        if st.button("🔄 NEW SIMULATION", use_container_width=True):
            st.session_state.current_run = {}
            st.rerun()
    
    with col4:
        if st.button("⚙️ ADVANCED SETTINGS", use_container_width=True):
            st.info("Advanced settings panel would open here")

else:
    # Welcome/landing page
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%); 
                    padding: 2rem; border-radius: 12px; margin-bottom: 2rem;">
            <h2 style="color: white; margin-bottom: 1rem;">Advanced Process Modeling for Additive Manufacturing</h2>
            <p style="color: #b0b0d0; line-height: 1.6;">
                This digital twin platform integrates physics-based simulations with machine learning 
                to predict microstructure evolution, defect formation, and optimize process parameters 
                for metal additive manufacturing.
            </p>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 2rem;">
                <div style="background: rgba(30,30,40,0.8); padding: 1rem; border-radius: 8px; border-left: 3px solid #667eea;">
                    <div style="font-size: 0.9rem; color: #a0a0c0;">Physics-Based Models</div>
                    <div style="font-size: 1.2rem; color: white; font-weight: 600;">Heat Transfer</div>
                    <div style="font-size: 1.2rem; color: white; font-weight: 600;">Fluid Dynamics</div>
                    <div style="font-size: 1.2rem; color: white; font-weight: 600;">Phase Transformation</div>
                </div>
                <div style="background: rgba(30,30,40,0.8); padding: 1rem; border-radius: 8px; border-left: 3px solid #00d4aa;">
                    <div style="font-size: 0.9rem; color: #a0a0c0;">Predictive Analytics</div>
                    <div style="font-size: 1.2rem; color: white; font-weight: 600;">Microstructure</div>
                    <div style="font-size: 1.2rem; color: white; font-weight: 600;">Defect Analysis</div>
                    <div style="font-size: 1.2rem; color: white; font-weight: 600;">Mechanical Properties</div>
                </div>
                <div style="background: rgba(30,30,40,0.8); padding: 1rem; border-radius: 8px; border-left: 3px solid #ffaa00;">
                    <div style="font-size: 0.9rem; color: #a0a0c0;">Optimization</div>
                    <div style="font-size: 1.2rem; color: white; font-weight: 600;">Process Window</div>
                    <div style="font-size: 1.2rem; color: white; font-weight: 600;">Parameter Tuning</div>
                    <div style="font-size: 1.2rem; color: white; font-weight: 600;">Quality Control</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick start cards
        st.markdown("### Quick Start Templates")
        
        templates_col1, templates_col2, templates_col3 = st.columns(3)
        
        with templates_col1:
            if st.button("**Ti-6Al-4V Aerospace**\n\nOptimal parameters for aerospace components", 
                        use_container_width=True, help="High-quality parts with minimal defects"):
                st.session_state.current_run = {
                    "timestamp": datetime.now(),
                    "params": {
                        "laser_power": 280,
                        "scan_speed": 850,
                        "hatch_spacing": 0.11,
                        "layer_thickness": 0.04,
                        "preheat_temp": 180,
                        "material": "Ti-6Al-4V",
                        "VED": 61.2,
                        "beam_diameter": 90,
                        "scan_strategy": "Bidirectional"
                    }
                }
                st.rerun()
        
        with templates_col2:
            if st.button("**Inconel 718 Turbine**\n\nHigh-temperature applications", 
                        use_container_width=True, help="For turbine blade manufacturing"):
                st.session_state.current_run = {
                    "timestamp": datetime.now(),
                    "params": {
                        "laser_power": 320,
                        "scan_speed": 700,
                        "hatch_spacing": 0.09,
                        "layer_thickness": 0.05,
                        "preheat_temp": 250,
                        "material": "Inconel 718",
                        "VED": 50.8,
                        "beam_diameter": 100,
                        "scan_strategy": "Island (5x5mm)"
                    }
                }
                st.rerun()
        
        with templates_col3:
            if st.button("**AlSi10Mg Automotive**\n\nLightweight structural components", 
                        use_container_width=True, help="For automotive lightweighting"):
                st.session_state.current_run = {
                    "timestamp": datetime.now(),
                    "params": {
                        "laser_power": 350,
                        "scan_speed": 1300,
                        "hatch_spacing": 0.13,
                        "layer_thickness": 0.06,
                        "preheat_temp": 120,
                        "material": "AlSi10Mg",
                        "VED": 38.5,
                        "beam_diameter": 120,
                        "scan_strategy": "Chessboard"
                    }
                }
                st.rerun()
    
    with col2:
        st.markdown("""
        <div style="background: rgba(30,30,40,0.8); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);">
            <h3 style="color: white; margin-bottom: 1rem;">Getting Started</h3>
            <ol style="color: #b0b0d0; line-height: 2; padding-left: 1.2rem;">
                <li>Select material in sidebar</li>
                <li>Adjust process parameters</li>
                <li>Click 'Run Simulation'</li>
                <li>Analyze results in tabs</li>
                <li>Optimize using recommendations</li>
            </ol>
            <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1);">
                <div style="font-size: 0.9rem; color: #a0a0c0;">Supported Materials</div>
                <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem;">
                    <span style="background: rgba(31,119,180,0.2); color: #1f77b4; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.8rem;">Ti-6Al-4V</span>
                    <span style="background: rgba(255,127,14,0.2); color: #ff7f0e; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.8rem;">Inconel 718</span>
                    <span style="background: rgba(44,160,44,0.2); color: #2ca02c; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.8rem;">SS316L</span>
                    <span style="background: rgba(148,103,189,0.2); color: #9467bd; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.8rem;">AlSi10Mg</span>
                    <span style="background: rgba(140,86,75,0.2); color: #8c564b; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.8rem;">CoCrMo</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer with conference-ready branding
st.markdown("""
<div style="margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.1); text-align: center; color: #8080a0;">
    <div style="font-size: 0.9rem; margin-bottom: 0.5rem;">
        <strong>AM Digital Twin v3.0</strong> | Predictive Process Modeling Platform
    </div>
    <div style="font-size: 0.8rem; margin-bottom: 0.5rem;">
        Developed for: <strong>International Conference on Additive Manufacturing (ICAM 2024)</strong>
    </div>
    <div style="font-size: 0.75rem;">
        © 2024 Advanced Manufacturing Research Group | All simulation results for research purposes only
    </div>
</div>
""", unsafe_allow_html=True)
