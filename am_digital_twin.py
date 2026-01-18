import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import altair as alt
from scipy import ndimage
import io
from datetime import datetime
import base64
from fpdf import FPDF
import time

# Page configuration
st.set_page_config(
    page_title="AM Digital Twin",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #764ba2;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏭 Digital Twin for Additive Manufacturing</h1>', unsafe_allow_html=True)

# Initialize session state for storing simulation runs
if 'simulation_history' not in st.session_state:
    st.session_state.simulation_history = []
if 'current_run' not in st.session_state:
    st.session_state.current_run = {}

# Material database
MATERIAL_DB = {
    "Ti-6Al-4V": {
        "density": 4430,
        "thermal_conductivity": 7.2,
        "specific_heat": 526,
        "melting_point": 1923,
        "thermal_expansion": 8.6e-6,
        "youngs_modulus": 113.8,
        "optimal_ved": 60
    },
    "Inconel 718": {
        "density": 8190,
        "thermal_conductivity": 11.4,
        "specific_heat": 435,
        "melting_point": 1600,
        "thermal_expansion": 13.0e-6,
        "youngs_modulus": 200,
        "optimal_ved": 50
    },
    "SS316L": {
        "density": 7990,
        "thermal_conductivity": 16.2,
        "specific_heat": 500,
        "melting_point": 1670,
        "thermal_expansion": 16.0e-6,
        "youngs_modulus": 193,
        "optimal_ved": 70
    },
    "AlSi10Mg": {
        "density": 2670,
        "thermal_conductivity": 120,
        "specific_heat": 880,
        "melting_point": 860,
        "thermal_expansion": 21.5e-6,
        "youngs_modulus": 70,
        "optimal_ved": 40
    }
}

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Process Parameters")
    
    # Parameter sliders
    laser_power = st.slider("Laser Power (W)", 100, 500, 300, step=10)
    scan_speed = st.slider("Scan Speed (mm/s)", 100, 2000, 800, step=50)
    hatch_spacing = st.slider("Hatch Spacing (mm)", 0.05, 0.2, 0.1, step=0.01)
    layer_thickness = st.slider("Layer Thickness (mm)", 0.02, 0.1, 0.05, step=0.01)
    preheat_temp = st.slider("Pre-heat Temperature (°C)", 20, 400, 100, step=10)
    
    material = st.selectbox("Material", list(MATERIAL_DB.keys()))
    beam_diameter = st.slider("Beam Diameter (µm)", 50, 200, 100, step=10)
    scan_strategy = st.selectbox("Scan Strategy", ["Bidirectional", "Island", "Spiral", "Custom"])
    
    # Calculate Volumetric Energy Density
    VED = laser_power / (scan_speed * hatch_spacing * layer_thickness)
    
    col1, col2 = st.columns(2)
    with col1:
        run_button = st.button("🚀 Run Simulation", use_container_width=True)
        if run_button:
            with st.spinner("Simulating..."):
                time.sleep(1)  # Simulate computation time
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
        save_button = st.button("💾 Save Run", use_container_width=True)
        if save_button:
            if st.session_state.current_run:
                st.session_state.simulation_history.append(st.session_state.current_run.copy())
                st.success("Run saved!")
    
    st.divider()
    
    # Display current VED
    material_props = MATERIAL_DB[material]
    optimal_ved = material_props["optimal_ved"]
    ved_ratio = VED / optimal_ved
    
    st.metric("Volumetric Energy Density", f"{VED:.1f} J/mm³")
    
    # VED indicator
    if ved_ratio < 0.8:
        st.warning("⚠️ Low VED - Risk of lack of fusion")
    elif ved_ratio > 1.2:
        st.error("⚠️ High VED - Risk of keyholing/balling")
    else:
        st.success("✅ VED within optimal range")
    
    st.divider()
    
    # Simulation history
    if st.session_state.simulation_history:
        st.subheader("📊 Simulation History")
        for i, run in enumerate(st.session_state.simulation_history[-5:]):
            with st.expander(f"Run {i+1}: {run['timestamp'].strftime('%H:%M:%S')}"):
                for key, value in run["params"].items():
                    if isinstance(value, (int, float)):
                        st.text(f"{key}: {value:.2f}")
                    else:
                        st.text(f"{key}: {value}")

# Scientific Models
def heat_transfer_model(params):
    """Simulate temperature distribution using heat transfer equation"""
    # Parameters
    laser_power = params["laser_power"]
    scan_speed = params["scan_speed"]
    beam_radius = params.get("beam_diameter", 100) / 2000  # Convert to mm
    material = MATERIAL_DB[params["material"]]
    
    # Create grid
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    
    # Gaussian heat source
    eta = 0.7  # Absorption efficiency
    Q = eta * laser_power / (np.pi * beam_radius**2 * scan_speed)
    T_dist = Q * np.exp(-(X**2 + Y**2) / (2 * beam_radius**2))
    
    # Add pre-heat
    T_dist += params["preheat_temp"]
    
    # Apply material properties scaling
    T_dist *= material["specific_heat"] / 500  # Normalize
    
    return X, Y, T_dist

def calculate_cooling_rate(T_dist):
    """Estimate cooling rate from temperature distribution"""
    # Simplified cooling rate estimation
    grad_x, grad_y = np.gradient(T_dist)
    cooling_rate = np.sqrt(grad_x**2 + grad_y**2).mean() * 100
    return cooling_rate

def calculate_thermal_gradient(T_dist):
    """Calculate maximum thermal gradient"""
    grad_x, grad_y = np.gradient(T_dist)
    thermal_gradient = np.sqrt(grad_x**2 + grad_y**2).max()
    return thermal_gradient

def predict_grain_size(cooling_rate):
    """Hunt model for grain size prediction"""
    if cooling_rate > 0:
        grain_size = 50 * (cooling_rate)**(-0.33)
    else:
        grain_size = 100  # Default large grain size
    return grain_size

def predict_porosity(params):
    """Porosity risk prediction based on VED"""
    VED = params["VED"]
    optimal_ved = MATERIAL_DB[params["material"]]["optimal_ved"]
    
    # Sigmoid function for porosity risk
    a = 0.1
    porosity_risk = 100 / (1 + np.exp(-a * (VED - optimal_ved)))
    
    # Adjust based on parameters
    if params["hatch_spacing"] > 0.15:
        porosity_risk *= 1.2
    if params["layer_thickness"] > 0.08:
        porosity_risk *= 1.1
    
    return min(porosity_risk, 100)

def predict_residual_stress(params):
    """Simplified residual stress model"""
    material = MATERIAL_DB[params["material"]]
    E = material["youngs_modulus"]
    alpha = material["thermal_expansion"]
    
    # Base stress calculation
    base_stress = E * alpha * (material["melting_point"] - params["preheat_temp"])
    
    # Adjust for parameters
    speed_factor = 1.0
    if params["scan_speed"] > 1500:
        speed_factor = 1.3
    elif params["scan_speed"] < 300:
        speed_factor = 0.8
    
    power_factor = params["laser_power"] / 300
    
    residual_stress = base_stress * speed_factor * power_factor * 0.01  # MPa
    
    return min(residual_stress, 800)  # Cap at 800 MPa

def simulate_microstructure(grain_size, material_name):
    """Generate simulated microstructure data"""
    phases = {}
    
    if material_name == "Ti-6Al-4V":
        phases = {"Alpha": 70, "Beta": 20, "Martensite": 10}
    elif material_name == "Inconel 718":
        phases = {"Gamma": 60, "Gamma'": 30, "Carbides": 10}
    elif material_name == "SS316L":
        phases = {"Austenite": 85, "Ferrite": 10, "Sigma": 5}
    else:  # AlSi10Mg
        phases = {"Aluminum": 90, "Silicon": 8, "Mg2Si": 2}
    
    return phases

# Main Dashboard
if st.session_state.current_run:
    params = st.session_state.current_run["params"]
    
    # Calculate metrics
    X, Y, T_dist = heat_transfer_model(params)
    cooling_rate = calculate_cooling_rate(T_dist)
    thermal_gradient = calculate_thermal_gradient(T_dist)
    grain_size = predict_grain_size(cooling_rate)
    porosity_risk = predict_porosity(params)
    residual_stress = predict_residual_stress(params)
    phases = simulate_microstructure(grain_size, params["material"])
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌡️ Real-time Simulation",
        "🔬 Microstructure Predictor", 
        "⚠️ Defect Risk Analysis",
        "📊 Process Window Optimizer"
    ])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 3D Melt Pool Visualization
            st.subheader("Melt Pool Temperature Distribution")
            
            fig_3d = go.Figure(data=[
                go.Surface(
                    z=T_dist,
                    x=X,
                    y=Y,
                    colorscale='Viridis',
                    contours_z=dict(
                        show=True,
                        usecolormap=True,
                        highlightcolor="limegreen",
                        project_z=True
                    )
                )
            ])
            
            fig_3d.update_layout(
                title="3D Temperature Profile",
                scene=dict(
                    xaxis_title="X (mm)",
                    yaxis_title="Y (mm)",
                    zaxis_title="Temperature (°C)",
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1)
                    )
                ),
                height=500,
                margin=dict(l=0, r=0, t=50, b=0)
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
        
        with col2:
            st.subheader("Process Metrics")
            
            # Key metrics in cards
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                st.metric("Peak Temperature", f"{T_dist.max():.0f} °C")
                st.metric("Cooling Rate", f"{cooling_rate:.0f} °C/s")
                st.metric("Melt Pool Depth", f"{(T_dist > MATERIAL_DB[params['material']]['melting_point']).sum()/100:.2f} mm")
            
            with metrics_col2:
                st.metric("VED", f"{params['VED']:.1f} J/mm³")
                st.metric("Heat Affected Zone", f"{(T_dist > 500).sum()/100:.2f} mm²")
                st.metric("Thermal Gradient", f"{thermal_gradient:.0f} °C/mm")
            
            st.divider()
            
            # Cooling animation control
            st.subheader("Cooling Visualization")
            time_step = st.slider("Time (ms)", 0, 100, 0, 10)
            
            # Simulate cooling over time
            cooling_factor = np.exp(-time_step / 50)
            T_cooled = T_dist * cooling_factor
            
            fig_cooling = go.Figure(data=[
                go.Heatmap(
                    z=T_cooled,
                    colorscale='RdBu_r',
                    zmid=params['preheat_temp']
                )
            ])
            
            fig_cooling.update_layout(
                title=f"Temperature after {time_step}ms",
                height=300,
                margin=dict(l=0, r=0, t=50, b=0)
            )
            
            st.plotly_chart(fig_cooling, use_container_width=True)
    
    with tab2:
        st.subheader("Microstructure Prediction")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.metric("Predicted Grain Size", f"{grain_size:.1f} µm")
            st.metric("Cooling Rate", f"{cooling_rate:.0f} °C/s")
            primary_phase = list(phases.keys())[0]
            st.metric(f"{primary_phase} Content", f"{phases[primary_phase]:.0f}%")
        
        with col2:
            # Phase distribution pie chart
            fig_pie = px.pie(
                values=list(phases.values()),
                names=list(phases.keys()),
                title="Phase Distribution",
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col3:
            # Texture visualization
            st.subheader("Texture Strength")
            
            # Simulate texture data
            theta = np.linspace(0, 2*np.pi, 100)
            texture_strength = 0.5 + 0.3 * np.cos(4 * theta) * (1 if cooling_rate > 500 else 0.5)
            
            fig_polar = go.Figure(data=[
                go.Scatterpolar(
                    r=texture_strength,
                    theta=theta * 180/np.pi,
                    fill='toself',
                    line_color='blue'
                )
            ])
            
            fig_polar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1])
                ),
                showlegend=False,
                height=300
            )
            
            st.plotly_chart(fig_polar, use_container_width=True)
        
        # Grain structure simulation
        st.subheader("Simulated Grain Structure")
        
        # Generate synthetic grain structure
        np.random.seed(42)
        grain_data = np.random.rand(200, 200)
        for _ in range(5):
            grain_data = ndimage.gaussian_filter(grain_data, sigma=2)
        
        fig_grains = px.imshow(
            grain_data,
            color_continuous_scale='Greys',
            title="Grain Boundary Network"
        )
        
        st.plotly_chart(fig_grains, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Defect Risk Indicators")
            
            # Porosity gauge
            fig_gauge1 = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=porosity_risk,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Porosity Risk %"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "green"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            
            fig_gauge1.update_layout(height=300)
            st.plotly_chart(fig_gauge1, use_container_width=True)
            
            # Lack of fusion gauge
            optimal_ved = MATERIAL_DB[params['material']]['optimal_ved']
            lof_risk = max(0, 100 - params['VED'] / optimal_ved * 100)
            
            fig_gauge2 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=lof_risk,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Lack of Fusion %"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 20], 'color': "green"},
                        {'range': [20, 50], 'color': "yellow"},
                        {'range': [50, 100], 'color': "red"}
                    ]
                }
            ))
            
            fig_gauge2.update_layout(height=300)
            st.plotly_chart(fig_gauge2, use_container_width=True)
        
        with col2:
            st.subheader("Residual Stress Distribution")
            
            # Generate residual stress map
            stress_map = np.zeros((100, 100))
            center_x, center_y = 50, 50
            
            for i in range(100):
                for j in range(100):
                    distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                    stress_map[i, j] = residual_stress * np.exp(-distance/30)
            
            # Add scanning pattern effect
            if params.get("scan_strategy", "Bidirectional") == "Bidirectional":
                stress_map += residual_stress * 0.3 * np.sin(np.linspace(0, 4*np.pi, 100))[:, None]
            
            fig_stress = px.imshow(
                stress_map,
                color_continuous_scale='RdBu_r',
                title="Residual Stress Distribution (MPa)",
                labels={'color': 'Stress (MPa)'}
            )
            
            st.plotly_chart(fig_stress, width='stretch')
            
            # Balling effect indicator
            st.subheader("Balling Effect Risk")
            
            balling_risk = max(0, params['VED'] / optimal_ved - 1) * 100
            
            fig_balling = go.Figure(go.Indicator(
                mode="number+gauge",
                value=balling_risk,
                number={'suffix': "%"},
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Balling Risk"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "orange"},
                    'steps': [
                        {'range': [0, 20], 'color': "green"},
                        {'range': [20, 40], 'color': "yellow"},
                        {'range': [40, 100], 'color': "red"}
                    ]
                }
            ))
            
            fig_balling.update_layout(height=200)
            st.plotly_chart(fig_balling, width='stretch')
    
    with tab4:
        st.subheader("Process Window Optimization")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Generate Ashby-style plot
            power_range = np.linspace(100, 500, 20)
            speed_range = np.linspace(100, 2000, 20)
            P, V = np.meshgrid(power_range, speed_range)
            
            # Calculate VED for each combination
            VED_grid = P / (V * params['hatch_spacing'] * params['layer_thickness'])
            
            # Define zones
            optimal_ved = MATERIAL_DB[params['material']]['optimal_ved']
            zone = np.zeros_like(VED_grid)
            
            zone[(VED_grid > optimal_ved * 0.8) & (VED_grid < optimal_ved * 1.2)] = 1  # Green
            zone[VED_grid <= optimal_ved * 0.8] = 2  # Red (low VED)
            zone[VED_grid >= optimal_ved * 1.2] = 3  # Red (high VED)
            
            # Create plot
            fig_ashby = go.Figure()
            
            # Add zones
            colors = ['lightgreen', 'red', 'red']
            zone_names = ['Optimal', 'Low VED (Lack of Fusion)', 'High VED (Keyholing)']
            
            for i in range(1, 4):
                mask = zone == i
                if mask.any():
                    fig_ashby.add_trace(go.Scatter(
                        x=P[mask].flatten(),
                        y=V[mask].flatten(),
                        mode='markers',
                        marker=dict(size=10, color=colors[i-1], opacity=0.7),
                        name=zone_names[i-1]
                    ))
            
            # Current point
            fig_ashby.add_trace(go.Scatter(
                x=[params['laser_power']],
                y=[params['scan_speed']],
                mode='markers',
                marker=dict(size=20, color='blue', symbol='star'),
                name='Current Parameters'
            ))
            
            fig_ashby.update_layout(
                title="Process Window: Laser Power vs Scan Speed",
                xaxis_title="Laser Power (W)",
                yaxis_title="Scan Speed (mm/s)",
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig_ashby, width='stretch')
        
        with col2:
            st.subheader("Optimization")
            
            # Calculate optimal parameters
            optimal_power = np.sqrt(params['laser_power'] * optimal_ved * params['scan_speed'] * params['hatch_spacing'] * params['layer_thickness'])
            optimal_speed = optimal_power / (optimal_ved * params['hatch_spacing'] * params['layer_thickness'])
            
            st.metric("Recommended Power", f"{optimal_power:.0f} W")
            st.metric("Recommended Speed", f"{optimal_speed:.0f} mm/s")
            st.metric("Target VED", f"{optimal_ved:.1f} J/mm³")
            
            st.divider()
            
            # Parameter suggestions
            st.subheader("Suggestions")
            
            ved_ratio = params['VED'] / optimal_ved
            if ved_ratio < 0.8:
                st.info("🔺 Increase laser power or decrease scan speed")
            elif ved_ratio > 1.2:
                st.info("🔻 Decrease laser power or increase scan speed")
            
            if params['hatch_spacing'] > 0.15:
                st.info("📏 Consider reducing hatch spacing")
            
            if cooling_rate > 1000:
                st.info("❄️ High cooling rate - consider pre-heat")
            
            st.divider()
            
            # Download report
            if st.button("📥 Generate Simulation Report", use_container_width=True):
                st.success("Report generation started!")
                
                # Create a simple PDF report
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                
                pdf.cell(200, 10, txt="AM Digital Twin Simulation Report", ln=1, align='C')
                pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=2)
                
                # Add parameters
                pdf.cell(200, 10, txt="Simulation Parameters:", ln=3)
                for key, value in params.items():
                    pdf.cell(200, 10, txt=f"{key}: {value}", ln=1)
                
                # Add results
                pdf.cell(200, 10, txt=f"Predicted Grain Size: {grain_size:.1f} µm", ln=1)
                pdf.cell(200, 10, txt=f"Porosity Risk: {porosity_risk:.1f}%", ln=1)
                pdf.cell(200, 10, txt=f"Residual Stress: {residual_stress:.0f} MPa", ln=1)
                
                # Save PDF
                pdf_output = pdf.output(dest='S').encode('latin1')
                b64 = base64.b64encode(pdf_output).decode()
                
                href = f'<a href="data:application/pdf;base64,{b64}" download="AM_Simulation_Report.pdf">Download Report</a>'
                st.markdown(href, unsafe_allow_html=True)
    
    # WOW Factor: Real-time parameter adjustment
    st.divider()
    st.subheader("⚡ Real-time Parameter Adjustment")
    
    adj_col1, adj_col2, adj_col3, adj_col4 = st.columns(4)
    
    with adj_col1:
        if st.button("+10% Power", use_container_width=True):
            params["laser_power"] *= 1.1
            params["VED"] = params["laser_power"] / (params["scan_speed"] * params["hatch_spacing"] * params["layer_thickness"])
            st.rerun()
    
    with adj_col2:
        if st.button("-10% Speed", use_container_width=True):
            params["scan_speed"] *= 0.9
            params["VED"] = params["laser_power"] / (params["scan_speed"] * params["hatch_spacing"] * params["layer_thickness"])
            st.rerun()
    
    with adj_col3:
        if st.button("Optimize Hatch", use_container_width=True):
            params["hatch_spacing"] = 0.1  # Optimal default
            params["VED"] = params["laser_power"] / (params["scan_speed"] * params["hatch_spacing"] * params["layer_thickness"])
            st.rerun()
    
    with adj_col4:
        if st.button("Reset", use_container_width=True):
            st.session_state.current_run = {}
            st.rerun()
    
else:
    # Welcome screen
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h2>Welcome to the AM Digital Twin</h2>
            <p>This advanced simulation tool predicts melt pool dynamics, microstructure evolution, 
            and defect formation in additive manufacturing processes.</p>
            <p>Configure your parameters in the sidebar and click "Run Simulation" to begin.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick start examples
        st.subheader("Quick Start Examples")
        
        example_col1, example_col2, example_col3 = st.columns(3)
        
        with example_col1:
            if st.button("Ti-6Al-4V Optimal", use_container_width=True):
                st.session_state.current_run = {
                    "timestamp": datetime.now(),
                    "params": {
                        "laser_power": 300,
                        "scan_speed": 800,
                        "hatch_spacing": 0.1,
                        "layer_thickness": 0.05,
                        "preheat_temp": 200,
                        "material": "Ti-6Al-4V",
                        "VED": 75.0,
                        "beam_diameter": 100,
                        "scan_strategy": "Bidirectional"
                    }
                }
                st.rerun()
        
        with example_col2:
            if st.button("SS316L High Speed", use_container_width=True):
                st.session_state.current_run = {
                    "timestamp": datetime.now(),
                    "params": {
                        "laser_power": 400,
                        "scan_speed": 1500,
                        "hatch_spacing": 0.08,
                        "layer_thickness": 0.04,
                        "preheat_temp": 100,
                        "material": "SS316L",
                        "VED": 104.2,
                        "beam_diameter": 100,
                        "scan_strategy": "Bidirectional"
                    }
                }
                st.rerun()
        
        with example_col3:
            if st.button("Inconel Fine Features", use_container_width=True):
                st.session_state.current_run = {
                    "timestamp": datetime.now(),
                    "params": {
                        "laser_power": 200,
                        "scan_speed": 500,
                        "hatch_spacing": 0.06,
                        "layer_thickness": 0.03,
                        "preheat_temp": 300,
                        "material": "Inconel 718",
                        "VED": 111.1,
                        "beam_diameter": 80,
                        "scan_strategy": "Island"
                    }
                }
                st.rerun()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>AM Digital Twin v2.0 | Powered by Streamlit | Scientific Models Based on AM Research</p>
    <p>⚠️ This tool is for research and educational purposes only.</p>
</div>
""", unsafe_allow_html=True)
