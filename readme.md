# 🏭 Additive Manufacturing Digital Twin: Predictive Process Modeling

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

[![Demo](https://img.shields.io/badge/🚀-Live_Demo-orange?style=for-the-badge)](https://am-digital-twin.streamlit.app/)
[![Paper](https://img.shields.io/badge/📄-Conference_Paper-blue?style=for-the-badge)](#)
[![DOI](https://img.shields.io/badge/DOI-10.xxxx/xxxxxx-blue?style=for-the-badge)](#)

</div>

---

## ✨ Overview

**AM Digital Twin** is a physics-informed simulation platform combined with data-driven components to predict thermal history, microstructure evolution, and defect risks in Laser Powder Bed Fusion (LPBF). It is intended as a research and teaching tool for process understanding, optimization, and digital twin concept validation.

Key capabilities:
- Predict melt pool geometry and 3D temperature fields
- Simulate microstructure evolution (grain size, phase fractions)
- Estimate defect risk (porosity, lack of fusion, keyholing)
- Provide process-parameter recommendations and process windows
- Interactive visualization via a Streamlit dashboard

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git

### 1. Clone the repository
```bash
git clone https://github.com/Areebrizz/Digital-Twin-of-Additive-Manufacturing.git
cd Digital-Twin-of-Additive-Manufacturing
```

### 2. Create and activate a virtual environment (recommended)

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

Linux / macOS:
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

Example requirements (included in `requirements.txt`):
```text
streamlit==1.28.0
numpy==1.24.3
pandas==2.0.3
plotly==5.17.0
matplotlib==3.7.2
scipy==1.11.1
```

### 4. Run the application
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501` by default.

---

## 🔧 Features

Core modules:
- Thermal analysis: transient heat transfer, Gaussian laser source, melt pool estimation
- Microstructure prediction: grain size estimation, phase fraction models (Hunt-style scaling)
- Defect risk assessment: porosity and lack-of-fusion estimators, keyhole likelihood
- Process optimization: VED-based parameter scanning and scoring
- Visualization: heatmaps, 3D surfaces, gauge widgets, process windows

User-facing features:
- Light / Dark theme toggle
- Real-time interactive controls
- Export simulation results as CSV
- Example configurations for common alloys

---

## 📦 Project Structure

```
Digital-Twin-of-Additive-Manufacturing/
│
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file

---

## 🔬 Scientific Models (brief)

Heat transfer (transient diffusion with a Gaussian laser source):

∂T/∂t = α∇²T + Q/(ρ Cp)

where Q is modeled as a moving Gaussian heat source:
Q(x, y, t) = η P /(π r^2 v) × exp(-(x^2 + y^2) / r^2)

- α: thermal diffusivity
- ρ: density
- Cp: specific heat
- η: absorption efficiency
- P: laser power
- r: characteristic radius of energy distribution
- v: scan speed

Grain size (Hunt-type empirical scaling):
d = k × (dT/dt)^(-n)
- k, n: material-specific constants

Defect risk:
- Porosity and lack of fusion modeled using logistic or sigmoid functions of VED deviation and melt pool metrics
- Residual stress estimated from thermal gradients and cooling rates (qualitative / relative scoring)

---

## 📊 Supported Materials (examples)

| Material     | Optimal VED (J/mm³) | Melting Point (°C) | Typical Applications               |
|--------------|----------------------|---------------------|------------------------------------|
| Ti-6Al-4V    | 60                   | 1923                | Aerospace, biomedical              |
| Inconel 718  | 50                   | 1600                | Turbine blades, high-temperature   |
| SS316L       | 70                   | 1670                | Marine, chemical processing        |
| AlSi10Mg     | 40                   | 860                 | Automotive, lightweight            |

To add materials, edit the MATERIAL_DB dictionary in `am_digital_twin.py` (example below).

---

## 🛠️ Configuration & Extension

Customizing the material database
```python
MATERIAL_DB["YourMaterial"] = {
    "density": 8000,                # kg/m³
    "thermal_conductivity": 15,     # W/m·K
    "specific_heat": 500,           # J/kg·K
    "melting_point": 1500,          # °C
    "optimal_ved": 55,              # J/mm³
    "k": 1.2,                       # Hunt model constant
    "n": 0.33,                      # Hunt model exponent
    "color": "#FF5733"              # UI color (hex)
}
```

Tuning model behavior:
- Laser absorption η, process efficiency adjustments
- Mesh / discretization parameters in `thermal.py`
- Empirical fit parameters for defect probability models

---

## ⚠️ Limitations

- Educational / research-grade — simplified physics for speed and interactivity
- Not intended for part qualification, certification, or safety-critical decisions
- Material database is limited — users should validate parameters with experimental data
- Some modules (e.g., microstructure, residual stress) use empirical or reduced-order approximations

---

## ✅ Intended Use Cases

- Academic research demonstrations and prototyping
- Teaching manufacturing/digital twin concepts
- Process-parameter sensitivity studies and visualization
- Conference and portfolio presentations

---

## 📚 References

- King, W. E., et al. (2014). Laser powder bed fusion additive manufacturing of metals; physics, computational, and materials challenges. Applied Physics Reviews.
- DebRoy, T., et al. (2018). Additive manufacturing of metallic components – Process, structure and properties. Progress in Materials Science.
- Hunt, J. D. (1984). Steady state columnar and equiaxed growth of dendrites and eutectic. Materials Science and Engineering.
- Mukherjee, T., et al. (2017). Printability of alloys for additive manufacturing. Scientific Reports.
- Vrancken, B., et al. (2012). Heat treatment of Ti6Al4V produced by Selective Laser Melting: Microstructure and mechanical properties. Journal of Alloys and Compounds.

---

## 🤝 Contributing

We welcome contributions!

1. Fork the repository
2. Create a branch: `git checkout -b feature/YourFeature`
3. Commit your changes: `git commit -m "Add feature: ..."`
4. Push to your branch: `git push origin feature/YourFeature`
5. Open a Pull Request describing the change

Please follow existing code style and include tests for new modules where applicable.

---

## 🐛 Reporting Issues

Use the GitHub Issues page for bug reports or feature requests:
[Issues · Areebrizz / Digital-Twin-of-Additive-Manufacturing](https://github.com/Areebrizz/Digital-Twin-of-Additive-Manufacturing/issues)

When reporting issues, please include:
- Steps to reproduce
- Python version and OS
- Minimal example input and expected behavior
- Any error tracebacks

---

## 📄 License

This project is released under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

## 👨‍💻 Author

Muhammad Areeb Rizwan Siddiqui  
Mechanical Engineer | Automation & Manufacturing Systems

- Website: [https://www.areebrizwan.com](https://www.areebrizwan.com)  
- LinkedIn: [https://www.linkedin.com/in/areebrizwan](https://www.linkedin.com/in/areebrizwan)  

---

If you find this project useful, please star the repository:
[Star on GitHub](https://github.com/Areebrizz/Digital-Twin-of-Additive-Manufacturing)
