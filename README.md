# 🏟️ AI-Powered Crowd Management & Safety Decision System

An intelligent, rule-based crowd management and incident dispatch system designed for large-scale venues (stadiums, pilgrimage sites, arenas). It dynamically detects bottleneck congestion, evaluates topological gate alternatives, and executes real-time crowd diversion orders along perimeter evacuation corridors.

---

## ⚡ Quick Start (Run on Any Laptop)

### 1. Clone the Repository
```bash
git clone <YOUR-REPO-URL>
cd capstone
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
py -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Run Options

### 🟢 Option 1: Interactive 2D Venue Simulation (Recommended for Demo)
Runs a fast, professional Command & Control (C2) 2D simulation with moving pedestrians, gate occupancy HUD, and real-time incident diversion. **Runs at 60 FPS on CPU with no GPU or videos needed!**

```bash
python examples/visual_simulation.py
```

**🎮 Keyboard Controls During Demo:**
| Key | Action |
|:---:|:---|
| **`[1]`** | Inject crowd surge into **Gate 1 (North)** |
| **`[2]`** | Inject crowd surge into **Gate 2 (East)** |
| **`[3]`** | Inject crowd surge into **Gate 3 (West)** |
| **`[F]`** | Toggle speed (`1.0x Normal` → `2.0x Fast` → `3.5x Turbo`) |
| **`[C]`** | Clear all crowd and reset venue |
| **`[SPACE]`** | Pause / Resume |
| **`[Q]`** | Exit |

> **How to show the diversion logic:** Press `1` a few times until Gate 1 exceeds its capacity limit (Red). Watch the Decision Engine trigger **`DIVERT G1 ➔ G2`**, and observe incoming pedestrians re-routing **strictly along the outer perimeter ring concourse (outside the event)** towards Gate 2!

---

### 🌐 Option 2: Live Web Operations Dashboard
Starts a FastAPI server with WebSocket real-time telemetry and the **Organiser Advisories Panel**.

```bash
python examples/run_web_dashboard.py
```
Open your browser at: **`http://localhost:8000`**

---

### 🤖 Option 3: Terminal Decision Engine Demo
Evaluates multi-gate scenarios with explainable rule metrics:
```bash
python examples/run_decision_demo.py
```

---

## 📐 Architecture Highlights
- **Rule-Based Decision Engine**: Evaluates gate capacities, distance proximity, crowd density, and flow trend to select optimal diversion routes with hysteresis (anti-oscillation).
- **Perimeter Ring Routing**: When a gate is closed or diverted, pedestrian traffic is directed through external bypass concourses without cutting through restricted event grounds.
- **Organiser Advisories**: Generates ISO safety alerts (`CRUSH_RISK`, `BOTTLENECK_FORMING`, `GATE_OVERLOAD`, `IMBALANCED_LOAD`).
