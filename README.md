# Multi-Project Resource-Constrained Scheduling Optimization (RCPSP) for EPC Solar Portfolios

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Optimization](https://img.shields.io/badge/Solver-IBM%20CPLEX%20%2F%20DOcplex-orange.svg)](https://www.ibm.com/analytics/cplex-optimizer)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Domain](https://img.shields.io/badge/Domain-Operations%20Research%20%7C%20Renewable%20Energy-brightgreen.svg)]()

Implementation of exact mathematical modeling (**Mixed-Integer Linear Programming - MILP**) and metaheuristic optimization (**Greedy Randomized Adaptive Search Procedure - GRASP**) to minimize weighted total tardiness across concurrent Engineering, Procurement, and Construction (EPC) solar rooftop projects subject to multi-department resource limits, task precedences, and contract deadlines.

Based on capstone research conducted at **Ho Chi Minh City University of Technology (HCMUT)** on industrial portfolios from **SolarBK**.

---

## 📌 Problem Overview & Industrial Context

In utility and commercial rooftop solar EPC operations, multiple projects run concurrently across shared cross-functional resource pools:
* **PMU** (Project Management Unit)
* **ENG** (Engineering & Design)
* **SC** (Supply Chain & Procurement)
* **QC** (Quality Control & Compliance)
* **LOG** (Logistics & Material Dispatch)
* **SITE** (Field Construction & Mounting)

Manual scheduling often fails to foresee cross-departmental bottlenecks. When shared teams (e.g., SITE mounting or SC procurement) are overcommitted simultaneously, critical-path activities stall, causing multi-month project delays (average historic portfolio SPI ≈ 0.533).

This repository formulates the multi-project portfolio as a **Single-Mode Resource-Constrained Project Scheduling Problem (RCPSP)** with department eligibility, task precedence lags, and capacity limits.

---

## 📐 Mathematical Formulation

### Objective Function
Minimize weighted project completion delays, augmented with an infinitesimal makespan tie-breaker to incentivize compact baseline schedules:

$$\min Z = \sum_{p \in \mathcal{P}} w_p \cdot T_p + \varepsilon \cdot \text{Makespan}$$

Where:
* $w_p$: Strategic commercial priority weight for project $p$.
* $T_p = \max(0, C_p - DD_p)$: Tardiness of project $p$ beyond contractual due date $DD_p$.
* $C_p$: Completion time of the terminal network activity of project $p$.
* $\varepsilon = 10^{-4}$: Secondary weight favoring shorter overall portfolio horizon.

### Core Constraints
1. **Assignment & Single Execution:** Each activity $a$ starts exactly once within valid planning periods:
   $$\sum_{t=0}^{H - D_{p,a}} X_{p,a,t} = 1 \quad \forall p \in \mathcal{P}, a \in \mathcal{A}_p$$
2. **Precedence with Lags (Intra- & Inter-Project):**
   $$X_{p, \text{pred}, t_1} + X_{p', \text{succ}, t_2} \le 1 \quad \forall t_2 < t_1 + D_{p,\text{pred}} + \text{Lag}$$
3. **Daily Department Capacity:** Total daily manpower usage cannot exceed available Full-Time Equivalent (FTE) capacity:
   $$\sum_{p} \sum_{a \in \mathcal{A}_p(d)} \sum_{\tau = \max(0, t - D_{p,a} + 1)}^{\min(t, H - D_{p,a})} \text{Manpower}_{p,a} \cdot X_{p,a,\tau} \le \text{Capacity}_d \quad \forall d \in \mathcal{D}, \forall t \in \mathcal{T}$$

---

## 🚀 Optimization Algorithms Implemented

| Method | Technology | Characteristics & Benchmark Performance |
| :--- | :--- | :--- |
| **Exact MILP** | Python, `docplex` (IBM CPLEX) | Solves linearized time-indexed binary formulation to provable global optimality for small and medium instances ($\le 4$ projects). Encounters combinatorial explosion on large-scale industrial portfolios ($>4$ hours without convergence). |
| **Metaheuristic GRASP** | Python (Pure Algorithm) | Two-phase solver: <br>1. **Greedy Randomized Construction** using Restricted Candidate Lists (RCL) controlled by parameter $\alpha$.<br>2. **Local Search** exploring earlier feasible start shifts and inter-project task swaps.<br>Achieves **0 days tardiness in 45.25 seconds** on the full 6-project industrial portfolio where exact MILP timed out. |

---

## 📊 Experimental Results

Computational evaluation across problem scales:

| Scale | Configuration | MILP Tardiness | MILP Runtime | GRASP Tardiness | GRASP Runtime | Solution Quality |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Small** | 2 Projects, 2 Depts, 10 Acts | **0 days** | 23.07 s | **0 days** | 3.59 s | Identical Makespan (6.4× faster) |
| **Medium** | 4 Projects, 3 Depts, 24 Acts | **0 days** | 82.80 s | **0 days** | 12.14 s | Optimal matched |
| **Industrial Scale** | 6 Projects, 6 Depts, 30+ Acts | DNF (>4h) | Timed out | **0 days** | **45.25 s** | Industrial Feasibility Restored |

---

## 🛠️ Installation & Usage

### 1. Prerequisites & Environment Setup
Clone this repository and install dependencies:
```bash
git clone [https://github.com/](https://github.com/)<your-username>/epc-solar-rcpsp-optimization.git
cd epc-solar-rcpsp-optimization
pip install -r requirements.txt
```

### 2. Run the Exact MILP Model (CPLEX)
```bash
python src/epc_optimized.py
```
*Note: Requires IBM ILOG CPLEX Community Edition or academic/commercial license installed for `docplex`.*

### 3. Run the GRASP Metaheuristic Scheduler
```bash
python src/grasp_scheduler_optimized.py
```
Outputs:
* Console audit of project completion dates, tardiness status, and department resource utilization.
* High-resolution Gantt charts exported to `results/grasp_gantt.png` displaying resource colors and precedence arcs.

---

## 📂 Data Input Schema

Excel instance workbooks (`.xlsx`) contain the following standardized sheets:
* **`Departments`**: `Department_ID`, `Capacity` (daily manpower limit), `Hourly_Cost`.
* **`Activities`**: `Project_ID`, `Activity_ID`, `Duration`, `Manpower_Qty`, `Resource_ID`, `Predecessor`, `Due_Date`.
* **`Settings`**: Planning `horizon`, project weights ($w_p$).

---

## 👤 Author
* **Lê Lam Điền** – Industrial Management & Operations Research, Ho Chi Minh City University of Technology (HCMUT).
* LinkedIn: [Your Profile Link]
* Email: [Your Email]
