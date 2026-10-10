"""
EPC Project Optimization Models - Model A: DAP/MILP
Mathematical Model Implementation - ADAPTED FOR NEW EXCEL FORMAT

Objective: Minimize Z = Σ_{p∈P} w_p T_p + ε · Makespan
  (primary: weighted total tardiness; secondary tie-break: prefer shorter max completion)
"""

import pandas as pd
import numpy as np 
from docplex.mp.model import Model
import os
import time
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Secondary objective weight: prefer lower max project completion when tardiness is equal
EPS_MAKESPAN = 1e-4

def department_utilization_over_makespan(
    dept_rows: pd.DataFrame,
    capacity: float,
    makespan: float,
) -> float:
    """
    Utilisation (%) = total manpower-days assigned to dept / (capacity × makespan).
    This measures how much of the department's total available capacity over the
    full schedule horizon was actually consumed by work.
    """
    if capacity <= 0 or makespan <= 0 or dept_rows is None or dept_rows.empty:
        return 0.0

    total_work = (dept_rows["Duration"] * dept_rows["Manpower"]).sum()
    return 100.0 * total_work / (capacity * makespan)


def load_standardized_excel(excel_file: str) -> Dict:
    """Load and parse standardized Excel file into structured dictionary - NEW FORMAT"""
    
    if not os.path.exists(excel_file):
        raise FileNotFoundError(f"Excel file not found: {excel_file}")
    
    raw_data = {
        'departments': {},
        'activities': {},
        'precedence': [],
        'projects': {},
        'settings': {}
    }
    
    # Load Departments
    df_depts = pd.read_excel(excel_file, sheet_name='Departments')
    
    for _, row in df_depts.iterrows():
        dept_id = str(row['Department_ID']).strip()
        raw_data['departments'][dept_id] = {
            'capacity': float(row['Capacity']) if pd.notna(row['Capacity']) else 0.0,
            'cost': float(row['Hourly_Cost']) if pd.notna(row['Hourly_Cost']) else 0.0
        }
    
    # Load Activities - NEW FORMAT with embedded predecessor and due date
    df_activities = pd.read_excel(excel_file, sheet_name='Activities')
    
    for _, row in df_activities.iterrows():
        project = str(row['Project_ID']).strip()
        activity = str(row['Activity_ID']).strip()
        
        # Resource_ID is the assigned department (use as single eligible dept)
        resource_id = str(row['Resource_ID']).strip() if pd.notna(row['Resource_ID']) else ''
        eligible_depts = [resource_id] if resource_id else []
        
        raw_data['activities'][(project, activity)] = {
            'duration': float(row['Duration']) if pd.notna(row['Duration']) else 0.0,
            'manpower_allocated': float(row['Manpower_Qty']) if pd.notna(row['Manpower_Qty']) else 0.0,
            'cost': 0.0,  # Can be calculated from duration * manpower * dept cost if needed
            'eligible_depts': eligible_depts
        }
        
        # Extract precedence from Predecessor column (supports comma-separated multiple predecessors)
        if pd.notna(row['Predecessor']):
            predecessor_str = str(row['Predecessor']).strip()
            # Split by comma so e.g. "P1-Logistics,P1-Roof-Mounting" -> two precedence arcs
            for pred in (p.strip() for p in predecessor_str.split(',') if p.strip()):
                raw_data['precedence'].append((project, pred, activity, 0.0))
        
        # Extract project due dates — take the minimum across all activity rows
        # to guard against inconsistent values in the Excel sheet
        if pd.notna(row['Due_Date']):
            due_date = float(row['Due_Date'])
            if project not in raw_data['projects']:
                raw_data['projects'][project] = {'due_date': due_date}
            else:
                raw_data['projects'][project]['due_date'] = min(
                    raw_data['projects'][project]['due_date'], due_date
                )
    
    # Load Settings
    df_settings = pd.read_excel(excel_file, sheet_name='Settings')
    
    for _, row in df_settings.iterrows():
        param = str(row['Parameters']).strip().lower()
        raw_data['settings'][param] = row['Value']
    
    return raw_data


class InputData:
    """Structured input data for optimization model"""
    
    def __init__(self, excel_file: str = "Sample data 4,3.xlsx"):
        raw_data = load_standardized_excel(excel_file)
        
        # Departments
        self.departments = {
            d_id: {
                'FTE': data['capacity'],
                'capacity_per_period': data['capacity'],
                'cost': data['cost']
            }
            for d_id, data in raw_data['departments'].items()
        }
        
        # Activities grouped by project
        self.activities = {}
        for (p, a), data in raw_data['activities'].items():
            if p not in self.activities:
                self.activities[p] = []
            self.activities[p].append({
                'name': a,
                'duration': data['duration'],
                'manpower_allocated': data['manpower_allocated'],
                'cost': data['cost'],
                'eligible_depts': data['eligible_depts']
            })
        
        # Precedence relationships
        self.precedence = raw_data['precedence']
        
        # Project due dates
        horizon = int(raw_data['settings'].get('horizon', 120))
        self.project_due_dates = {}
        
        for p in self.get_all_projects():
            if 'projects' in raw_data and p in raw_data['projects']:
                self.project_due_dates[p] = float(raw_data['projects'][p]['due_date'])
            else:
                # Use horizon as default
                self.project_due_dates[p] = float(horizon)
        
        # Planning horizon
        self.planning_horizon = horizon
        self.time_periods = list(range(horizon + 1))

        # Project weights (e.g. P1_weight, P2_weight …) — default 1.0
        self.project_weights = {}
        for p in self.get_all_projects():
            key = f'{p.lower()}_weight'
            self.project_weights[p] = float(raw_data['settings'].get(key, 1.0))
    
    def get_all_projects(self) -> List[str]:
        return list(self.activities.keys())
    
    def get_all_departments(self) -> List[str]:
        return list(self.departments.keys())
    
    def calculate_department_capacity(self, dept_name: str) -> float:
        return self.departments[dept_name]['capacity_per_period']
    
    def get_project_due_date(self, project_id: str) -> float:
        return self.project_due_dates.get(project_id, self.planning_horizon)
    
    def get_terminal_activities(self, project_id: str) -> Optional[str]:
        """Get the last activity (terminal node) for a project"""
        activities = self.activities.get(project_id, [])
        if not activities:
            return None
        
        # Find activities with out-degree = 0 (not predecessors to any other activity)
        predecessors = {pred for p, pred, succ, _ in self.precedence if p == project_id}
        end_activities = [a['name'] for a in activities if a['name'] not in predecessors]
        
        return end_activities[-1] if end_activities else activities[-1]['name']

    def resolve_predecessor_project(self, successor_project: str, pred_name: str) -> Optional[str]:
        """Resolve which project the predecessor activity belongs to (for cross-project precedence).
        Returns the project id if found, else None. Prefers successor_project if pred exists there."""
        if successor_project in self.activities:
            if any(a['name'] == pred_name for a in self.activities[successor_project]):
                return successor_project
        for p, acts in self.activities.items():
            if any(a['name'] == pred_name for a in acts):
                return p
        return None


class ModelA_DAP:
    """
    Model A: DAP/MILP - Unified Model with Time-Indexed Variables
    Objective: Minimize Z = Σ_{p∈P} w_p T_p + ε · Makespan (Makespan = max_p C[p])
    """

    def __init__(self, data: InputData):
        self.data = data
        self.model = Model(name='DAP_Model_A')
        self.X = {}  # X[p, a, t]: activity start time
        self.Y = {}  # Y[p, a, d]: activity-department assignment
        self.C = {}  # C[p]: project completion time
        self.T = {}  # T[p]: project tardiness
        self.Makespan = None  # auxiliary: max project completion (tie-break in objective)
        self.solution = None
    
    def build_model(self):
        """Build the MILP model"""
        projects = self.data.get_all_projects()
        departments = self.data.get_all_departments()
        time_periods = self.data.time_periods
        
        print(f"Building Model A: {len(projects)} projects, {len(departments)} departments, "
              f"horizon={self.data.planning_horizon}")
        
        # Decision Variables
        for p in projects:
            activities = self.data.activities[p]
            
            for act in activities:
                a = act['name']
                
                # X[p, a, t]: Binary variable for activity start time
                # Only create for valid start times: t + duration <= horizon
                # (starting at t=horizon with duration>0 would finish beyond the horizon)
                max_start = self.data.planning_horizon - int(act['duration'])
                for t in range(max_start + 1):
                    self.X[p, a, t] = self.model.binary_var(name=f'X_{p}_{a}_{t}')
                
                # Y[p, a, d]: Binary variable for department assignment.
                # Only created when the activity has more than one eligible department.
                # For activities with a single fixed department (as in current data),
                # Y is trivially 1 and the assign_once constraint is skipped.
                eligible = act['eligible_depts']
                if not eligible:
                    eligible = departments
                
                if len(eligible) > 1:
                    for d in eligible:
                        self.Y[p, a, d] = self.model.binary_var(name=f'Y_{p}_{a}_{d}')
            
            # C[p]: Continuous variable for project completion time
            self.C[p] = self.model.continuous_var(name=f'C_{p}', lb=0)
            
            # T[p]: Continuous variable for project tardiness
            self.T[p] = self.model.continuous_var(name=f'T_{p}', lb=0)
        
        # Constraints
        self._add_assignment_constraints(projects)
        self._add_precedence_constraints(projects)
        self._add_capacity_constraints(projects, departments, time_periods)
        self._add_completion_constraints(projects)
        self._add_tardiness_constraints(projects)

        # Makespan >= C[p] for all p → at optimum Makespan = max_p C[p] (drives tighter schedules
        # when weighted tardiness is already minimal).
        self.Makespan = self.model.continuous_var(name='Makespan', lb=0)
        for p in projects:
            self.model.add_constraint(
                self.Makespan >= self.C[p],
                ctname=f'makespan_ge_{p}',
            )

        # Objective: weighted total tardiness + small makespan term (tie-break)
        self.model.minimize(
            self.model.sum(
                self.data.project_weights.get(p, 1.0) * self.T[p] for p in projects
            )
            + EPS_MAKESPAN * self.Makespan
        )
        
        print(f"Model built: {self.model.number_of_variables} variables, "
              f"{self.model.number_of_constraints} constraints")
    
    def _add_assignment_constraints(self, projects):
        """Each activity must start exactly once and be assigned to exactly one department"""
        for p in projects:
            for act in self.data.activities[p]:
                a = act['name']
                eligible = act['eligible_depts']
                if not eligible:
                    eligible = self.data.get_all_departments()
                
                # Start exactly once (only over valid start times for this activity)
                max_start = self.data.planning_horizon - int(act['duration'])
                valid_times = range(max_start + 1)
                self.model.add_constraint(
                    self.model.sum(self.X[p, a, t] for t in valid_times) == 1,
                    ctname=f'start_once_{p}_{a}'
                )
                
                # Assign to exactly one department — only needed when there is a choice
                eligible = act['eligible_depts']
                if not eligible:
                    eligible = self.data.get_all_departments()
                if len(eligible) > 1:
                    self.model.add_constraint(
                        self.model.sum(self.Y[p, a, d] for d in eligible) == 1,
                        ctname=f'assign_once_{p}_{a}'
                    )
    
    def _add_precedence_constraints(self, projects):
        """Precedence relationships with time lags (same-project and cross-project)."""
        for p, pred, succ, lag in self.data.precedence:
            # Resolve which project the predecessor belongs to
            pred_project = self.data.resolve_predecessor_project(p, pred)
            if pred_project is None:
                print(f"  [Warning] Precedence skipped: predecessor '{pred}' not found in any project (successor: {succ} in {p})")
                continue

            pred_act = next((a for a in self.data.activities[pred_project] if a['name'] == pred), None)
            succ_act = next((a for a in self.data.activities[p] if a['name'] == succ), None)
            if not pred_act or not succ_act:
                continue

            duration_pred = pred_act['duration']
            duration_succ = succ_act['duration']

            # Valid start-time windows for each activity
            max_t_pred = self.data.planning_horizon - int(duration_pred)
            max_t_succ = self.data.planning_horizon - int(duration_succ)

            # Start time of successor >= Finish time of predecessor + lag
            if pred_project == p:
                # Same project: use X[p, pred, t_pred] and X[p, succ, t_succ]
                for t_pred in range(max_t_pred + 1):
                    for t_succ in range(max_t_succ + 1):
                        if t_succ < t_pred + duration_pred + lag:
                            self.model.add_constraint(
                                self.X[p, pred, t_pred] + self.X[p, succ, t_succ] <= 1,
                                ctname=f'prec_{p}_{pred}_{succ}_{t_pred}_{t_succ}'
                            )
            else:
                # Cross-project: link X[pred_project, pred, t_pred] and X[p, succ, t_succ]
                for t_pred in range(max_t_pred + 1):
                    for t_succ in range(max_t_succ + 1):
                        if t_succ < t_pred + duration_pred + lag:
                            self.model.add_constraint(
                                self.X[pred_project, pred, t_pred] + self.X[p, succ, t_succ] <= 1,
                                ctname=f'prec_{pred_project}_{pred}_{p}_{succ}_{t_pred}_{t_succ}'
                            )
    
    def _add_capacity_constraints(self, projects, departments, time_periods):
        """Resource capacity constraints — fully linear formulation.

        Since each activity has exactly one eligible department (Resource_ID is always
        specified in the Excel), Y[p,a,d] = 1 for that dept and 0 otherwise.
        We therefore drop Y from the workload expression entirely and simply sum the
        X variables for activities assigned to each department.  This keeps the model
        a pure MILP (no bilinear X*Y terms that would create a much harder MIQCP).
        """
        for d in departments:
            capacity = self.data.calculate_department_capacity(d)

            for t in time_periods:
                workload_expr = []

                for p in projects:
                    for act in self.data.activities[p]:
                        a = act['name']
                        eligible = act['eligible_depts']
                        if not eligible:
                            eligible = departments

                        # Only include activities whose sole eligible dept is d
                        if d not in eligible:
                            continue

                        duration = act['duration']
                        manpower = act['manpower_allocated']
                        duration_periods = int(math.ceil(duration))
                        max_start = self.data.planning_horizon - int(duration)

                        # Activity is active at period t if it started in [t-dur+1, t]
                        for t_start in range(max(0, t - duration_periods + 1), min(t, max_start) + 1):
                            if (p, a, t_start) in self.X:
                                # Linear: no Y multiplication needed — dept is fixed
                                workload_expr.append(manpower * self.X[p, a, t_start])

                if workload_expr:
                    self.model.add_constraint(
                        self.model.sum(workload_expr) <= capacity,
                        ctname=f'capacity_{d}_{t}'
                    )

    def _activity_finish_expr(self, project: str, activity_name: str, duration: float):
        """Finish time of activity as linear expr.: Σ_t t·X[p,a,t] + duration (exactly one X is 1)."""
        max_start = self.data.planning_horizon - int(duration)
        valid_times = range(max_start + 1)
        return (
            self.model.sum(t * self.X[project, activity_name, t] for t in valid_times)
            + duration
        )

    def _add_completion_constraints(self, projects):
        """Project completion time C[p] >= finish time of terminal activity (last in network)."""
        for p in projects:
            last_activity = self.data.get_terminal_activities(p)
            if not last_activity:
                continue

            last_act = next((a for a in self.data.activities[p] if a['name'] == last_activity), None)
            if not last_act:
                continue

            duration = last_act['duration']
            finish_expr = self._activity_finish_expr(p, last_activity, duration)

            self.model.add_constraint(
                self.C[p] >= finish_expr,
                ctname=f'completion_{p}',
            )
    
    def _add_tardiness_constraints(self, projects):
        """Tardiness = max(0, completion - due_date)"""
        for p in projects:
            due_date = self.data.get_project_due_date(p)
            
            # T[p] >= C[p] - DD[p]
            self.model.add_constraint(
                self.T[p] >= self.C[p] - due_date,
                ctname=f'tardiness_{p}'
            )
    
    def solve(self, time_limit: int = 3600) -> bool:
        """Solve the model"""
        print(f"\nSolving with time limit: {time_limit}s")
        self.model.set_time_limit(time_limit)
        
        solution = self.model.solve(log_output=True)
        
        if solution:
            self.solution = solution
            print(f"✓ Solution found!")
            print(f"  Objective: {solution.objective_value:.2f}")
            print(f"  Gap: {self.model.solve_details.mip_relative_gap * 100:.2f}%")
            return True
        else:
            print(f"✗ No solution found")
            return False
    
    def print_solution(self, schedule_df=None):
        """Print solution summary: objective, per-project breakdown, and department utilisation."""
        if not self.solution:
            print("No solution available")
            return

        print("\n" + "=" * 80)
        print("SOLUTION SUMMARY")
        print("=" * 80)

        # ── Objective ────────────────────────────────────────────────────────
        projects_list = self.data.get_all_projects()
        weighted_tard = sum(
            self.data.project_weights.get(p, 1.0) * self.solution.get_value(self.T[p])
            for p in projects_list
        )
        makespan_obj = max(self.solution.get_value(self.C[p]) for p in projects_list)
        secondary_term = EPS_MAKESPAN * makespan_obj
        combined_obj = self.solution.objective_value
        unweighted = sum(
            max(0.0, self.solution.get_value(self.C[p]) - self.data.get_project_due_date(p))
            for p in projects_list
        )
        print(f"\n  Weighted tardiness (primary):   {weighted_tard:.4f}")
        print(f"  ε × makespan (ε={EPS_MAKESPAN:g}): {secondary_term:.6f}")
        print(f"  Combined objective:             {combined_obj:.6f}")
        print(f"  Unweighted total tardiness:       {unweighted:.2f} days")
        print(f"  MIP gap:                          {self.model.solve_details.mip_relative_gap * 100:.2f}%")

        # ── Per-project breakdown ─────────────────────────────────────────────
        print(f"\n{chr(9472)*70}")
        print(f"  {'Project':<10} {'Weight':<8} {'Completion':<12} {'Due Date':<12} "
              f"{'Tardiness':<12} {'Status'}")
        print(f"{chr(9472)*70}")

        makespan = 0.0
        for p in self.data.get_all_projects():
            completion = self.solution.get_value(self.C[p])
            due_date   = self.data.get_project_due_date(p)
            tardiness  = self.solution.get_value(self.T[p])
            weight     = self.data.project_weights.get(p, 1.0)
            status     = "✓ ON TIME" if tardiness < 0.01 else "⚠  LATE"
            makespan   = max(makespan, completion)
            print(f"  {p:<10} {weight:<8.1f} {completion:<12.1f} {due_date:<12.1f} "
                  f"{tardiness:<12.2f} {status}")

        print(f"\n  Makespan: {makespan:.1f} days")

        # ── Department utilisation ────────────────────────────────────────────
        # Use schedule_df if available for exact per-day peak load calculation
        if schedule_df is not None and not schedule_df.empty:
            df = schedule_df
        else:
            rows = []
            for p in self.data.get_all_projects():
                for act in self.data.activities[p]:
                    a      = act["name"]
                    max_st = self.data.planning_horizon - int(act["duration"])
                    start  = next(
                        (t for t in range(max_st + 1)
                         if (p, a, t) in self.X
                         and self.solution.get_value(self.X[p, a, t]) > 0.5),
                        None,
                    )
                    elig = act["eligible_depts"] or self.data.get_all_departments()
                    dept = (elig[0] if len(elig) == 1
                            else next(
                                (d for d in elig
                                 if (p, a, d) in self.Y
                                 and self.solution.get_value(self.Y[p, a, d]) > 0.5),
                                None,
                            ))
                    if start is not None and dept is not None:
                        rows.append({
                            "Department": dept,
                            "Start":      start,
                            "Duration":   act["duration"],
                            "Finish":     start + act["duration"],
                            "Manpower":   act["manpower_allocated"],
                        })
            df = pd.DataFrame(rows)

        print(f"\n{chr(9472)*80}")
        print(
            f"  Utilisation = total manpower-days / (capacity × makespan)"
        )
        print(
            f"  {'Department':<12} {'Activities':<12} {'Total Work':<14} "
            f"{'Capacity/Day':<14} {'Util. %':<10} {'Peak Load'}"
        )
        print(f"{chr(9472)*80}")

        for dept_name in sorted(self.data.get_all_departments()):
            capacity  = self.data.calculate_department_capacity(dept_name)
            dept_rows = df[df["Department"] == dept_name] if not df.empty else df

            num_acts   = len(dept_rows)
            total_work = (dept_rows["Duration"] * dept_rows["Manpower"]).sum() if num_acts else 0.0
            util = department_utilization_over_makespan(dept_rows, capacity, makespan)

            # Peak simultaneous manpower on any single day
            peak = 0.0
            if not dept_rows.empty and "Start" in dept_rows.columns:
                day_load: dict = {}
                for _, row in dept_rows.iterrows():
                    for day in range(int(row["Start"]), int(row["Finish"])):
                        day_load[day] = day_load.get(day, 0.0) + row["Manpower"]
                peak = max(day_load.values()) if day_load else 0.0

            over     = " ⚠ OVER CAPACITY" if peak > capacity + 1e-6 else ""
            peak_str = f"{peak:.1f} / {capacity:.1f}" if peak > 0 else f"— / {capacity:.1f}"

            print(f"  {dept_name:<12} {num_acts:<12} {total_work:<14.1f} "
                  f"{capacity:<14.1f} {util:>8.1f} %  {peak_str}{over}")

        print(f"{chr(9472)*80}")
    
    def get_schedule(self) -> Optional[pd.DataFrame]:
        """Extract schedule as DataFrame"""
        if not self.solution:
            return None
        
        schedule_data = []
        
        for p in self.data.get_all_projects():
            for act in self.data.activities[p]:
                a = act['name']
                
                # Find start time — only iterate over valid start times for this activity
                max_start = self.data.planning_horizon - int(act['duration'])
                start_time = None
                for t in range(max_start + 1):
                    if (p, a, t) in self.X and self.solution.get_value(self.X[p, a, t]) > 0.5:
                        start_time = t
                        break
                
                # Find department:
                # If the activity has exactly one eligible dept, Y was never created — use it directly.
                # If multiple depts were eligible, read from Y.
                eligible = act['eligible_depts']
                if not eligible:
                    eligible = self.data.get_all_departments()
                
                if len(eligible) == 1:
                    department = eligible[0]
                else:
                    department = None
                    for d in eligible:
                        if (p, a, d) in self.Y and self.solution.get_value(self.Y[p, a, d]) > 0.5:
                            department = d
                            break
                
                if start_time is not None:
                    schedule_data.append({
                        'Project': p,
                        'Activity': a,
                        'Department': department,
                        'Start': start_time,
                        'Duration': act['duration'],
                        'Finish': start_time + act['duration'],
                        'Manpower': act['manpower_allocated'],
                        'Due_Date': self.data.get_project_due_date(p),
                        'Tardiness': self.solution.get_value(self.T[p])
                    })
        
        return pd.DataFrame(schedule_data)


def validate_schedule(schedule_df: pd.DataFrame, data: InputData) -> bool:
    """Validate that the schedule satisfies all precedence constraints (same-project and cross-project).
    Returns True if valid, False otherwise. Prints any violations."""
    if schedule_df is None or len(schedule_df) == 0:
        return True
    # Build lookup: (Project, Activity) -> (Start, Duration, Finish)
    schedule_lookup = {}
    for _, row in schedule_df.iterrows():
        key = (row['Project'], row['Activity'])
        schedule_lookup[key] = (float(row['Start']), float(row['Duration']), float(row['Finish']))
    all_ok = True
    for p, pred, succ, lag in data.precedence:
        pred_project = data.resolve_predecessor_project(p, pred)
        if pred_project is None:
            continue
        key_pred = (pred_project, pred)
        key_succ = (p, succ)
        if key_pred not in schedule_lookup or key_succ not in schedule_lookup:
            continue
        _, _, pred_finish = schedule_lookup[key_pred]
        succ_start, _, _ = schedule_lookup[key_succ]
        required_start = pred_finish + lag
        if succ_start < required_start - 1e-6:
            print(f"  [VALIDATION] Precedence violation: {pred_project}/{pred} (finish {pred_finish}) "
                  f"-> {p}/{succ} (start {succ_start}); required start >= {required_start}")
            all_ok = False
    if all_ok:
        print("  [VALIDATION] All precedence constraints satisfied.")
    return all_ok


# ============================================================================
# VISUALIZATION
# ============================================================================

def export_gantt_chart(
    schedule_df: pd.DataFrame,
    filename: str = 'gantt_chart.png',
    precedence: Optional[List[Tuple[str, str, str, float]]] = None,
):
    """Export Gantt chart with department labels on bars (no separate table)."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("Matplotlib not available - skipping Gantt chart")
        return
    
    if schedule_df is None or len(schedule_df) == 0:
        print("No schedule data to plot")
        return
    
    # Sort by project and start time
    plot_df = schedule_df.sort_values(['Project', 'Start']).reset_index(drop=True)
    
    # ------------------------------------------------------------------
    # LAYOUT: SINGLE GANTT AXIS (no table)
    # ------------------------------------------------------------------
    n_rows = len(plot_df)
    fig_height = max(8, n_rows * 0.45)
    fig, ax_gantt = plt.subplots(figsize=(16, fig_height))
    
    # -------------------------
    # GANTT BARS
    # -------------------------
    dept_colors = {
        'PMU': '#4F81BD',
        'ENG': '#9BBB59',
        'SC':  '#F79646',
        'QC':  '#C0504D',
        'LOG': '#8064A2',
        'SITE':'#6B7280',
    }
    color_map = {}
    for dept in plot_df['Department'].unique():
        color_map[dept] = dept_colors.get(dept, '#6B7280')
    
    activity_y = {}
    y_positions = []
    for idx, (_, row) in enumerate(plot_df.iterrows()):
        y = n_rows - idx
        activity_y[(row['Project'], row['Activity'])] = y
        y_positions.append(y)
        
        dept = row['Department'] if row['Department'] else 'N/A'
        bar_color = color_map.get(dept, '#6B7280')
        duration = row['Finish'] - row['Start']
        ax_gantt.barh(
            y,
            duration,
            left=row['Start'],
            height=0.6,
            color=bar_color,
            edgecolor='black',
            linewidth=1.0,
        )
        
        # --- Department label inside the bar (if wide enough) or just after ---
        bar_center_x = row['Start'] + duration / 2
        label = dept
        if duration >= 3:
            ax_gantt.text(
                bar_center_x, y,
                label,
                ha='center', va='center',
                fontsize=6.5, fontweight='bold',
                color='white',
                clip_on=True,
            )
        else:
            # Narrow bar: place label just to the right
            ax_gantt.text(
                row['Finish'] + 0.4, y,
                label,
                ha='left', va='center',
                fontsize=6.5, fontweight='bold',
                color=bar_color,
                clip_on=True,
            )
    
    # Dependency arrows if precedence information is provided
    if precedence:
        for p, pred, succ, lag in precedence:
            succ_row = plot_df[(plot_df['Project'] == p) & (plot_df['Activity'] == succ)]
            pred_row = plot_df[plot_df['Activity'] == pred]
            if pred_row.empty or succ_row.empty:
                continue
            succ_start = float(succ_row['Start'].iloc[0])
            # If multiple rows with same activity name (e.g. different projects), pick predecessor that finishes before successor starts
            pred_finishes = pred_row.apply(lambda r: (r['Project'], r['Start'] + r['Duration']), axis=1)
            valid_preds = [(proj, fin) for proj, fin in pred_finishes if fin <= succ_start + 1e-6]
            if valid_preds:
                pred_project = max(valid_preds, key=lambda x: x[1])[0]
            else:
                pred_project = pred_row['Project'].iloc[0]
            pred_row = pred_row[pred_row['Project'] == pred_project]
            if pred_row.empty:
                continue
            key_pred = (pred_project, pred)
            key_succ = (p, succ)
            if key_pred not in activity_y or key_succ not in activity_y:
                continue

            y0 = activity_y[key_pred]
            y1 = activity_y[key_succ]
            pred_start = float(pred_row['Start'].iloc[0])
            pred_dur = float(pred_row['Duration'].iloc[0])

            x0 = pred_start + pred_dur
            x1 = succ_start

            # Elbow connector: horizontal → vertical → horizontal
            mid_x = (x0 + x1) / 2
            ax_gantt.plot(
                [x0, mid_x, mid_x, x1],
                [y0, y0, y1, y1],
                color='#444444', linewidth=0.8, zorder=3,
            )
            ax_gantt.annotate(
                '',
                xy=(x1, y1),
                xytext=(x1 - 1e-6, y1),
                arrowprops={
                    'arrowstyle': '-|>',
                    'color': '#444444',
                    'linewidth': 0.8,
                },
            )
    
    # Y axis labels - "Project / Activity" for clarity
    ax_gantt.set_yticks(y_positions)
    ax_gantt.set_yticklabels(
        [f"{row['Project']} / {row['Activity']}" for _, row in plot_df.iterrows()],
        fontsize=8,
    )
    ax_gantt.invert_yaxis()
    ax_gantt.set_xlabel('Time (days)', fontsize=11, fontweight='bold')
    ax_gantt.set_ylabel('Project / Task', fontsize=11, fontweight='bold')
    
    max_finish = float(plot_df['Finish'].max())
    ax_gantt.set_xlim(-2, max_finish * 1.1)
    ax_gantt.grid(axis='x', linestyle='--', alpha=0.3, linewidth=0.8)

    # Department legend
    dept_full_names = {
        'PMU': 'PMU – Project Mgmt Unit',
        'ENG': 'ENG – Engineering',
        'SC':  'SC – Supply Chain',
        'QC':  'QC – Quality Control',
        'LOG': 'LOG – Logistics',
        'SITE':'SITE – Site Operations',
    }
    legend_patches = [
        mpatches.Patch(
            color=color_map[dept],
            label=dept_full_names.get(dept, dept)
        )
        for dept in sorted(color_map.keys())
    ]
    ax_gantt.legend(
        handles=legend_patches,
        title='Departments',
        title_fontsize=9,
        fontsize=8,
        loc='lower right',
        framealpha=0.9,
        edgecolor='#CCCCCC',
    )
    
    # Title on whole figure
    if 'Tardiness' in plot_df.columns:
        tardiness_values = plot_df.groupby('Project')['Tardiness'].first()
        total_tardiness = tardiness_values.fillna(0).sum()
    else:
        total_tardiness = 0.0
    fig.suptitle(
        'EPC Project Schedule\n'
        f'Total tardiness: {total_tardiness:.1f} days  |  '
        f'Makespan: {max_finish:.0f} days',
        fontsize=14,
        fontweight='bold',
    )
    
    # Project deadlines as vertical lines on Gantt
    if 'Due_Date' in plot_df.columns:
        due_dates_shown = set()
        for project in sorted(plot_df['Project'].unique()):
            proj_rows = plot_df[plot_df['Project'] == project]
            if proj_rows.empty:
                continue
            due_date = proj_rows['Due_Date'].iloc[0]
            if pd.isna(due_date) or due_date in due_dates_shown:
                continue
            completion = float(proj_rows['Finish'].max())
            is_late = completion > due_date
            line_color = '#C00000' if is_late else '#00A65A'
            ax_gantt.axvline(
                x=due_date,
                color=line_color,
                linestyle='--',
                linewidth=2.0,
                alpha=0.9,
                zorder=5,
            )
            due_dates_shown.add(due_date)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ Enhanced Gantt chart exported: {filename}")


# ============================================================================

def run_optimization(excel_file: str = None, time_limit: int = 6000,
                     output_prefix: str = 'results', export_gantt: bool = True):
    """Main optimization function"""
    
    print("\n" + "="*80)
    print("EPC PROJECT OPTIMIZATION - MODEL A: DAP/MILP")
    print("="*80)
    
    start_time = time.perf_counter()
    
    # Load data
    try:
        data = InputData(excel_file)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None
    
    # Build and solve model
    print("\n" + "="*80)
    print("SOLVING MODEL A")
    print("="*80)
    
    model = ModelA_DAP(data)
    model.build_model()
    
    solve_start = time.perf_counter()
    if not model.solve(time_limit=time_limit):
        print(f"❌ No solution found (time: {time.perf_counter() - solve_start:.2f}s)")
        return None, None
    
    solve_time = time.perf_counter() - solve_start

    # Export results
    schedule = model.get_schedule()
    model.print_solution(schedule_df=schedule)  # pass schedule for accurate utilisation metrics
    if schedule is not None:
        validate_schedule(schedule, data)
        schedule.to_excel(f'{output_prefix}_schedule.xlsx', index=False)
        if export_gantt:
            export_gantt_chart(schedule, f'{output_prefix}_gantt.png', model.data.precedence)
    
    total_time = time.perf_counter() - start_time
    
    print("\n" + "="*80)
    print("✓ OPTIMIZATION COMPLETE!")
    print(f"Solver time: {solve_time:.2f}s | Total time: {total_time:.2f}s")
    print("="*80)
    
    return model, schedule


if __name__ == "__main__":
    run_optimization(excel_file='Sample data 2,2,10.xlsx', time_limit=3000, output_prefix='results')
