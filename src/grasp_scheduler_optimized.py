"""
GRASP-based scheduler for Resource-Constrained Project Scheduling
ADAPTED FOR NEW EXCEL FORMAT
Objective: Minimize Total Tardiness Z = Σ_{p∈P} max(0, C_p - DD_p)
"""

from __future__ import annotations
import numpy as np
import random
import logging
import time
import sys
import os
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Configuration
MAX_ITERATION = 2000
TIME_LIMIT = 28800

# Logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True)
class Activity:
    """Project activity with resource requirements"""
    id: str
    duration: int
    manpower_allocated: float
    predecessors: List[str] = field(default_factory=list)
    eligible_departments: List[str] = field(default_factory=list)

    @property
    def workload_rate(self) -> float:
        return self.manpower_allocated


@dataclass(frozen=True)
class Department:
    """Resource pool with capacity"""
    name: str
    capacity_per_period: float


@dataclass
class ActivityAssignment:
    """Activity scheduling assignment"""
    activity_id: str
    start_time: int
    department: str


@dataclass
class Schedule:
    """Complete project schedule"""
    assignments: Dict[str, ActivityAssignment] = field(default_factory=dict)
    activities: Optional[Dict[str, Activity]] = None
    departments: Optional[Dict[str, Department]] = None
    precedence_lags: Optional[Dict[Tuple[str, str], int]] = None
    project_due_dates: Optional[Dict[str, float]] = None
    project_weights: Optional[Dict[str, float]] = None

    def add(self, assignment: ActivityAssignment) -> None:
        self.assignments[assignment.activity_id] = assignment

    def start_time(self, activity_id: str) -> Optional[int]:
        assignment = self.assignments.get(activity_id)
        return assignment.start_time if assignment else None

    def objective(self) -> float:
        """Calculate weighted total tardiness"""
        if not self.assignments or not self.activities:
            return float('inf')
        return calculate_total_tardiness(
            self, self.activities, self.project_due_dates, self.project_weights
        )

    def is_feasible(self) -> bool:
        """Check constraint satisfaction"""
        if not self.assignments or not self.activities or not self.departments:
            return False
        return is_feasible_schedule(self, self.activities, self.departments, self.precedence_lags)

    def clone(self) -> Schedule:
        return Schedule(
            assignments=dict(self.assignments),
            activities=self.activities,
            departments=self.departments,
            precedence_lags=self.precedence_lags,
            project_due_dates=self.project_due_dates,
            project_weights=self.project_weights,
        )


# ============================================================================
# DATA LOADING - ADAPTED FOR NEW EXCEL FORMAT
# ============================================================================

def load_standardized_excel(excel_file: str) -> Dict:
    """Load Excel data - NEW FORMAT"""
    if not os.path.exists(excel_file):
        raise FileNotFoundError(f"Excel file not found: {excel_file}")
    
    raw_data = {'departments': {}, 'activities': {}, 'precedence': [], 'settings': {}, 'projects': {}}
    
    # Departments
    df_depts = pd.read_excel(excel_file, sheet_name='Departments')
    
    for _, row in df_depts.iterrows():
        dept_id = str(row['Department_ID']).strip()
        raw_data['departments'][dept_id] = {
            'capacity': float(row['Capacity']) if pd.notna(row['Capacity']) else 0.0,
            'cost': float(row['Hourly_Cost']) if pd.notna(row['Hourly_Cost']) else 0.0
        }
    
    # Activities - NEW FORMAT with embedded predecessor and due date
    df_activities = pd.read_excel(excel_file, sheet_name='Activities')
    
    for _, row in df_activities.iterrows():
        project = str(row['Project_ID']).strip()
        activity = str(row['Activity_ID']).strip()
        
        # Resource_ID is the assigned department
        resource_id = str(row['Resource_ID']).strip() if pd.notna(row['Resource_ID']) else ''
        eligible_depts = [resource_id] if resource_id else []
        
        raw_data['activities'][(project, activity)] = {
            'duration': float(row['Duration']) if pd.notna(row['Duration']) else 0.0,
            'manpower_allocated': float(row['Manpower_Qty']) if pd.notna(row['Manpower_Qty']) else 0.0,
            'eligible_depts': eligible_depts
        }
        
        # Extract precedence from Predecessor column
        if pd.notna(row['Predecessor']):
            for predecessor in str(row['Predecessor']).strip().split(','):
                predecessor = predecessor.strip()
                if predecessor:
                    raw_data['precedence'].append((project, predecessor, activity, 0.0))
        
        # Extract project due dates (guard against non-numeric values)
        if pd.notna(row['Due_Date']):
            try:
                due_date = float(row['Due_Date'])
            except (TypeError, ValueError):
                # Skip rows where Due_Date is not numeric (e.g., text like 'PMU')
                due_date = None
            if due_date is not None:
                if project not in raw_data['projects']:
                    raw_data['projects'][project] = {'due_date': due_date}
                else:
                    raw_data['projects'][project]['due_date'] = min(
                        raw_data['projects'][project]['due_date'], due_date
                    )
    
    # Settings
    df_settings = pd.read_excel(excel_file, sheet_name='Settings')
    for _, row in df_settings.iterrows():
        param = str(row['Parameters']).strip().lower()
        raw_data['settings'][param] = row['Value']
    
    return raw_data



def load_project_data_adapter(excel_path: str):
    """Adapt Excel data to GRASP data structures"""
    raw = load_standardized_excel(excel_path)
    
    # Load project due dates
    project_due_dates = {}
    if 'projects' in raw and raw['projects']:
        for project_id, data in raw['projects'].items():
            project_due_dates[project_id] = data['due_date']
    else:
        horizon = int(raw['settings'].get('horizon', 120))
        for (p, a) in raw['activities'].keys():
            if p not in project_due_dates:
                project_due_dates[p] = float(horizon)
    # After loading horizon from settings, add this guard:
    horizon = int(raw['settings'].get('horizon', 150))

    # Auto-scale if too small for the data:
    total_site_mandays = sum(
        data['manpower_allocated'] * data['duration']
        for data in raw['activities'].values()
        if 'SITE' in data['eligible_depts']
    )
    
    site_cap = raw['departments'].get('SITE', {}).get('capacity', 12)
    min_required = int(total_site_mandays / site_cap) + 300  # 300 day buffer
    if horizon < min_required:  
        print(f"[Warning] Horizon {horizon} too small, auto-scaling to {min_required}")
        horizon = min_required
       
    # Determine required capacities
    dept_max_requirement = defaultdict(float)
    parsed_activities = []
    
    for (p, a), data in raw['activities'].items():
        # Normalize activity ID
        if a.startswith(f"{p}_"):
            act_id = a.replace("_", "-", 1)
        elif not a.startswith(f"{p}-"):
            act_id = f"{p}-{a}"
        else:
            act_id = a
        
        duration = int(data['duration'])
        manpower = float(data['manpower_allocated'])
        eligible = data['eligible_depts']
        
        # Track max requirement per dept
        for dept in eligible:
            dept_max_requirement[dept] = max(dept_max_requirement[dept], manpower)
        
        parsed_activities.append({
            'id': act_id,
            'duration': duration,
            'manpower': manpower,
            'eligible_depts': eligible
        })
    
    # Build precedence relationships
    precedence_map = defaultdict(list)
    precedence_lags = {}
    
    for p, pred, succ, lag in raw['precedence']:
        # Normalize IDs
        if pred.startswith(f"{p}_"):
            pred_id = pred.replace("_", "-", 1)
        elif not pred.startswith(f"{p}-"):
            pred_id = f"{p}-{pred}"
        else:
            pred_id = pred
        
        if succ.startswith(f"{p}_"):
            succ_id = succ.replace("_", "-", 1)
        elif not succ.startswith(f"{p}-"):
            succ_id = f"{p}-{succ}"
        else:
            succ_id = succ
        
        precedence_map[succ_id].append(pred_id)
        if lag > 0:
            precedence_lags[(pred_id, succ_id)] = int(lag)
    
    # Create Activity objects
    activities = {}
    for act_data in parsed_activities:
        act_id = act_data['id']
        predecessors = precedence_map.get(act_id, [])
        
        activities[act_id] = Activity(
            id=act_id,
            duration=act_data['duration'],
            manpower_allocated=act_data['manpower'],
            predecessors=predecessors,
            eligible_departments=act_data['eligible_depts']
        )
    
    # Create Department objects — use capacity exactly as specified in Excel
    departments = {}
    for dept_id, dept_data in raw['departments'].items():
        departments[dept_id] = Department(
            name=dept_id,
            capacity_per_period=dept_data['capacity']
        )
    
    # Load project weights from Settings (e.g. P1_weight, P2_weight …)
    project_weights = {}
    for p in project_due_dates:
        key = f'{p.lower()}_weight'
        project_weights[p] = float(raw['settings'].get(key, 1.0))

    horizon = int(raw['settings'].get('horizon', 150))

    return activities, departments, precedence_lags, project_due_dates, project_weights, horizon



# ============================================================================
# FEASIBILITY CHECKING
# ============================================================================

def is_feasible_schedule(schedule: Schedule, activities: Dict[str, Activity],
                        departments: Dict[str, Department],
                        precedence_lags: Dict[Tuple[str, str], int]) -> bool:
    """Verify schedule feasibility"""
    
    # Check precedence constraints
    for activity_id, activity in activities.items():
        assignment = schedule.assignments.get(activity_id)
        if not assignment:
            return False
        
        for pred_id in activity.predecessors:
            pred_assignment = schedule.assignments.get(pred_id)
            if not pred_assignment:
                return False
            
            pred_activity = activities[pred_id]
            pred_finish = pred_assignment.start_time + pred_activity.duration
            lag = precedence_lags.get((pred_id, activity_id), 0)
            
            if assignment.start_time < pred_finish + lag:
                return False
    
    # Check resource capacity constraints
    time_usage = defaultdict(lambda: defaultdict(float))
    
    for activity_id, assignment in schedule.assignments.items():
        activity = activities[activity_id]
        dept = assignment.department
        
        for t in range(assignment.start_time, assignment.start_time + activity.duration):
            time_usage[dept][t] += activity.workload_rate
    
    for dept_name, dept in departments.items():
        for t, usage in time_usage[dept_name].items():
            if usage > dept.capacity_per_period + 1e-6:
                return False
    
    return True


def calculate_total_tardiness(schedule: Schedule, activities: Dict[str, Activity],
                              project_due_dates: Dict[str, float],
                              project_weights: Optional[Dict[str, float]] = None) -> float:
    """Calculate objective: weighted total tardiness across all projects"""
    if not project_due_dates:
        return 0.0

    project_completions = {}

    for activity_id, assignment in schedule.assignments.items():
        project = activity_id.split('-')[0] if '-' in activity_id else "Other"
        activity = activities[activity_id]
        completion = assignment.start_time + activity.duration

        if project not in project_completions:
            project_completions[project] = completion
        else:
            project_completions[project] = max(project_completions[project], completion)

    total_tardiness = 0.0
    for project, completion in project_completions.items():
        due_date = project_due_dates.get(project, float('inf'))
        weight = project_weights.get(project, 1.0) if project_weights else 1.0
        tardiness = max(0.0, completion - due_date)
        total_tardiness += weight * tardiness

    return total_tardiness


# ============================================================================
# GRASP ALGORITHM
# ============================================================================

def compute_earliest_start(activity: Activity, schedule: Schedule,
                           activities: Dict[str, Activity],
                           precedence_lags: Dict[Tuple[str, str], int]) -> int:
    """Calculate earliest start time for an activity"""
    earliest = 0
    
    for pred_id in activity.predecessors:
        pred_assignment = schedule.assignments.get(pred_id)
        if pred_assignment:
            pred_activity = activities[pred_id]
            pred_finish = pred_assignment.start_time + pred_activity.duration
            lag = precedence_lags.get((pred_id, activity.id), 0)
            earliest = max(earliest, pred_finish + lag)
    
    return earliest


def find_feasible_slot(activity: Activity, department: Department,
                      earliest_start: int, schedule: Schedule,
                      activities: Dict[str, Activity],
                      horizon: int = 150) -> Optional[int]:
    """Find earliest feasible start time for activity in department.
    horizon must match the planning horizon from Settings — do not leave as default."""
    
    # Build resource usage timeline
    usage = [0.0] * horizon
    
    for aid, assignment in schedule.assignments.items():
        if assignment.department == department.name:
            act = activities[aid]
            for t in range(assignment.start_time, min(assignment.start_time + act.duration, horizon)):
                usage[t] += act.workload_rate
    
    # Search for feasible slot
    for t in range(earliest_start, horizon):
        feasible = True
        for offset in range(activity.duration):
            if t + offset >= horizon:
                feasible = False
                break
            if usage[t + offset] + activity.workload_rate > department.capacity_per_period + 1e-6:
                feasible = False
                break
        
        if feasible:
            return t
    
    return None


def greedy_randomized_construction(activities: Dict[str, Activity],
                                  departments: Dict[str, Department],
                                  precedence_lags: Dict[Tuple[str, str], int],
                                  project_due_dates: Dict[str, float],
                                  project_weights: Optional[Dict[str, float]] = None,
                                  alpha: float = 0.2,
                                  horizon: int = 150) -> Schedule:
    """GRASP construction phase"""
    
    schedule = Schedule(
        activities=activities,
        departments=departments,
        precedence_lags=precedence_lags,
        project_due_dates=project_due_dates,
        project_weights=project_weights,
    )
    
    # Build dependency graph
    in_degree = {aid: len(act.predecessors) for aid, act in activities.items()}
    ready_activities = [aid for aid, deg in in_degree.items() if deg == 0]
    scheduled = set()
    
    while ready_activities:
        # Compute priority for each ready activity
        candidates = []
        
        for activity_id in ready_activities:
            activity = activities[activity_id]
            earliest = compute_earliest_start(activity, schedule, activities, precedence_lags)
            
            # Try all eligible departments
            best_start = float('inf')
            best_dept = None
            
            for dept_name in activity.eligible_departments:
                dept = departments[dept_name]
                start = find_feasible_slot(
                    activity, dept, earliest, schedule, activities, horizon
                )
                
                if start is not None and start < best_start:
                    best_start = start
                    best_dept = dept_name
            
            if best_dept is not None:
                project = activity_id.split('-')[0] if '-' in activity_id else "Other"
                due_date = project_due_dates.get(project, float(horizon))
                weight = project_weights.get(project, 1.0) if project_weights else 1.0
                # Priority: earliest weighted finish relative to due date
                # Higher weight + closer due date = lower (better) priority score
                priority = best_start / weight + (due_date - best_start) * 0.1 / weight
                
                candidates.append((activity_id, best_start, best_dept, priority))
        
        if not candidates:
            break
        
        # Select from RCL (Restricted Candidate List)
        candidates.sort(key=lambda x: x[3])
        min_priority = candidates[0][3]
        max_priority = candidates[-1][3]
        threshold = min_priority + alpha * (max_priority - min_priority)
        
        rcl = [c for c in candidates if c[3] <= threshold]
        chosen = random.choice(rcl)
        
        activity_id, start_time, dept_name, _ = chosen
        
        # Schedule the activity
        schedule.add(ActivityAssignment(activity_id, start_time, dept_name))
        scheduled.add(activity_id)
        ready_activities.remove(activity_id)
        
        # Update ready list
        for aid, act in activities.items():
            if aid not in scheduled and aid not in ready_activities:
                if all(pred in scheduled for pred in act.predecessors):
                    ready_activities.append(aid)
    
    return schedule


def local_search(schedule: Schedule, activities: Dict[str, Activity],
                departments: Dict[str, Department],
                precedence_lags: Dict[Tuple[str, str], int],
                horizon: int = 150) -> Schedule:
    """Local search: shift activities earlier OR swap resource priority between projects.
    Also tries delaying low-weight activities to free resources for high-weight ones."""
    
    current = schedule
    improved = True
    max_iterations = 50
    iteration = 0
    project_weights = current.project_weights or {}
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        activity_list = list(current.assignments.keys())
        random.shuffle(activity_list)
        
        for activity_id in activity_list:
            activity = activities[activity_id]
            current_assignment = current.assignments[activity_id]
            earliest = compute_earliest_start(activity, current, activities, precedence_lags)
            
            if current_assignment.start_time > earliest:
                dept = departments[current_assignment.department]
                new_start = find_feasible_slot(
                    activity, dept, earliest, current, activities, horizon
                )
                
                if new_start is not None and new_start < current_assignment.start_time:
                    test_schedule = current.clone()
                    test_schedule.assignments[activity_id] = ActivityAssignment(
                        activity_id, new_start, current_assignment.department
                    )
                    
                    if test_schedule.is_feasible() and test_schedule.objective() < current.objective():
                        current = test_schedule
                        improved = True
    
    return current


def launch(activities: Dict[str, Activity],
          departments: Dict[str, Department],
          precedence_lags: Dict[Tuple[str, str], int],
          project_due_dates: Dict[str, float],
          project_weights: Optional[Dict[str, float]] = None,
          alpha: float = 0.2,
          horizon: int = 150) -> Tuple[Optional[Schedule], float, int]:
    """Main GRASP loop"""
    
    best_schedule = None
    best_objective = float('inf')
    start_time = time.perf_counter()
    iteration = 0
    
    print(f"\nStarting GRASP with alpha={alpha}, horizon={horizon}")
    print(f"Activities: {len(activities)}, Departments: {len(departments)}")
    
    while iteration < MAX_ITERATION:
        iteration += 1
        elapsed = time.perf_counter() - start_time
        
        if elapsed > TIME_LIMIT:
            print(f"Time limit reached: {elapsed:.2f}s")
            break
        
        # Construction phase
        schedule = greedy_randomized_construction(
            activities, departments, precedence_lags, project_due_dates,
            project_weights, alpha, horizon
        )
        
        if not schedule.is_feasible():
            continue
        
        # Local search phase
        schedule = local_search(schedule, activities, departments, precedence_lags, horizon)
        
        # Update best
        obj = schedule.objective()
        if obj < best_objective:
            best_objective = obj
            best_schedule = schedule
            print(f"Iteration {iteration}: New best = {obj:.2f}")
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: Best = {best_objective:.2f}, Time = {elapsed:.2f}s")
    
    runtime = time.perf_counter() - start_time
    return best_schedule, runtime, iteration


# ============================================================================
# VISUALIZATION
# ============================================================================

def export_gantt_chart(schedule: Schedule, activities: Dict[str, Activity],
                      filename: str = 'grasp_gantt.png',
                      project_due_dates: Optional[Dict[str, float]] = None) -> bool:
    """Export Gantt chart in a style similar to MS Project views."""
    
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available")
        return False
    
    if not schedule or not schedule.assignments:
        print("No schedule to plot")
        return False
    
    # ------------------------------------------------------------------
    # PREPARE DATA
    # ------------------------------------------------------------------
    plot_data: List[Dict] = []
    
    for activity_id, assignment in schedule.assignments.items():
        activity = activities[activity_id]
        project = activity_id.split('-')[0] if '-' in activity_id else "Other"
        
        plot_data.append(
            {
                'project': project,
                'activity': activity_id,
                'department': assignment.department,
                'start': assignment.start_time,
                'finish': assignment.start_time + activity.duration,
                'duration': activity.duration,
            }
        )
    
    # Sort by project then start time to mimic the ordering in the examples
    plot_data.sort(key=lambda x: (x['project'], x['start']))
    
    # Compute per-project completion times and max tardiness
    project_completions = {}
    for d in plot_data:
        proj = d['project']
        if proj not in project_completions:
            project_completions[proj] = d['finish']
        else:
            project_completions[proj] = max(project_completions[proj], d['finish'])
    
    max_project_tardiness = 0.0
    if project_due_dates:
        for proj, completion in project_completions.items():
            due_date = project_due_dates.get(proj, float('inf'))
            tardiness = max(0.0, completion - due_date)
            if tardiness > max_project_tardiness:
                max_project_tardiness = tardiness
    
    # ------------------------------------------------------------------
    # LAYOUT: SINGLE GANTT AXIS (no table)
    # ------------------------------------------------------------------
    n_rows = len(plot_data)
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
    color_map: Dict[str, str] = {}
    for d in plot_data:
        dept = d['department']
        if dept not in color_map:
            color_map[dept] = dept_colors.get(dept, '#6B7280')

    # Map activity -> y position (top-down)
    activity_y: Dict[str, float] = {}
    y_positions = []
    for idx, data in enumerate(plot_data):
        y = n_rows - idx
        activity_y[data['activity']] = y
        y_positions.append(y)

        dept = data['department']
        bar_color = color_map[dept]
        duration = data['finish'] - data['start']

        ax_gantt.barh(
            y,
            duration,
            left=data['start'],
            height=0.6,
            color=bar_color,
            edgecolor='black',
            linewidth=1.0,
        )

        # Department label: inside bar if wide enough, else just to the right
        if duration >= 3:
            ax_gantt.text(
                data['start'] + duration / 2, y,
                dept,
                ha='center', va='center',
                fontsize=6.5, fontweight='bold',
                color='white',
                clip_on=True,
            )
        else:
            ax_gantt.text(
                data['finish'] + 0.4, y,
                dept,
                ha='left', va='center',
                fontsize=6.5, fontweight='bold',
                color=bar_color,
                clip_on=True,
            )

    # -------------------------
    # PERPENDICULAR DEPENDENCY CONNECTORS
    # -------------------------
    for succ_id, activity in activities.items():
        succ_assignment = schedule.assignments.get(succ_id)
        if not succ_assignment:
            continue
        for pred_id in activity.predecessors:
            pred_assignment = schedule.assignments.get(pred_id)
            if not pred_assignment:
                continue
            pred_act = activities[pred_id]
            x0 = pred_assignment.start_time + pred_act.duration
            y0 = activity_y.get(pred_id)
            x1 = succ_assignment.start_time
            y1 = activity_y.get(succ_id)
            if y0 is None or y1 is None:
                continue

            # Elbow connector: horizontal → vertical → horizontal
            mid_x = (x0 + x1) / 2
            ax_gantt.plot(
                [x0, mid_x, mid_x, x1],
                [y0, y0, y1, y1],
                color='#444444', linewidth=0.8, zorder=3,
            )
            # Arrowhead at destination
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

    # -------------------------
    # Y-AXIS & AXES LABELS
    # -------------------------
    ax_gantt.set_yticks(y_positions)
    ax_gantt.set_yticklabels(
        [f"{d['project']} / {d['activity']}" for d in plot_data],
        fontsize=8,
    )
    ax_gantt.invert_yaxis()
    ax_gantt.set_xlabel('Time (days)', fontsize=11, fontweight='bold')
    ax_gantt.set_ylabel('Project / Task', fontsize=11, fontweight='bold')

    max_finish = max(d['finish'] for d in plot_data)
    ax_gantt.set_xlim(-2, max_finish * 1.1)
    ax_gantt.grid(axis='x', linestyle='--', alpha=0.3, linewidth=0.8)

    # -------------------------
    # DEPARTMENT LEGEND
    # -------------------------
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
            label=dept_full_names.get(dept, dept),
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
    
    # Title on the whole figure
    total_tardiness = schedule.objective()
    fig.suptitle(
        'GRASP Heuristic Schedule\n'
        f'Total tardiness: {total_tardiness:.1f} days  |  '
        f'Max project tardiness: {max_project_tardiness:.1f} days  |  '
        f'Makespan: {max_finish} days',
        fontsize=14,
        fontweight='bold',
    )
    
    # Project deadlines (vertical lines on Gantt area)
    if project_due_dates:
        projects = {}
        for d in plot_data:
            proj = d['project']
            if proj not in projects:
                projects[proj] = d['finish']
            else:
                projects[proj] = max(projects[proj], d['finish'])
        
        due_dates_shown = set()
        for project, due_date in sorted(project_due_dates.items()):
            if due_date in due_dates_shown:
                continue
            completion = projects.get(project, 0)
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
    return True


# ============================================================================
# REPORTING
# ============================================================================

def department_utilization_over_makespan(
    schedule: Schedule,
    activities: Dict[str, Activity],
    dept_name: str,
    capacity: float,
    makespan: float,
) -> float:
    """
    Utilisation (%) = total manpower-days assigned to dept / (capacity × makespan).
    This measures how much of the department's total available capacity over the
    full schedule horizon was actually consumed by work.
    """
    if capacity <= 0 or makespan <= 0:
        return 0.0

    total_work = sum(
        activities[aid].duration * activities[aid].manpower_allocated
        for aid, assignment in schedule.assignments.items()
        if assignment.department == dept_name
    )
    return 100.0 * total_work / (capacity * makespan)


def print_summary(schedule: Schedule, activities: Dict[str, Activity],
                 departments: Dict[str, Department], run_time: float, iterations: int):
    """Print comprehensive summary"""
    print("\n" + "=" * 80)
    print("GRASP SOLUTION SUMMARY")
    print("=" * 80)
    
    total_tardiness = schedule.objective()
    makespan = max(assignment.start_time + activities[activity_id].duration
                   for activity_id, assignment in schedule.assignments.items())
    
    print(f"\n[OVERALL]")
    print(f"   Total Tardiness: {total_tardiness:.2f} days")
    print(f"   Makespan:        {makespan} days")
    print(f"   Runtime:         {run_time:.2f} seconds")
    print(f"   Iterations:      {iterations}")
    print(f"   Feasible:        {'✓' if schedule.is_feasible() else '✗'}")
    
    # Project breakdown
    if schedule.project_due_dates:
        print(f"\n[PROJECTS]")
        print(f"   {'Project':<15} {'Completion':<12} {'Due':<12} {'Tardiness':<12} {'Status'}")
        print("   " + "-" * 65)
        
        projects = defaultdict(list)
        for activity_id, assignment in schedule.assignments.items():
            project = activity_id.split('-')[0] if '-' in activity_id else "Other"
            projects[project].append((activity_id, assignment))
        
        for project in sorted(projects.keys()):
            completion = max(assignment.start_time + activities[aid].duration
                           for aid, assignment in projects[project])
            due_date = schedule.project_due_dates.get(project, float('inf'))
            tardiness = max(0.0, completion - due_date)
            status = "⚠️ LATE" if tardiness > 0.01 else "✅ ON TIME"
            print(f"   {project:<15} {completion:<12.1f} {due_date:<12.1f} {tardiness:<12.1f} {status}")
    
    # Department utilization (total work / capacity × makespan)
    print(f"\n[DEPARTMENT UTILIZATION]")
    print(f"   Utilisation = total manpower-days / (capacity × makespan)")
    print(f"   {'Department':<12} {'Activities':<10} {'Total Manpower':<16} {'Capacity/Period':<16} "
          f"{'Util. %'}")
    print("   " + "-" * 70)

    for dept_name in sorted(departments.keys()):
        dept = departments[dept_name]
        assigned = [
            (aid, assignment)
            for aid, assignment in schedule.assignments.items()
            if assignment.department == dept_name
        ]

        num_activities = len(assigned)
        total_manpower = sum(
            activities[aid].manpower_allocated * activities[aid].duration
            for aid, _ in assigned
        )

        capacity = dept.capacity_per_period
        utilization = department_utilization_over_makespan(
            schedule, activities, dept_name, capacity, makespan
        )

        print(
            f"   {dept_name:<12} {num_activities:<10} "
            f"{total_manpower:<16.1f} {capacity:<16.1f} {utilization:>6.1f} %"
        )

    print("\n" + "=" * 80)


def main():
    """Main execution"""
    script_dir = Path(__file__).parent
    possible_paths = [
        "Sample data 2,2,10.xlsx",
    ]
    
    excel_path = None
    for path in possible_paths:
        if os.path.exists(path):
            excel_path = str(path)
            break
    
    if not excel_path:
        logger.error(f"Excel file not found. Tried: {possible_paths}")
        return
    
    logger.info(f"Loading: {excel_path}")
    activities, departments, precedence_lags, project_due_dates, project_weights, horizon = load_project_data_adapter(excel_path)
    
    logger.info("Starting GRASP optimization")
    best_sol, run_time, iterations = launch(
        activities, departments, precedence_lags, project_due_dates,
        project_weights, alpha=0.2, horizon=horizon
    )
    
    if not best_sol:
        logger.error("No feasible solution found")
        return
    
    print_summary(best_sol, activities, departments, run_time, iterations)
    export_gantt_chart(best_sol, activities, 'grasp_gantt.png', project_due_dates)


if __name__ == "__main__":
    random.seed(0)
    main()
