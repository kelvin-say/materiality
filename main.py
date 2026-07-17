### Start this streamlit app with: streamlit run main_redo.py

### # 1. Save the entire session variables and state
### dill.dump_session('my_session.pkl')

### # 2. Restore the session later (in a new notebook or terminal)
### dill.load_session('my_session.pkl')

#%% ===========================================================================
# Setup libraries
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import datetime
import os.path
import time
import os
import shutil
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import io
import dill

from datetime import datetime, time as datetime_time
from pytz import reference
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CarConfig:
    fuel_type: str                   # 'Petrol', 'Diesel', or 'Electric'
    annual_distance_km: int
    efficiency: float                # L/100km for ICE; kWh/100km for EV
    schedule: Optional[list] = None  # list of time-block dicts for EVs
    charging_strategy: Optional[str] = "Immediately upon return"
    charger_speed: Optional[str] = "Level 1 (2.4 kW)"

# =============================================================================
# Setup constants
# =============================================================================
MONTHS_PER_YEAR = 12
DAYS_PER_YEAR = 365
DAYS_PER_WEEK = 7
HOURS_PER_DAY = 24
MINUTES_PER_HOUR = 60
SECONDS_PER_MINUTE = 60
HOURS_PER_YEAR = DAYS_PER_YEAR * HOURS_PER_DAY
MINUTES_PER_DAY = HOURS_PER_DAY * MINUTES_PER_HOUR
SECONDS_PER_HOUR = MINUTES_PER_HOUR * SECONDS_PER_MINUTE
SECONDS_PER_DAY = HOURS_PER_DAY * SECONDS_PER_HOUR
QUARTERS_PER_YEAR = 4

MONDAY = 0
TUESDAY = 1
WEDNESDAY = 2
THURSDAY = 3
FRIDAY = 4
SATURDAY = 5
SUNDAY = 6

daytostring_dict = {
  MONDAY: 'Monday',
  TUESDAY: 'Tuesday',
  WEDNESDAY: 'Wednesday',
  THURSDAY: 'Thursday',
  FRIDAY: 'Friday',
  SATURDAY: 'Saturday',
  SUNDAY: 'Sunday',
}

CONVERT_KWH_TO_MJ = 3.6
CONVERT_MJ_TO_KWH = 1 / CONVERT_KWH_TO_MJ

# Configuration spreadsheet
# -------------------------
INPUT_FILE = 'input_parameters.xlsx'

#%% ===========================================================================
# Define all the subroutines
# =============================================================================
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def processEnergyStorageDispatch():
    return 1

def logMessage(message, OUTPUT_DIRECTORY):
    print(message)
    f = open(OUTPUT_DIRECTORY + "/output_log.txt", "a+")
    f.write(str(datetime.now().strftime("%a, %d-%b-%Y %I:%M:%S")) + "\n" + str(message) + "\n\n")
    f.close()

def plot(x, title="", xlabel = "", ylabel="", filename=""):
    fig, ax = plt.subplots()
    ax.plot(x)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if filename != "":
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()

def plot2(x1, x2, title="", xlabel = "", ylabel="", labels=["",""], filename=""):
    fig, ax = plt.subplots()
    ax.plot(x1, label=labels[0])
    ax.plot(x2, label=labels[1])
    ax.set_ylim(bottom=0)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if filename != "":
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()
    

#%% ===========================================================================
# Simulation engine
# =============================================================================
# i_show_output = False
def simulatePVB(i_df_energy_ts, i_static_param, i_show_output):
    time_start = time.time()

    timestep_seconds = i_static_param['timestep_seconds']
    convert_energy_to_power = int(SECONDS_PER_HOUR / timestep_seconds)
    convert_power_to_energy = 1 / convert_energy_to_power

    timesteps_per_year = int(HOURS_PER_YEAR * SECONDS_PER_HOUR / timestep_seconds)
    num_timesteps = len(i_df_energy_ts)
    num_years = int(num_timesteps/timesteps_per_year)

    # Setup the exogeneous parameters (units are assumed to be in kWh or kW)
    # ----------------------------------------------------------------------
    # Discount factors
    for y in range(num_years):
        if y == 0:
            exo_discount_factor = np.repeat(1, timesteps_per_year)
        else:
            exo_discount_factor = np.append(exo_discount_factor, np.repeat((1+i_static_param['discount_rate'])**-y,timesteps_per_year))

    # Battery
    exo_batt_aging_cal_perstep = (timestep_seconds / SECONDS_PER_HOUR) / (HOURS_PER_YEAR * i_static_param['bess_eol_years'])

    # PV
    v_pv_aging_cal = np.linspace(1, 1-num_years*(1-i_static_param['pv_eol_capacity'])/i_static_param['pv_eol_years'], num_timesteps, False) - (1-i_static_param['pv_init_soh']) # type: ignore
    v_pv_aging_cal[v_pv_aging_cal < 0] = 0 # Ensure that once PV is EOL, it remains disabled
    exo_E_pv_unitgeneration = v_pv_aging_cal * i_df_energy_ts['E_pvunitgeneration_kWh'] 
    
    # Prices
    exo_C_export = i_df_energy_ts['C_tariff_export_$/kWh']
    exo_C_import = i_df_energy_ts['C_tariff_import_$/kWh']
    exo_C_fixed = i_df_energy_ts['C_tariff_fixed_$/day']
    
    # Existing load, exports and curtailed
    v_E_netload_kWh = i_df_energy_ts['E_netload_kWh']
    exo_E_existing_load = np.where(v_E_netload_kWh >= 0, v_E_netload_kWh, 0)
    exo_E_existing_grid = np.clip(np.where(v_E_netload_kWh < 0, -v_E_netload_kWh, 0), 0, i_static_param['grid_feedin_max'] * convert_power_to_energy)
    exo_E_existing_curtail = np.clip(np.where(v_E_netload_kWh < 0, -v_E_netload_kWh, 0), i_static_param['grid_feedin_max'] * convert_power_to_energy, None) - i_static_param['grid_feedin_max'] * convert_power_to_energy

    # Existing electricity bill timeseries
    exo_C_existing_bill = exo_E_existing_load * exo_C_import - exo_E_existing_grid * exo_C_export + exo_C_fixed

    # Setup the PV capacity (kWp) and search range
    sim_pv_capacity = i_static_param['solar_capacity']
    
    # Setup the battery capacity (kWh) and search range
    sim_batt_capacity = i_static_param['battery_energy_capacity']
    sim_batt_inverter_capacity = i_static_param['battery_power_capacity']


    # Run the PV and BESS simulation model (using integers only to avoid floating point errors)
    # -----------------------------------------------------------------------------------------
    # Determine how much excess self-generation is available to offset the remaining load
    sim_E_underlyingload_Wh = np.round(exo_E_existing_load * 1000).astype(int)
    sim_E_PV_generation_Wh = np.round(exo_E_pv_unitgeneration * sim_pv_capacity * 1000).astype(int) 
    sim_E_existing_excess_generation_Wh = np.round((exo_E_existing_grid + exo_E_existing_curtail) * 1000).astype(int)    
    
    sim_E_initial_residual_load_Wh = sim_E_underlyingload_Wh - sim_E_PV_generation_Wh - sim_E_existing_excess_generation_Wh    
    sim_E_initial_grid_import_Wh = np.where(sim_E_initial_residual_load_Wh > 0, sim_E_initial_residual_load_Wh, 0)
    sim_E_initial_excess_generation_Wh = np.where(sim_E_initial_residual_load_Wh < 0, -sim_E_initial_residual_load_Wh, 0)

    sim_E_max_grid_export_Wh = np.round(i_static_param['grid_feedin_max'] * convert_power_to_energy * 1000).astype(int)

    # Set up the battery operational parameters
    sim_E_batt_charge_Wh = np.zeros(num_timesteps, dtype=int)
    sim_E_batt_discharge_Wh = np.zeros(num_timesteps, dtype=int)
    sim_E_batt_SoC_Wh = np.zeros(num_timesteps, dtype=int)
    sim_E_batt_SoH_Wh = np.round((sim_batt_capacity - np.arange(num_timesteps) * exo_batt_aging_cal_perstep * i_static_param['bess_eol_capacity'] * sim_batt_capacity)*1000).astype(int)
    sim_E_batt_max_charge_Wh = np.round(sim_batt_inverter_capacity * convert_power_to_energy * 1000).astype(int)
    sim_E_batt_max_discharge_Wh = np.round(sim_batt_inverter_capacity * convert_power_to_energy * 1000).astype(int)
    sim_E_batt_inverterlosses_Wh = np.zeros(num_timesteps, dtype=int)
        
    # Setup the remaining grid timeseries after PV self-consumption
    sim_E_final_grid_import_Wh = np.zeros(num_timesteps, dtype=int)
    sim_E_final_grid_export_Wh = np.zeros(num_timesteps, dtype=int)
    sim_E_final_grid_export_curtailed_Wh = np.zeros(num_timesteps, dtype=int)
    sim_E_final_batt_SoH_losses_Wh = np.zeros(num_timesteps, dtype=int)

    # Simulate the battery operation over the analysis period   
    t = 0
    for t in range(num_timesteps):
        
        # Consider any stored energy losses from SoH degradation
        if t > 0:
            if sim_E_batt_SoC_Wh[t-1] > sim_E_batt_SoH_Wh[t]:
                sim_E_final_batt_SoH_losses_Wh[t] = sim_E_batt_SoC_Wh[t-1] - sim_E_batt_SoH_Wh[t]

        # If there is still excess generation (after load), then charge the battery with it before exporting to grid
        if sim_E_initial_excess_generation_Wh[t] > 0:
            excess_energy_Wh = sim_E_initial_excess_generation_Wh[t]
            available_storage_Wh = sim_E_batt_SoH_Wh[t] - sim_E_batt_SoC_Wh[t-1] + sim_E_final_batt_SoH_losses_Wh[t] if t > 0 else sim_E_batt_SoH_Wh[t]
            available_storage_preinverter_Wh = np.round(available_storage_Wh / i_static_param['bess_eff_charge']).astype(int)
            batt_charge_Wh = min(excess_energy_Wh, available_storage_preinverter_Wh, sim_E_batt_max_charge_Wh)
            inverter_losses_Wh = np.round(batt_charge_Wh * (1 - i_static_param['bess_eff_charge'])).astype(int)

            initial_export_energy_Wh = excess_energy_Wh - batt_charge_Wh
            if initial_export_energy_Wh < sim_E_max_grid_export_Wh:
                export_energy_Wh = excess_energy_Wh - batt_charge_Wh
                curtailed_energy_Wh = 0
            else:
                export_energy_Wh = sim_E_max_grid_export_Wh
                curtailed_energy_Wh = initial_export_energy_Wh - sim_E_max_grid_export_Wh

            sim_E_batt_charge_Wh[t] = batt_charge_Wh
            sim_E_batt_SoC_Wh[t] = sim_E_batt_SoC_Wh[t-1] - sim_E_final_batt_SoH_losses_Wh[t] + sim_E_batt_charge_Wh[t] - inverter_losses_Wh if t > 0 else sim_E_batt_charge_Wh[t] - inverter_losses_Wh           
            sim_E_batt_inverterlosses_Wh[t] = inverter_losses_Wh
            sim_E_final_grid_import_Wh[t] = 0
            sim_E_final_grid_export_Wh[t] = export_energy_Wh
            sim_E_final_grid_export_curtailed_Wh[t] = curtailed_energy_Wh
        
        # There is remaining load to be served
        else:
            remaining_load_Wh = sim_E_initial_grid_import_Wh[t]

            # First discharge the battery to meet the load
            available_energy_Wh = sim_E_batt_SoC_Wh[t-1] if t > 0 else 0
            available_energy_atload_Wh = np.round(available_energy_Wh * i_static_param['bess_eff_discharge']).astype(int)
            batt_discharge_Wh = min(remaining_load_Wh, available_energy_atload_Wh, sim_E_batt_max_discharge_Wh)

            remaining_load_Wh_after_batt = remaining_load_Wh - batt_discharge_Wh
            SoC_discharge_Wh = np.round(batt_discharge_Wh / i_static_param['bess_eff_discharge']).astype(int)
            inverter_losses_Wh = SoC_discharge_Wh - batt_discharge_Wh
            
            sim_E_batt_discharge_Wh[t] = batt_discharge_Wh
            sim_E_batt_SoC_Wh[t] = sim_E_batt_SoC_Wh[t-1] - SoC_discharge_Wh if t > 0 else 0
            sim_E_batt_inverterlosses_Wh[t] = inverter_losses_Wh
            sim_E_final_grid_import_Wh[t] = remaining_load_Wh_after_batt
            sim_E_final_grid_export_Wh[t] = 0
            sim_E_final_grid_export_curtailed_Wh[t] = 0
    
    # Check energy balance
    sim_E_pv_load_Wh = np.zeros(num_timesteps, dtype=int)
    for t in range(num_timesteps):
        sim_E_pv_load_Wh[t] = min(sim_E_PV_generation_Wh[t], sim_E_underlyingload_Wh[t])
    valid_pv_energy_balance = (sim_E_PV_generation_Wh.sum() - (sim_E_pv_load_Wh.sum() + sim_E_final_grid_export_Wh.sum() + sim_E_final_grid_export_curtailed_Wh.sum() + sim_E_batt_charge_Wh.sum())) == 0    
    valid_load_energy_balance = (sim_E_underlyingload_Wh - sim_E_pv_load_Wh - sim_E_final_grid_import_Wh - sim_E_batt_discharge_Wh).sum() == 0
#    print("Energy balance check: PV [{pv}] Load [{load}]".format(pv=valid_pv_energy_balance, load=valid_load_energy_balance))

    # Create dataframe with all simulation results
    df_results = pd.DataFrame({
        'E_underlyingload_Wh': sim_E_underlyingload_Wh,
        'E_PV_generation_Wh': sim_E_PV_generation_Wh,
        'E_initial_residual_load_Wh': sim_E_initial_residual_load_Wh,
        'E_initial_grid_import_Wh': sim_E_initial_grid_import_Wh,
        'E_initial_excess_generation_Wh': sim_E_initial_excess_generation_Wh,
        'E_batt_charge_Wh': sim_E_batt_charge_Wh,
        'E_batt_discharge_Wh': sim_E_batt_discharge_Wh,
        'E_batt_SoC_Wh': sim_E_batt_SoC_Wh,
        'E_batt_SoH_Wh': sim_E_batt_SoH_Wh,
        'E_batt_SoH_losses_Wh': sim_E_final_batt_SoH_losses_Wh,
        'E_batt_inverterlosses_Wh': sim_E_batt_inverterlosses_Wh,       
        'E_final_grid_import_Wh': sim_E_final_grid_import_Wh,
        'E_final_grid_export_Wh': sim_E_final_grid_export_Wh,
        'E_final_grid_export_curtailed_Wh': sim_E_final_grid_export_curtailed_Wh
    })
    
    #df_results.to_excel(OUTPUT_DIRECTORY + "/scenario_dispatch_results.xlsx", index=True)
    
    summary = {
        'pv_capacity_kWp': sim_pv_capacity,
        'batt_capacity_kWh': sim_batt_capacity,
        'batt_capacity_kW': sim_batt_inverter_capacity,
        'total_underlyingload_kWh': sim_E_underlyingload_Wh.sum() / 1000,
        'total_pv_generation_kWh': sim_E_PV_generation_Wh.sum() / 1000,
        'total_pv_load_kWh': sim_E_pv_load_Wh.sum() / 1000,
        'total_pv_grid_kWh': sim_E_final_grid_export_Wh.sum() / 1000,
        'total_pv_curtailed_kWh': sim_E_final_grid_export_curtailed_Wh.sum() / 1000,
        'total_pv_batt_kWh': sim_E_batt_charge_Wh.sum() / 1000,
        'total_batt_load_kWh': sim_E_batt_discharge_Wh.sum() / 1000,
        'total_batt_losses_kWh': (sim_E_batt_inverterlosses_Wh + sim_E_final_batt_SoH_losses_Wh).sum() / 1000,
        'total_grid_load_kWh': sim_E_final_grid_import_Wh.sum() / 1000,        
    }
    
    return [df_results, summary]

#%% ===========================================================================
# Word bubbles for system metrics
# =============================================================================
def create_metrics_wordcloud(summary):
    """
    Create a bubble chart visualization based on key system metrics.
    Bubble size represents the magnitude of each metric.
    """
    # Extract metrics
    grid_dependence = summary['total_grid_load_kWh']
    grid_independence = summary['total_underlyingload_kWh'] - grid_dependence
    grid_feedin = summary['total_pv_grid_kWh']
    spilled_pv = summary['total_pv_curtailed_kWh']
    fuel_imports = summary['fuel_imports_kWh']
    gas_imports = summary['gas_imports_kWh']
    public_charging = summary['public_charging_kWh']
    
    # Create data for bubbles
    labels = ['Grid Imports', 'Self-sufficient', 'Grid Exports', 'Spilled', 'Fuel Imports', 'Gas Imports', 'Public Charging']
    values = [grid_dependence, grid_independence, grid_feedin, spilled_pv, fuel_imports, gas_imports, public_charging]
    
    # Define custom colors for each metric
    color_map = {
        'Grid Imports': '#FF6B6B',              # Red - for grid dependence
        'Self-sufficient': '#6BCF7F',  # Green - for self-sufficiency
        'Grid Exports': "#FFD93D",           # Yellow - for exports
        'Spilled': "#EF6C00",            # Orange - for spilled energy
        'Fuel Imports': "#483A2E",            # Brown - for fuel imports
        'Gas Imports': "#496687",             # Blue-grey - for gas imports
        'Public Charging': "#9C6FE4",         # Purple - for public EV charging
    }
    
    # Filter out zero values for better visualization
    data = [(label, value) for label, value in zip(labels, values) if value > 0]
    labels_filtered = [d[0] for d in data]
    values_filtered = [d[1] for d in data]
    
    # Map colors to filtered labels
    colors_filtered = [color_map[label] for label in labels_filtered]
    
    # Calculate bubble radii (normalized for visualization)
    max_value = max(values_filtered) if values_filtered else 1
    radii_dict = {label: np.sqrt(value / max_value) * 2 for label, value in zip(labels_filtered, values_filtered)}
    
    # Get radius for self-sufficiency (center bubble)
    center_radius = radii_dict.get('Self-sufficient', 1)
    
    # Small gap for visual separation
    gap = 0.1
    
    # Define fixed zone positions:
    #   Left side: Grid_Imports, Fuel_Imports, Gas_Imports (stacked vertically)
    #   Center:     Self-sufficient
    #   Right side:  Grid_Exports, Spilled (stacked vertically)
    
    left_labels = ['Grid Imports', 'Fuel Imports', 'Gas Imports', 'Public Charging']
    right_labels  = ['Grid Exports', 'Spilled']
    
    position_map = {}
    
    # Self-sufficient always at center
    position_map['Self-sufficient'] = (0, 0)
    
    # Right side: stack bubbles vertically, touching center bubble
    right_present = [l for l in right_labels if l in radii_dict]
    if right_present:
        # Find the widest bubble on the right to set x baseline
        max_right_radius = max(radii_dict[l] for l in right_present)
        x_right = center_radius + max_right_radius + gap
        
        # Stack vertically: calculate total height needed
        total_height = sum(radii_dict[l] * 2 for l in right_present) + gap * (len(right_present) - 1)
        y_cursor = total_height / 2
        for l in right_present:
            r = radii_dict[l]
            y_cursor -= r
            position_map[l] = (x_right, y_cursor)
            y_cursor -= r + gap
    
    # Left side: stack bubbles vertically, touching center bubble
    left_present = [l for l in left_labels if l in radii_dict]
    if left_present:
        max_left_radius = max(radii_dict[l] for l in left_present)
        x_left = -(center_radius + max_left_radius + gap)
        
        total_height = sum(radii_dict[l] * 2 for l in left_present) + gap * (len(left_present) - 1)
        y_cursor = total_height / 2
        for l in left_present:
            r = radii_dict[l]
            y_cursor -= r
            position_map[l] = (x_left, y_cursor)
            y_cursor -= r + gap
    
    # Build final arrays based on filtered labels with fixed positions
    x_positions = []
    y_positions = []
    marker_sizes = []
    final_labels = []
    final_colors = []
    final_values = []
    
    for label, value, color in zip(labels_filtered, values_filtered, colors_filtered):
        x, y = position_map[label]
        x_positions.append(x)
        y_positions.append(y)
        marker_sizes.append(radii_dict[label] * 80)  # Scale up for visibility
        final_labels.append(label)
        final_colors.append(color)
        final_values.append(value)
    
    # Calculate required height based on bubble extents to prevent overlap
    # Each bubble occupies y +/- radius; find the full vertical span
    pixel_scale = 80  # matches marker_sizes scaling above
    if y_positions:
        y_extents = [abs(y) + radii_dict[l] for y, l in zip(y_positions, final_labels)]
        max_y_extent = max(y_extents)
        # Convert from data units to pixels: each radius unit ~ pixel_scale/2 px
        content_height_px = max_y_extent * pixel_scale + 100  # +100 for margins
        chart_height = max(500, int(content_height_px))
    else:
        chart_height = 500
    
    # Create bubble chart
    fig = go.Figure()
    
    # Add bubbles
    fig.add_trace(go.Scatter(
        x=x_positions,
        y=y_positions,
        mode='markers+text',
        marker=dict(
            size=marker_sizes,
            color=final_colors,
            showscale=False,
            line=dict(width=0.1, color='white'),
            opacity=0.8
        ),
        text=[f"{label}" for label in final_labels],
        textfont=dict(size=12, color='white', family='Arial'),
        textposition='middle center',
        customdata=final_values,
        hovertemplate='<br>%{customdata:.0f} kWh<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        showlegend=False,
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False
        ),
        width=800,
        height=chart_height,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

#%% ===========================================================================
# Energy sankey diagram
# =============================================================================
def create_energy_sankey(df_results, summary, filename=""):

    # Calculate total flows in kWh
    pv_gen = summary['total_pv_generation_kWh'] * 1000
    pv_load = summary['total_pv_load_kWh'] * 1000
    grid_import = summary['total_grid_load_kWh'] * 1000
    grid_export = summary['total_pv_grid_kWh'] * 1000
    curtailed = summary['total_pv_curtailed_kWh'] * 1000
    batt_charge = summary['total_pv_batt_kWh'] * 1000
    batt_discharge = summary['total_batt_load_kWh'] * 1000
    batt_losses = summary['total_batt_losses_kWh'] * 1000
    fuel_imports = summary.get('fuel_imports_kWh', 0) * 1000
    gas_imports  = summary.get('gas_imports_kWh', 0) * 1000
    public_charging = summary.get('public_charging_kWh', 0) * 1000

    # Calculate load (what was consumed)
    load = summary['total_underlyingload_kWh']

    # Define all possible nodes (x fixed, y computed dynamically below)
    all_nodes = [
        {'id': 0, 'label': 'Grid Imports',            'color': '#FF6B6B', 'x': 0.01, 'side': 'left',   'flow': grid_import},
        {'id': 1, 'label': 'PV Generation',           'color': '#FFD93D', 'x': 0.01, 'side': 'left',   'flow': pv_gen},
        {'id': 2, 'label': 'Battery Storage',         'color': '#6BCF7F', 'x': 0.5,  'side': 'middle', 'flow': batt_charge},
        {'id': 3, 'label': 'Electricity Consumption', 'color': '#4ECDC4', 'x': 0.99, 'side': 'right',  'flow': grid_import + pv_load + batt_discharge + public_charging},
        {'id': 4, 'label': 'Grid Exports',            'color': '#FFD93D', 'x': 0.99, 'side': 'right',  'flow': grid_export},
        {'id': 5, 'label': 'Spilled PV',              'color': '#EF6C00', 'x': 0.99, 'side': 'right',  'flow': curtailed},
        {'id': 6, 'label': 'Fuel Imports',            'color': '#483A2E', 'x': 0.01, 'side': 'left',   'flow': fuel_imports},
        {'id': 7, 'label': 'Transport',               'color': '#7B5E4A', 'x': 0.99, 'side': 'right',  'flow': fuel_imports},
        {'id': 8, 'label': 'Gas Imports',             'color': '#3C81D0', 'x': 0.01, 'side': 'left',   'flow': gas_imports},
        {'id': 9, 'label': 'Gas Appliances',          'color': '#5A9FE0', 'x': 0.99, 'side': 'right',  'flow': gas_imports},
        {'id': 10, 'label': 'Public Charging',         'color': '#9C6FE4', 'x': 0.01, 'side': 'left',   'flow': public_charging},
    ]

    # Define all possible flows
    flow_definitions = [
        (0, 3, grid_import,    "rgba(255,107,107,0.4)"),
        (1, 2, batt_charge,    "rgba(255,217,61,0.4)"),
        (1, 3, pv_load,        "rgba(255,217,61,0.4)"),
        (1, 4, grid_export,    "rgba(139,125,4,0.4)"),
        (1, 5, curtailed,      "rgba(255,217,61,0.4)"),
        (2, 3, batt_discharge, "rgba(107,207,127,0.4)"),
        (6, 7, fuel_imports,   "rgba(72,58,46,0.4)"),
        (8, 9, gas_imports,    "rgba(60,129,208,0.4)"),
        (10, 3, public_charging, "rgba(156,111,228,0.4)"),
    ]

    # Filter to active flows and track connected nodes
    sources, targets, values, colors = [], [], [], []
    connected_nodes = set()
    for src, tgt, val, col in flow_definitions:
        if val > 0.001:
            sources.append(src)
            targets.append(tgt)
            values.append(val)
            colors.append(col)
            connected_nodes.add(src)
            connected_nodes.add(tgt)

    # Only keep nodes that participate in a flow
    visible_nodes = [n for n in all_nodes if n['id'] in connected_nodes]

    # --- Dynamic y-positioning ---
    # For each side (left, right, middle), space nodes evenly based on their flow size.
    # y positions are in [0.02, 0.98] to stay within the plot area.
    def assign_y_positions(nodes_on_side):
        """Assign y positions proportional to flow size with even spacing."""
        total_flow = sum(n['flow'] for n in nodes_on_side if n['flow'] > 0)
        if total_flow == 0 or len(nodes_on_side) == 0:
            return
        Y_MIN, Y_MAX = 0.02, 0.98
        usable = Y_MAX - Y_MIN
        # Each node gets space proportional to its share of total flow
        cursor = Y_MIN
        for n in nodes_on_side:
            share = n['flow'] / total_flow if n['flow'] > 0 else 0
            height = share * usable
            n['y'] = round(cursor + height / 2, 4)  # centre of its slot
            cursor += height

    for side in ('left', 'right', 'middle'):
        side_nodes = [n for n in visible_nodes if n['side'] == side]
        assign_y_positions(side_nodes)

    # Position Battery Storage just below the bottom edge of Electricity Consumption.
    # The bottom of a node = y_centre + half_its_share_of_usable_height.
    # We approximate that half-height as flow/total_right * usable/2.
    elec_node = next((n for n in visible_nodes if n['id'] == 3), None)
    batt_node = next((n for n in visible_nodes if n['id'] == 2), None)
    if elec_node is not None and batt_node is not None:
        right_nodes = [n for n in visible_nodes if n['side'] == 'right']
        total_right_flow = sum(n['flow'] for n in right_nodes if n['flow'] > 0)
        Y_MIN, Y_MAX = 0.02, 0.98
        usable = Y_MAX - Y_MIN
        elec_half_height = (elec_node['flow'] / total_right_flow * usable) / 2 if total_right_flow > 0 else 0.05
        gap = 0.02
        batt_node['y'] = round(elec_node['y'] + elec_half_height + gap, 4)

    # Remap indices to visible node positions
    old_to_new = {n['id']: idx for idx, n in enumerate(visible_nodes)}
    sources = [old_to_new[s] for s in sources]
    targets = [old_to_new[t] for t in targets]

    node_labels = [n['label'] for n in visible_nodes]
    node_colors = [n['color'] for n in visible_nodes]
    node_x      = [n['x']     for n in visible_nodes]
    node_y      = [n['y']     for n in visible_nodes]

    if len(sources) == 0:
        node_labels = ['Grid Import', 'Energy Consumption']
        node_colors = ['#FF6B6B', '#4ECDC4']
        node_x = [0.01, 0.99]
        node_y = [0.1, 0.1]
        sources = [0]
        targets = [1]
        values = [0.001]
        colors = ["rgba(255,107,107,0.4)"]

    fig = go.Figure(data=[go.Sankey(
        arrangement='fixed',
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors,
            x=node_x,
            y=node_y
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=colors
        )
    )])

    fig.update_layout(title_text="Sankey Energy Flow for the Household PV-Battery system <br> [{pv} kWp & {batt_kWh} kWh / {batt_kW:.1f} kW]".format(pv=summary['pv_capacity_kWp'], batt_kWh=summary['batt_capacity_kWh'], batt_kW=summary['batt_capacity_kW']), font_size=10)
    if filename != "":
        fig.write_html(filename)
    return fig

def create_bill_savings_waterfall(original_bill, bill_reduction_import, bill_value_export, new_bill, original_bill_import, original_bill_fixed):
    """
    Create a waterfall chart showing how bill savings are achieved.
    Shows the original bill as a stacked bar with import and fixed charges.
    
    Parameters:
    - original_bill: Original electricity bill without PV/Battery ($)
    - bill_reduction_import: Savings from avoided grid imports ($)
    - bill_value_export: Income from grid exports ($)
    - new_bill: Final bill with PV/Battery system ($)
    - original_bill_import: Import charges portion of original bill ($)
    - original_bill_fixed: Fixed charges portion of original bill ($)
    """
    
    # Calculate total savings
    total_savings = original_bill - new_bill
    
    # Create figure with stacked bar for original bill and waterfall for the rest
    fig = go.Figure()

    # Add stacked bar chart for original bill (fixed charges)
    fig.add_trace(go.Bar(
        name='Fixed Charges',
        x=['Original Bill<br>(No Solar/Battery)'],
        y=[original_bill_fixed],
        marker_color="#012B47",
        text=[f'Fixed: ${original_bill_fixed:.0f}'],
        textposition='inside',
        hovertemplate='Fixed Charges: $%{y:.0f}<extra></extra>'
    ))
    
    # Add stacked bar chart for original bill (import charges)
    fig.add_trace(go.Bar(
        name='Usage Charges',
        x=['Original Bill<br>(No Solar/Battery)'],
        y=[original_bill_import],
        marker_color="#56B3FF",
        text=[f'Usage: ${original_bill_import:.0f}'],
        textposition='inside',
        hovertemplate='Usage Charges: $%{y:.0f}<extra></extra>'
    ))
    
    # Calculate positions for waterfall components
    x_positions = [1, 2, 3]
    x_labels = ['Avoided Import<br>Costs', 'Export<br>Credits', 'New Bill<br>(With Solar/Battery)']
    
    # Add connector lines and bars manually for waterfall effect
    current_value = original_bill
    
    # Avoided Import Costs (decreasing)
    fig.add_trace(go.Bar(
        name='Savings',
        x=[x_labels[0]],
        y=[bill_reduction_import],
        base=[current_value - bill_reduction_import],
        marker_color='#6BCF7F',
        text=[f'-${bill_reduction_import:.0f}'],
        textposition='outside',
        hovertemplate='Avoided Import: -$%{y:.0f}<extra></extra>',
        showlegend=False
    ))
    current_value -= bill_reduction_import
    
    # Export Credits (decreasing)
    fig.add_trace(go.Bar(
        name='Savings',
        x=[x_labels[1]],
        y=[bill_value_export],
        base=[current_value - bill_value_export],
        marker_color='#6BCF7F',
        text=[f'-${bill_value_export:.0f}'],
        textposition='outside',
        hovertemplate='Export Credits: -$%{y:.0f}<extra></extra>',
        showlegend=False
    ))
    current_value -= bill_value_export
    
    # New Bill (total)
    fig.add_trace(go.Bar(
        name='New Bill',
        x=[x_labels[2]],
        y=[new_bill],
        marker_color='#4ECDC4',
        text=[f'${new_bill:.0f}'],
        textposition='outside',
        hovertemplate='New Bill: $%{y:.0f}<extra></extra>',
        showlegend=False
    ))
    
    # Add connector lines using category labels instead of numeric positions
    fig.add_trace(go.Scatter(
        x=['Original Bill<br>(No Solar/Battery)', x_labels[0]],
        y=[original_bill, original_bill],
        mode='lines',
        line=dict(color='gray', width=1, dash='dot'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=[x_labels[0], x_labels[1]],
        y=[current_value + bill_value_export, current_value + bill_value_export],
        mode='lines',
        line=dict(color='gray', width=1, dash='dot'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=[x_labels[1], x_labels[2]],
        y=[current_value, current_value],
        mode='lines',
        line=dict(color='gray', width=1, dash='dot'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=f"Annual Bill Savings Breakdown (Total Savings: ${total_savings:.0f})",
        barmode='stack',
        yaxis_title="Amount ($)",
        template='plotly_white',
        height=500,
        margin=dict(l=50, r=50, t=80, b=100),
        showlegend=False,
        xaxis_type='category',  # Force categorical x-axis
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        annotations=[
            dict(
                x='Original Bill<br>(No Solar/Battery)',
                y=original_bill,
                text=f'${original_bill:.0f}',
                showarrow=False,
                yshift=10,
            )
        ]
    )
    
    return fig

def create_ghg_stacked_bar(summary, scenario_param):
    """
    Stacked bar chart of annual GHG emissions broken down by source:
      - Grid electricity imports
      - Fuel combustion (petrol / diesel)
      - Gas combustion (natural gas)
    """
    # Emission factors
    grid_ef_kgCO2_per_kWh = scenario_param.get('grid_emissionsfactor', 679) / 1000  # kgCO2e/MWh → kgCO2e/kWh
    # Australian NGA factors (kg CO2e per kWh of fuel energy)
    petrol_ef_kgCO2_per_GJ = scenario_param['gasoline_GHG_emissions'] #kgCO2e/GJ
    diesel_ef_kgCO2_per_GJ = scenario_param['diesel_GHG_emissions'] #kgCO2e/GJ
    gas_ef_kgCO2_per_GJ = scenario_param['natgas_GHG_emissions'] #kgCO2e/GJ    

    petrol_imports_MJ = summary['petrol_imports_MJ']
    diesel_imports_MJ = summary['diesel_imports_MJ']

    grid_emissions = summary['total_grid_load_kWh'] * grid_ef_kgCO2_per_kWh
    fuel_emissions = (petrol_imports_MJ * petrol_ef_kgCO2_per_GJ + diesel_imports_MJ * diesel_ef_kgCO2_per_GJ) / 1000  # Convert MJ to GJ
    gas_emissions  = summary['gas_imports_kWh'] * CONVERT_KWH_TO_MJ * gas_ef_kgCO2_per_GJ / 1000  # Convert kWh to GJ

    categories = ['Grid Electricity', 'Fuel Combustion', 'Gas Combustion']
    values     = [grid_emissions, fuel_emissions, gas_emissions]
    colors     = ['#FF6B6B', '#483A2E', '#496687']

    fig = go.Figure()
    for cat, val, col in zip(categories, values, colors):
        fig.add_trace(go.Bar(
            name=cat,
            x=['Annual GHG Emissions'],
            y=[val],
            marker_color=col,
            text=[f'{val:,.0f}'],
            textposition='inside',
            hovertemplate=f'{cat}: %{{y:,.0f}} kg CO₂e<extra></extra>'
        ))

    total = sum(values)
    fig.update_layout(
        barmode='stack',
        title=dict(text='Annual GHG Emissions', font=dict(size=14)),
        yaxis_title='Emissions (kg CO₂e)',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=60, b=20),
        height=400,
        annotations=[dict(
            x='Annual GHG Emissions',
            y=total,
            text=f'<b>{total:,.0f} kg CO₂e</b>',
            xanchor='center',
            yanchor='bottom',
            yshift=6,
            showarrow=False,
            font=dict(size=13),
        )],
    )
    return fig


def create_emissions_reduction_waterfall(original_emissions_electricity, original_emissions_heating, original_emissions_transport,
                                         new_emissions_electricity, new_emissions_heating, new_emissions_transport):
    """
    Create a waterfall chart showing how GHG emissions are reduced.
    Shows the original emissions as a stacked bar with electricity, heating/cooking, and transport.
    
    Parameters:
    - original_emissions_electricity: Original electricity emissions (kg CO2e)
    - original_emissions_heating: Original heating/cooking emissions (kg CO2e)
    - original_emissions_transport: Original transport emissions (kg CO2e)
    - new_emissions_electricity: New electricity emissions with PV/Battery (kg CO2e)
    - new_emissions_heating: New heating/cooking emissions (kg CO2e)
    - new_emissions_transport: New transport emissions (kg CO2e)
    """
    
    # Calculate totals and reductions
    original_emissions = original_emissions_electricity + original_emissions_heating + original_emissions_transport
    new_emissions = new_emissions_electricity + new_emissions_heating + new_emissions_transport
    total_reduction = original_emissions - new_emissions
    
    reduction_electricity = original_emissions_electricity - new_emissions_electricity
    reduction_heating = original_emissions_heating - new_emissions_heating
    reduction_transport = original_emissions_transport - new_emissions_transport
    
    # Create figure with stacked bar for original emissions
    fig = go.Figure()

    # Add stacked bar chart for original emissions (transport)
    fig.add_trace(go.Bar(
        name='Transport',
        x=['Original Emissions<br>(No Solar/Battery)'],
        y=[original_emissions_transport],
        marker_color="#8B4513",
        text=[f'Transport: {original_emissions_transport:.0f}'],
        textposition='inside',
        hovertemplate='Transport: %{y:.0f} kg CO2e<extra></extra>'
    ))
    
    # Add stacked bar chart for original emissions (heating/cooking)
    fig.add_trace(go.Bar(
        name='Heating/Cooking',
        x=['Original Emissions<br>(No Solar/Battery)'],
        y=[original_emissions_heating],
        marker_color="#FF6B35",
        text=[f'Heating: {original_emissions_heating:.0f}'],
        textposition='inside',
        hovertemplate='Heating/Cooking: %{y:.0f} kg CO2e<extra></extra>'
    ))
    
    # Add stacked bar chart for original emissions (electricity)
    fig.add_trace(go.Bar(
        name='Electricity',
        x=['Original Emissions<br>(No Solar/Battery)'],
        y=[original_emissions_electricity],
        marker_color="#004E89",
        text=[f'Electricity: {original_emissions_electricity:.0f}'],
        textposition='inside',
        hovertemplate='Electricity: %{y:.0f} kg CO2e<extra></extra>'
    ))
    
    # Calculate positions for waterfall components
    x_labels = ['Electricity<br>Reduction', 'Heating/Cooking<br>Reduction', 'Transport<br>Reduction', 'New Emissions<br>(With Solar/Battery)']
    
    # Add connector lines and bars manually for waterfall effect
    current_value = original_emissions
    
    # Electricity Reduction (decreasing)
    if reduction_electricity > 0:
        fig.add_trace(go.Bar(
            name='Avoided Grid Emissions',
            x=[x_labels[0]],
            y=[reduction_electricity],
            base=[current_value - reduction_electricity],
            marker_color='#6BCF7F',
            text=[f'-{reduction_electricity:.0f}'],
            textposition='outside',
            hovertemplate=f'Avoided Grid Emissions: -{reduction_electricity:.0f} kg CO2e<extra></extra>',
            showlegend=False
        ))
        current_value -= reduction_electricity
    
    # Heating/Cooking Reduction (decreasing)
    if reduction_heating > 0:
        fig.add_trace(go.Bar(
            name='Emissions Reduction',
            x=[x_labels[1]],
            y=[reduction_heating],
            base=[current_value - reduction_heating],
            marker_color='#6BCF7F',
            text=[f'-{reduction_heating:.0f}'],
            textposition='outside',
            hovertemplate=f'Heating/Cooking Reduction: -{reduction_heating:.0f} kg CO2e<extra></extra>',
            showlegend=False
        ))
        current_value -= reduction_heating
    
    # Transport Reduction (decreasing)
    if reduction_transport > 0:
        fig.add_trace(go.Bar(
            name='Emissions Reduction',
            x=[x_labels[2]],
            y=[reduction_transport],
            base=[current_value - reduction_transport],
            marker_color='#6BCF7F',
            text=[f'-{reduction_transport:.0f}'],
            textposition='outside',
            hovertemplate=f'Transport Reduction: -{reduction_transport:.0f} kg CO2e<extra></extra>',
            showlegend=False
        ))
        current_value -= reduction_transport
    
    # New Emissions (total)
    fig.add_trace(go.Bar(
        name='New Emissions',
        x=[x_labels[3]],
        y=[new_emissions],
        marker_color='#4ECDC4',
        text=[f'{new_emissions:.0f}'],
        textposition='outside',
        hovertemplate='New Emissions: %{y:.0f} kg CO2e<extra></extra>',
        showlegend=False
    ))
    
    # Add connector lines
    connector_start = original_emissions
    fig.add_trace(go.Scatter(
        x=['Original Emissions<br>(No Solar/Battery)', x_labels[0]],
        y=[connector_start, connector_start],
        mode='lines',
        line=dict(color='gray', width=1, dash='dot'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    if reduction_electricity > 0:
        connector_start -= reduction_electricity
        fig.add_trace(go.Scatter(
            x=[x_labels[0], x_labels[1]],
            y=[connector_start, connector_start],
            mode='lines',
            line=dict(color='gray', width=1, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    if reduction_heating > 0:
        connector_start -= reduction_heating
        fig.add_trace(go.Scatter(
            x=[x_labels[1], x_labels[2]],
            y=[connector_start, connector_start],
            mode='lines',
            line=dict(color='gray', width=1, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    if reduction_transport > 0:
        connector_start -= reduction_transport
        fig.add_trace(go.Scatter(
            x=[x_labels[2], x_labels[3]],
            y=[connector_start, connector_start],
            mode='lines',
            line=dict(color='gray', width=1, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=f"Change in Annual GHG Emissions (Total Reduction: {total_reduction:.0f} kg CO2e)",
        barmode='stack',
        yaxis_title="Emissions (kg CO2e)",
        template='plotly_white',
        height=500,
        margin=dict(l=50, r=50, t=80, b=100),
        showlegend=False,
        xaxis_type='category',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        annotations=[
            dict(
                x='Original Emissions<br>(No Solar/Battery)',
                y=original_emissions,
                text=f'{original_emissions:.0f}',
                showarrow=False,
                yshift=10,
            )
        ]
    )
    
    return fig

@st.cache_data
def create_consumption_profile_chart(df_consumption, household_name, df_datetime=None):
    """
    Create a line chart showing the electricity consumption profile over a year (Jan 1 - Dec 31).
    
    Parameters:
    - df_consumption: Series or DataFrame column with consumption data (kWh)
    - household_name: Name of the household for the title
    - df_datetime: Optional datetime series for x-axis
    """
    num_points = len(df_consumption)
    
    # Always remap to 2021 for consistent display
    if df_datetime is not None and len(df_datetime) > 0:
        try:
            dt_index = pd.to_datetime(df_datetime)
            # Remap to 2021 keeping the day of year
            x_axis = dt_index.map(lambda x: x.replace(year=2021))
        except:
            # Create synthetic dates from Jan 1 to Dec 31 2021
            x_axis = pd.date_range(start='2021-01-01', periods=num_points, freq='h')
    else:
        # Create synthetic dates from Jan 1 to Dec 31 2021
        x_axis = pd.date_range(start='2021-01-01', periods=num_points, freq='h')
    
    total_consumption = df_consumption.sum()
    
    # Always use 2021 as the reference year
    year_start = pd.Timestamp('2021-01-01')
    year_end = pd.Timestamp('2021-12-31 23:59:59')
    
    # Create the figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=df_consumption.values,
        mode='lines',
        name='Consumption',
        line=dict(color='#1f77b4', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ))
    
    fig.update_layout(
        title=f"Annual Electricity Consumption Profile - {household_name} ({total_consumption/1000:.2f} MWh)",
        xaxis_title="Date",
        yaxis_title="Consumption (kW)",
        hovermode='x unified',
        template='plotly_white',
        height=400,
        margin=dict(l=50, r=50, t=60, b=50),
        xaxis=dict(
            range=[year_start, year_end],  # Force full year range
            tickformat='%b',  # Show month names
            dtick='M1',  # Tick every month
            tickangle=-45
        )
    )
    
    return fig

@st.cache_data
def create_solarpv_profile_chart(df_solar, solar_capacity, df_datetime=None):
    """
    Create a line chart showing the solar PV generation profile over a year (Jan 1 - Dec 31).
    
    Parameters:
    - df_solar: Series or DataFrame column with solar generation data (kWh/kWp)
    - solar_capacity: Solar panel capacity in kWp
    - df_datetime: Optional datetime series for x-axis
    """
    num_points = len(df_solar)
    
    # Always remap to 2021 for consistent display
    if df_datetime is not None and len(df_datetime) > 0:
        try:
            dt_index = pd.to_datetime(df_datetime)
            # Remap to 2021 keeping the day of year
            x_axis = dt_index.map(lambda x: x.replace(year=2021))
        except:
            # Create synthetic dates from Jan 1 to Dec 31 2021
            x_axis = pd.date_range(start='2021-01-01', periods=num_points, freq='h')
    else:
        # Create synthetic dates from Jan 1 to Dec 31 2021
        x_axis = pd.date_range(start='2021-01-01', periods=num_points, freq='h')
    
    capacity_factor = (df_solar).sum() / 8760
    
    # Always use 2021 as the reference year
    year_start = pd.Timestamp('2021-01-01')
    year_end = pd.Timestamp('2021-12-31 23:59:59')
    
    # Create the figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=(df_solar * solar_capacity).values,
        mode='lines',
        name='PV Generation',
        line=dict(color='#ff7f0e', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(255, 127, 14, 0.2)'
    ))
    
    fig.update_layout(
        title=f"Annual PV Generation Profile (CF = {capacity_factor*100:.1f}%)",
        xaxis_title="Date",
        yaxis_title="Generation (kW)",
        hovermode='x unified',
        template='plotly_white',
        height=400,
        margin=dict(l=50, r=50, t=60, b=50),
        xaxis=dict(
            range=[year_start, year_end],  # Force full year range
            tickformat='%b',  # Show month names
            dtick='M1',  # Tick every month
            tickangle=-45
        )
    )
    
    return fig

# df_datetime = df_opt_exogenous_timeseries['t']
@st.cache_data
def create_average_daily_simulation_profiles(df_results, scenario_param, df_opt_exogenous_timeseries):
    """
    Create average 24-hour consumption profiles for the full year and each season.
    
    Parameters:
    - df_results: Series or DataFrame column with simulation results
    - scenario_param: Scenario parameters for the simulation
    - df_opt_exogenous_timeseries: Optional DataFrame with exogenous timeseries for x-axis
    """
    num_points = len(df_results)
    
    # Create datetime index
    if df_opt_exogenous_timeseries is not None and len(df_opt_exogenous_timeseries) > 0:
        try:
            datetime_index = pd.to_datetime(df_opt_exogenous_timeseries['t'])
        except:
            datetime_index = pd.date_range(start='2021-01-01', periods=num_points, freq='h')
    else:
        datetime_index = pd.date_range(start='2021-01-01', periods=num_points, freq='h')

    # df_results.to_csv('df_results.csv')
    
    df_residualloadprofile_Wh = df_results['E_final_grid_import_Wh'] - df_results['E_final_grid_export_Wh'] - df_results['E_final_grid_export_curtailed_Wh']  # Net grid load profile (positive = import, negative = export)    
    
    df_underlyingloadprofile_Wh = df_results['E_underlyingload_Wh']
    df_exportprofile_Wh = df_results['E_final_grid_export_Wh']
    
    # Create DataFrame with datetime and consumption
    df = pd.DataFrame({
        'datetime': datetime_index,
        'residual_consumption': df_residualloadprofile_Wh / 1000,  # Convert to kWh
        'underlying_consumption': df_underlyingloadprofile_Wh / 1000,  # Convert to kWh
        'grid_export': df_exportprofile_Wh / 1000,  # Convert to kWh
    })
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    
    # Define seasons (Southern Hemisphere)
    # Summer: Dec, Jan, Feb (12, 1, 2)
    # Autumn: Mar, Apr, May (3, 4, 5)
    # Winter: Jun, Jul, Aug (6, 7, 8)
    # Spring: Sep, Oct, Nov (9, 10, 11)
    
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Summer'
        elif month in [3, 4, 5]:
            return 'Autumn'
        elif month in [6, 7, 8]:
            return 'Winter'
        else:
            return 'Spring'
    
    df['season'] = df['month'].apply(get_season)
    
        
    # Calculate average profiles for consumption
    annual_profile = df.groupby('hour')['residual_consumption'].mean()
    summer_profile = df[df['season'] == 'Summer'].groupby('hour')['residual_consumption'].mean()
    autumn_profile = df[df['season'] == 'Autumn'].groupby('hour')['residual_consumption'].mean()
    winter_profile = df[df['season'] == 'Winter'].groupby('hour')['residual_consumption'].mean()
    spring_profile = df[df['season'] == 'Spring'].groupby('hour')['residual_consumption'].mean()

    annual_underlyingprofile = df.groupby('hour')['underlying_consumption'].mean()
    summer_underlyingprofile = df[df['season'] == 'Summer'].groupby('hour')['underlying_consumption'].mean()
    autumn_underlyingprofile = df[df['season'] == 'Autumn'].groupby('hour')['underlying_consumption'].mean()
    winter_underlyingprofile = df[df['season'] == 'Winter'].groupby('hour')['underlying_consumption'].mean()
    spring_underlyingprofile = df[df['season'] == 'Spring'].groupby('hour')['underlying_consumption'].mean()

    annual_export_profile = df.groupby('hour')['grid_export'].mean()

    # Create the figure with subplots - Annual on top row (split), seasons on bottom
    fig = make_subplots(
        rows=3, cols=4,
        subplot_titles=('Annual Average (Net Load)',
                       'Annual (Consumption vs Exports)',
                       'Summer', 'Autumn', 'Winter', 'Spring'),
        vertical_spacing=0.1,
        horizontal_spacing=0.08,
        row_heights=[1, 2, 1],
        specs=[[{"colspan": 4}, None, None, None],
               [{"colspan": 4}, None, None, None],
               [{}, {}, {}, {}]]
    )
    
    hours = list(range(24))
    
    # Annual average (spans all columns in row 1)
    fig.add_trace(go.Scatter(
        x=hours, y=annual_profile.values,
        mode='lines',
        name='Net-load',
        line=dict(color='#2E86AB', width=2),
        fill='tozeroy',
        fillcolor='rgba(46, 134, 171, 0.2)',
        showlegend=True
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=hours, y=annual_underlyingprofile.values,
        mode='lines',
        name='Original',
        line=dict(color='#2E86AB', width=2, dash='dash'),
        showlegend=True
    ), row=1, col=1)

    # Annual stacked bar: consumption breakdown vs exports (spans all columns in row 1)
    fig.add_trace(go.Bar(
        x=hours, y=annual_underlyingprofile.values,
        name='Consumption',
        marker_color='rgba(46, 134, 171, 0.7)',
        showlegend=True
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=hours, y=annual_export_profile.values,
        name='Exports',
        marker_color='rgba(255, 217, 61, 0.8)',
        showlegend=True
    ), row=2, col=1)

    # Summer
    fig.add_trace(go.Scatter(
        x=hours, y=summer_profile.values,
        mode='lines',
        name='Summer',
        line=dict(color='#F77F00', width=1),
        fill='tozeroy',
        fillcolor='rgba(247, 127, 0, 0.2)',
        showlegend=False
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=hours, y=summer_underlyingprofile.values,
        mode='lines',
        name='Original',
        line=dict(color='#F77F00', width=1, dash='dash'),
        showlegend=False
    ), row=3, col=1)
    
    # Autumn
    fig.add_trace(go.Scatter(
        x=hours, y=autumn_profile.values,
        mode='lines',
        name='Autumn',
        line=dict(color='#D62828', width=1),
        fill='tozeroy',
        fillcolor='rgba(214, 40, 40, 0.2)',
        showlegend=False
    ), row=3, col=2)
    
    fig.add_trace(go.Scatter(
        x=hours, y=autumn_underlyingprofile.values,
        mode='lines',
        name='Original',
        line=dict(color='#D62828', width=1, dash='dash'),
        showlegend=False
    ), row=3, col=2)


    # Winter
    fig.add_trace(go.Scatter(
        x=hours, y=winter_profile.values,
        mode='lines',
        name='Winter',
        line=dict(color="#006297", width=1),
        fill='tozeroy',
        fillcolor='rgba(0, 48, 73, 0.2)',
        showlegend=False
    ), row=3, col=3)
    
    fig.add_trace(go.Scatter(
        x=hours, y=winter_underlyingprofile.values,
        mode='lines',
        name='Original',
        line=dict(color="#006297", width=1, dash='dash'),
        showlegend=False
    ), row=3, col=3)

    # Spring
    fig.add_trace(go.Scatter(
        x=hours, y=spring_profile.values,
        mode='lines',
        name='Spring',
        line=dict(color='#06A77D', width=1),
        fill='tozeroy',
        fillcolor='rgba(6, 167, 125, 0.2)',
        showlegend=False
    ), row=3, col=4)
    
    fig.add_trace(go.Scatter(
        x=hours, y=spring_underlyingprofile.values,
        mode='lines',
        name='Original',
        line=dict(color='#06A77D', width=1, dash='dash'),
        showlegend=False
    ), row=3, col=4)

    # Calculate the maximum value across all profiles for consistent y-axis
    max_values = [
        annual_profile.max(),
        summer_profile.max(),
        autumn_profile.max(),
        winter_profile.max(),
        spring_profile.max(),
        annual_underlyingprofile.max(),
        summer_underlyingprofile.max(),
        autumn_underlyingprofile.max(),
        winter_underlyingprofile.max(),
        spring_underlyingprofile.max()
    ]
    
    min_values = [
        annual_profile.min(),
        summer_profile.min(),
        autumn_profile.min(),
        winter_profile.min(),
        spring_profile.min(),
        annual_underlyingprofile.min(),
        summer_underlyingprofile.min(),
        autumn_underlyingprofile.min(),
        winter_underlyingprofile.min(),
        spring_underlyingprofile.min()
    ]

    max_value = max(max_values)
    y_max = max_value * 1.1  # Add 10% margin

    min_value = min(min_values)
    y_min = min_value * 1.1  # Add 10% margin

    bar_y_max = max(annual_underlyingprofile.max(), annual_export_profile.max()) * 1.1

    # Update axes with consistent y-axis range
    # Row 1, col 1 (Annual line) — hourly tick marks + vertical gridlines
    fig.update_xaxes(title_text="Hour of Day", row=1, col=1, range=[0, 23],
                     tickmode='linear', tick0=0, dtick=1,
                     showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.15)')
    fig.update_yaxes(title_text="Avg Power (kW)", row=1, col=1, range=[y_min, y_max])
    # Row 2, col 1 (Annual stacked bar)
    fig.update_xaxes(title_text="Hour of Day", row=2, col=1, range=[-0.5, 23.5],
                     tickmode='linear', tick0=0, dtick=1,
                     showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.15)')
    fig.update_yaxes(title_text="Avg Power (kW)", row=2, col=1, range=[0, bar_y_max])
    
    # Row 3 (Seasons) — tick every 2 hours + vertical gridlines
    for j in range(1, 5):
        fig.update_xaxes(title_text="Hour of Day", row=3, col=j, range=[0, 23],
                         tickmode='linear', tick0=0, dtick=2,
                         showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.15)',
                         tickfont=dict(size=10))
        fig.update_yaxes(title_text="Avg Power (kW)", row=3, col=j, range=[y_min, y_max])
    
    household_name = scenario_param.get('meter_name')
    
    fig.update_layout(
        title_text=f"Average 24-Hour Residual Load Profiles - {household_name}",
        height=900,
        template='plotly_white',
        barmode='stack',
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig



@st.cache_data
def create_average_daily_profiles(df_consumption, household_name, df_datetime=None, df_solar=None, solar_capacity=0.0):
    """
    Create average 24-hour consumption profiles for the full year and each season.
    
    Parameters:
    - df_consumption: Series or DataFrame column with consumption data (kWh)
    - household_name: Name of the household for the title
    - df_datetime: Optional datetime series for x-axis
    - df_solar: Optional Series with solar generation data (kWh/kWp)
    - solar_capacity: Solar panel capacity in kWp
    """
    num_points = len(df_consumption)
    
    # Create datetime index
    if df_datetime is not None and len(df_datetime) > 0:
        try:
            datetime_index = pd.to_datetime(df_datetime)
        except:
            datetime_index = pd.date_range(start='2021-01-01', periods=num_points, freq='h')
    else:
        datetime_index = pd.date_range(start='2021-01-01', periods=num_points, freq='h')
    
    # Create DataFrame with datetime and consumption
    df = pd.DataFrame({
        'datetime': datetime_index,
        'consumption': df_consumption.values
    })
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    
    # Add solar data if provided
    if df_solar is not None and solar_capacity > 0:
        df['solar'] = df_solar.values * solar_capacity
    
    # Define seasons (Southern Hemisphere)
    # Summer: Dec, Jan, Feb (12, 1, 2)
    # Autumn: Mar, Apr, May (3, 4, 5)
    # Winter: Jun, Jul, Aug (6, 7, 8)
    # Spring: Sep, Oct, Nov (9, 10, 11)
    
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Summer'
        elif month in [3, 4, 5]:
            return 'Autumn'
        elif month in [6, 7, 8]:
            return 'Winter'
        else:
            return 'Spring'
    
    df['season'] = df['month'].apply(get_season)
    
    # Calculate average profiles for consumption
    annual_profile = df.groupby('hour')['consumption'].mean()
    summer_profile = df[df['season'] == 'Summer'].groupby('hour')['consumption'].mean()
    autumn_profile = df[df['season'] == 'Autumn'].groupby('hour')['consumption'].mean()
    winter_profile = df[df['season'] == 'Winter'].groupby('hour')['consumption'].mean()
    spring_profile = df[df['season'] == 'Spring'].groupby('hour')['consumption'].mean()
    
    # Calculate average profiles for solar if available
    if df_solar is not None and solar_capacity > 0:
        annual_solar = df.groupby('hour')['solar'].mean()
        summer_solar = df[df['season'] == 'Summer'].groupby('hour')['solar'].mean()
        autumn_solar = df[df['season'] == 'Autumn'].groupby('hour')['solar'].mean()
        winter_solar = df[df['season'] == 'Winter'].groupby('hour')['solar'].mean()
        spring_solar = df[df['season'] == 'Spring'].groupby('hour')['solar'].mean()
    
    # Create the figure with subplots - Annual on top row, seasons on bottom
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=('Annual Average',
                       'Summer', 'Autumn', 'Winter', 'Spring'),
        vertical_spacing=0.2,
        horizontal_spacing=0.08,
        specs=[[{"colspan": 4}, None, None, None],
               [{}, {}, {}, {}]]
    )
    
    hours = list(range(24))
    
    # Annual average (spans all columns in row 1)
    fig.add_trace(go.Scatter(
        x=hours, y=annual_profile.values,
        mode='lines',
        name='Consumption',
        line=dict(color='#2E86AB', width=2),
        fill='tozeroy',
        fillcolor='rgba(46, 134, 171, 0.2)',
        showlegend=True
    ), row=1, col=1)
    
    if df_solar is not None and solar_capacity > 0:
        fig.add_trace(go.Scatter(
            x=hours, y=annual_solar.values,
            mode='lines',
            name='Solar PV',
            line=dict(color='#FFA500', width=2),
            showlegend=True
        ), row=1, col=1)
    
    # Summer
    fig.add_trace(go.Scatter(
        x=hours, y=summer_profile.values,
        mode='lines',
        name='Summer',
        line=dict(color='#F77F00', width=1),
        fill='tozeroy',
        fillcolor='rgba(247, 127, 0, 0.2)',
        showlegend=False
    ), row=2, col=1)
    
    if df_solar is not None and solar_capacity > 0:
        fig.add_trace(go.Scatter(
            x=hours, y=summer_solar.values,
            mode='lines',
            name='Solar PV',
            line=dict(color='#FFA500', width=1),
            showlegend=False
        ), row=2, col=1)
    
    # Autumn
    fig.add_trace(go.Scatter(
        x=hours, y=autumn_profile.values,
        mode='lines',
        name='Autumn',
        line=dict(color='#D62828', width=1),
        fill='tozeroy',
        fillcolor='rgba(214, 40, 40, 0.2)',
        showlegend=False
    ), row=2, col=2)
    
    if df_solar is not None and solar_capacity > 0:
        fig.add_trace(go.Scatter(
            x=hours, y=autumn_solar.values,
            mode='lines',
            name='Solar PV',
            line=dict(color='#FFA500', width=1),
            showlegend=False
        ), row=2, col=2)
    
    # Winter
    fig.add_trace(go.Scatter(
        x=hours, y=winter_profile.values,
        mode='lines',
        name='Winter',
        line=dict(color="#006297", width=1),
        fill='tozeroy',
        fillcolor='rgba(0, 48, 73, 0.2)',
        showlegend=False
    ), row=2, col=3)
    
    if df_solar is not None and solar_capacity > 0:
        fig.add_trace(go.Scatter(
            x=hours, y=winter_solar.values,
            mode='lines',
            name='Solar PV',
            line=dict(color='#FFA500', width=1),
            showlegend=False
        ), row=2, col=3)
    
    # Spring
    fig.add_trace(go.Scatter(
        x=hours, y=spring_profile.values,
        mode='lines',
        name='Spring',
        line=dict(color='#06A77D', width=1),
        fill='tozeroy',
        fillcolor='rgba(6, 167, 125, 0.2)',
        showlegend=False
    ), row=2, col=4)
    
    if df_solar is not None and solar_capacity > 0:
        fig.add_trace(go.Scatter(
            x=hours, y=spring_solar.values,
            mode='lines',
            name='Solar PV',
            line=dict(color='#FFA500', width=1),
            showlegend=False
        ), row=2, col=4)
    
    # Calculate the maximum value across all profiles for consistent y-axis
    max_values = [
        annual_profile.max(),
        summer_profile.max(),
        autumn_profile.max(),
        winter_profile.max(),
        spring_profile.max()
    ]
    
    if df_solar is not None and solar_capacity > 0:
        max_values.extend([
            annual_solar.max(),
            summer_solar.max(),
            autumn_solar.max(),
            winter_solar.max(),
            spring_solar.max()
        ])
    
    max_value = max(max_values)
    y_max = max_value * 1.1  # Add 10% margin
    
    # Update axes with consistent y-axis range
    # Row 1 (Annual)
    fig.update_xaxes(title_text="Hour of Day", row=1, col=1, range=[0, 23])
    fig.update_yaxes(title_text="Avg Power (kW)", row=1, col=1, range=[0, y_max])
    
    # Row 2 (Seasons)
    for j in range(1, 5):
        fig.update_xaxes(title_text="Hour of Day", row=2, col=j, range=[0, 23])
        fig.update_yaxes(title_text="Avg Power (kW)", row=2, col=j, range=[0, y_max])
    
    fig.update_layout(
        title_text=f"Average 24-Hour Consumption Profiles - {household_name}",
        height=600,
        template='plotly_white',
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

@st.cache_data
def create_average_daily_profiles_statistical(df_consumption, household_name, df_datetime=None):
    """
    Create box and whisker plots showing the distribution of 24-hour consumption profiles 
    for the full year and each season.
    
    Parameters:
    - df_consumption: Series or DataFrame column with consumption data (kWh)
    - household_name: Name of the household for the title
    - df_datetime: Optional datetime series for x-axis
    """
    num_points = len(df_consumption)
    
    # Create datetime index
    if df_datetime is not None and len(df_datetime) > 0:
        try:
            datetime_index = pd.to_datetime(df_datetime)
        except:
            datetime_index = pd.date_range(start='2021-01-01', periods=num_points, freq='h')
    else:
        datetime_index = pd.date_range(start='2021-01-01', periods=num_points, freq='h')
    
    # Create DataFrame with datetime and consumption
    df = pd.DataFrame({
        'datetime': datetime_index,
        'consumption': df_consumption.values
    })
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    
    # Define seasons (Southern Hemisphere)
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Summer'
        elif month in [3, 4, 5]:
            return 'Autumn'
        elif month in [6, 7, 8]:
            return 'Winter'
        else:
            return 'Spring'
    
    df['season'] = df['month'].apply(get_season)
    
    # Create the figure with subplots - Annual on top row, seasons on bottom
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=('Annual Consumption Distribution',
                       'Summer', 'Autumn', 'Winter', 'Spring'),
        vertical_spacing=0.2,
        horizontal_spacing=0.08,
        specs=[[{"colspan": 4}, None, None, None],
               [{}, {}, {}, {}]]
    )
    
    hours = list(range(24))
    
    # Annual consumption distribution (box plot for each hour)
    for hour in hours:
        hour_data = df[df['hour'] == hour]['consumption'].values
        fig.add_trace(go.Box(
            y=hour_data,
            name=str(hour),
            marker_color='#2E86AB',
            showlegend=False,
            boxmean='sd'  # Show mean and standard deviation
        ), row=1, col=1)
    
    # Summer distribution
    df_summer = df[df['season'] == 'Summer']
    for hour in hours:
        hour_data = df_summer[df_summer['hour'] == hour]['consumption'].values
        fig.add_trace(go.Box(
            y=hour_data,
            name=str(hour),
            marker_color='#F77F00',
            showlegend=False,
            boxmean='sd'
        ), row=2, col=1)
    
    # Autumn distribution
    df_autumn = df[df['season'] == 'Autumn']
    for hour in hours:
        hour_data = df_autumn[df_autumn['hour'] == hour]['consumption'].values
        fig.add_trace(go.Box(
            y=hour_data,
            name=str(hour),
            marker_color='#D62828',
            showlegend=False,
            boxmean='sd'
        ), row=2, col=2)
    
    # Winter distribution
    df_winter = df[df['season'] == 'Winter']
    for hour in hours:
        hour_data = df_winter[df_winter['hour'] == hour]['consumption'].values
        fig.add_trace(go.Box(
            y=hour_data,
            name=str(hour),
            marker_color='#006297',
            showlegend=False,
            boxmean='sd'
        ), row=2, col=3)
    
    # Spring distribution
    df_spring = df[df['season'] == 'Spring']
    for hour in hours:
        hour_data = df_spring[df_spring['hour'] == hour]['consumption'].values
        fig.add_trace(go.Box(
            y=hour_data,
            name=str(hour),
            marker_color='#06A77D',
            showlegend=False,
            boxmean='sd'
        ), row=2, col=4)
    
    # Calculate the maximum value across all data for consistent y-axis
    max_value = df['consumption'].max()
    y_max = max_value * 1.1  # Add 10% margin
    
    # Update axes with consistent y-axis range
    # Row 1 (Annual Consumption)
    fig.update_xaxes(title_text="Hour of Day", row=1, col=1, range=[-0.5, 23.5])
    fig.update_yaxes(title_text="Consumption (kW)", row=1, col=1, range=[0, y_max])
    
    # Row 2 (Seasonal Consumption)
    for j in range(1, 5):
        fig.update_xaxes(title_text="Hour of Day", row=2, col=j, range=[-0.5, 23.5])
        fig.update_yaxes(title_text="Consumption (kW)", row=2, col=j, range=[0, y_max])
    
    fig.update_layout(
        title_text=f"24-Hour Consumption Distribution (Box & Whisker) - {household_name}",
        height=600,
        template='plotly_white',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

# scenario_param = SCENARIO_PARAM

@st.cache_data
def create_transport_electricity_profile(selected_cars, df_opt_exogenous_timeseries, scenario_param):
    """
    Create electricity consumption profile for selected electric vehicles.
    
    Parameters:
    - selected_cars: List of CarConfig objects
    - df_opt_exogenous_timeseries: DataFrame containing the exogenous timeseries data
    - scenario_param: Dictionary containing scenario parameters
    
    Returns:
    - List containing electricity consumption profiles and other related data
    """

    transport_fuel_energy_MJ = 0
    transport_petrolfuel_energy_MJ = 0
    transport_dieselfuel_energy_MJ = 0
    transport_electricity_energy_kWh = 0
    transport_fuel_energy_kWh = 0
    transport_public_charging_energy_kWh = 0
    num_solarcharging_EVs = 0
    solarcharging_cars = []
    solarcharging_schedules = []
    num_timesteps = len(df_opt_exogenous_timeseries)
    timesteps_per_day = int(num_timesteps / DAYS_PER_YEAR)    
    convert_kW_to_kWh = 24 / timesteps_per_day
    
    # Count the number of EVs
    num_EVs = 0
    for i in range(len(selected_cars)):
        if selected_cars[i].fuel_type == 'Electric':
            num_EVs += 1
    df_ev_charging_Wh_annualts = pd.DataFrame(index=range(num_timesteps), dtype=int)  # Initialize with zeros, columns based on number of EVs
    # Add columns for each EV's charging profile, based on the car index
    for i in range (len(selected_cars)):
        if selected_cars[i].fuel_type == 'Electric':
            df_ev_charging_Wh_annualts[f'Car_{i+1}'] = 0
        
    # i = 1; dist = 100000; selected_cars[0].annual_distance_km = 100000; selected_cars[0].charging_strategy = "Solar self-charging first"; selected_cars[1].charging_strategy = "Solar self-charging first"
    for i in range (len(selected_cars)):
        car_column = f'Car_{i+1}'
        car_type = selected_cars[i].fuel_type
        dist = selected_cars[i].annual_distance_km
        eff = selected_cars[i].efficiency
        if car_type in ['Petrol', 'Diesel', 'Electric']:
            if car_type == 'Petrol':
                transport_fuel_energy_MJ += dist / 100 * eff * scenario_param['gasoline_energy_content']
                transport_petrolfuel_energy_MJ += dist / 100 * eff * scenario_param['gasoline_energy_content']
            elif car_type == 'Diesel':
                transport_fuel_energy_MJ += dist / 100 * eff * scenario_param['diesel_energy_content']
                transport_dieselfuel_energy_MJ += dist / 100 * eff * scenario_param['diesel_energy_content']
            else:
                transport_electricity_energy_kWh += dist / 100 * eff
                schedule = selected_cars[i].schedule
                charger_speed = selected_cars[i].charger_speed
                charging_strategy = selected_cars[i].charging_strategy
                
                # Map schedule to a 24 hours array of 0s and 1s (1 = at home, can charge, 0 = away, cannot charge)
                df_diurnal_car_schedule = pd.DataFrame(0, index=range(timesteps_per_day), 
                                                       columns=['at_home', 
                                                                'away', 
                                                                'E_consumed_Wh',
                                                                'E_to_refill_Wh',
                                                                'E_charge_max_Wh',
                                                                ], 
                                                       dtype=np.int64)
                for j in range(len(schedule)):
                    depart_time = schedule[j]['depart_hour'] + schedule[j]['depart_minute'] / 60 
                    arrive_time = schedule[j]['arrive_hour'] + schedule[j]['arrive_minute'] / 60 
                    t = 0
                    for t in range(timesteps_per_day):
                        time_of_day = t * (24 / timesteps_per_day)
                        if arrive_time > time_of_day and time_of_day >= depart_time:
                            df_diurnal_car_schedule.loc[t, 'away'] = 1
                df_diurnal_car_schedule['at_home'] = 1 - df_diurnal_car_schedule['away']
                daily_dist_km = dist / DAYS_PER_YEAR
                time_away = df_diurnal_car_schedule['away'].sum() * (24 / timesteps_per_day)
                df_diurnal_car_schedule['E_consumed_Wh'] = df_diurnal_car_schedule['away'] * int(np.round(daily_dist_km / time_away / 100 * eff * 1000))

                sequence_E_consumed_Wh = 0
                b_sequence_started = False
                for t in range(timesteps_per_day):
                    if df_diurnal_car_schedule['away'][t] == 1:
                        b_sequence_started = True
                        sequence_E_consumed_Wh += df_diurnal_car_schedule['E_consumed_Wh'][t]                    
                    else:
                        if b_sequence_started:
                            b_sequence_started = False
                            df_diurnal_car_schedule.loc[t, 'E_to_refill_Wh'] = sequence_E_consumed_Wh
                            sequence_E_consumed_Wh = 0
                                           
                charger_power_kW = 0
                if charger_speed == "Level 1 (2.4 kW)":
                    charger_power_kW = 2.4
                elif charger_speed == "Level 2 single phase (7.2 kW)":
                    charger_power_kW = 7.2
                elif charger_speed == "Level 2 three phase (22 kW)":
                    charger_power_kW = 22
                for t in range(timesteps_per_day):
                    df_diurnal_car_schedule.loc[t, 'E_charge_max_Wh'] = int(np.round(df_diurnal_car_schedule.loc[t, 'at_home'] * charger_power_kW * convert_kW_to_kWh * 1000)) # Max energy that can be charged in that timestep (if at home)  # type: ignore

                # Figure out the annual charging profile for this car
                # ----------------------------------------------------------------------------------------------------
                # ====================================================================================================
                # Charge the car each time it returns home. Charge enough to replace the previous trip's distance, if 
                # not, then remember so you refill it when you return once again. If there still isn't enough capacity
                # to fully charge the car over the course of the day, top up with public charging.
                # ====================================================================================================
                if charging_strategy == "Immediately upon return":
                    
                    # Initialise
                    df_diurnal_car_schedule['can_charge'] = df_diurnal_car_schedule['at_home']
                    df_diurnal_car_schedule['E_charge_Wh'] = 0
                    df_diurnal_car_schedule['E_chargeremaining_Wh'] = 0
                    diurnal_public_charge_Wh = 0
                    
                    # Find the first timestep that charging should start
                    t_firstcharge = -1
                    for t in range(timesteps_per_day):
                        if df_diurnal_car_schedule['E_to_refill_Wh'][t] > 0:
                            t_firstcharge = t
                            break
                    
                    # If there is a charge needed, fill the charging profile according to the max charge power and energy to refill, starting from the first timestep that charging should start
                    if t_firstcharge >= 0:
                        #x = 0
                        for x in range(timesteps_per_day):
                            t = x + t_firstcharge
                            if t >= timesteps_per_day:
                                t = t - timesteps_per_day
                            t_prev = t - 1
                            if t_prev < 0:
                                t_prev = timesteps_per_day - 1
                            
                            if df_diurnal_car_schedule.at[t, 'can_charge'] == 1: # type: ignore
                                charge_required_Wh = int(df_diurnal_car_schedule.at[t,'E_to_refill_Wh'] + df_diurnal_car_schedule.at[t_prev, 'E_chargeremaining_Wh']) # type: ignore
                                charge_possible_Wh = int(df_diurnal_car_schedule.at[t, 'E_charge_max_Wh']) # type: ignore
                                charge_Wh = min(charge_required_Wh, charge_possible_Wh)
                                df_diurnal_car_schedule.at[t, 'E_charge_Wh'] = charge_Wh
                                df_diurnal_car_schedule.at[t, 'E_chargeremaining_Wh'] = charge_required_Wh - charge_Wh
                            else:
                                df_diurnal_car_schedule.at[t, 'E_charge_Wh'] = 0
                                df_diurnal_car_schedule.at[t, 'E_chargeremaining_Wh'] = int(df_diurnal_car_schedule.at[t,'E_to_refill_Wh'] + df_diurnal_car_schedule.at[t_prev, 'E_chargeremaining_Wh']) # type: ignore
                        
                        # If the car cannot be completely charged at home, top up with public charging
                        if df_diurnal_car_schedule.at[t, 'E_chargeremaining_Wh'] > 0: # type: ignore
                            diurnal_public_charge_Wh = int(df_diurnal_car_schedule.at[t, 'E_chargeremaining_Wh']) # type: ignore

                    # Fill the annual charging profile for this car by repeating the diurnal profile for each day of the year
                    df_ev_charging_Wh_annualts[car_column] = np.tile(df_diurnal_car_schedule['E_charge_Wh'].values, DAYS_PER_YEAR) # type: ignore
                    transport_public_charging_energy_kWh += int(np.round(diurnal_public_charge_Wh * DAYS_PER_YEAR / 1000))

                # ====================================================================================================
                # Only charge during the overnight period, if there isn't enough charging capacity then top up with 
                # public charging. Also charge as much as you can to cover the whole day's electricity demand.
                # ====================================================================================================
                elif charging_strategy == "Overnight only (11pm-7am)":
                    
                    # Initialise
                    df_diurnal_car_schedule['can_charge'] = 0
                    df_diurnal_car_schedule['E_charge_Wh'] = 0
                    df_diurnal_car_schedule['E_chargeremaining_Wh'] = 0
                    diurnal_public_charge_Wh = 0

                    for t in range(timesteps_per_day):
                        if t * (24 / timesteps_per_day) >= 23 or t * (24 / timesteps_per_day) < 7:
                            df_diurnal_car_schedule.loc[t, 'can_charge'] = int(df_diurnal_car_schedule.loc[t, 'at_home']) # type: ignore
                        else:
                            df_diurnal_car_schedule.loc[t, 'can_charge'] = int(0) # type: ignore

                    # Figure out the first timestep you can charge (after 11pm)
                    t_firstcharge = -1
                    t_11pm = int(23 / 24 * timesteps_per_day)
                    for x in range(timesteps_per_day):
                        t = x + t_11pm
                        if t >= timesteps_per_day:
                            t = t - timesteps_per_day
                        if df_diurnal_car_schedule['can_charge'][t] > 0:
                            t_firstcharge = t
                            break

                    # Total charging required
                    daily_charge_required_Wh = df_diurnal_car_schedule['E_to_refill_Wh'].sum()
                    daily_charge_available_Wh = (df_diurnal_car_schedule['E_charge_max_Wh'] * df_diurnal_car_schedule['can_charge']).sum()
                    if daily_charge_required_Wh > daily_charge_available_Wh:
                        diurnal_public_charge_Wh = daily_charge_required_Wh - daily_charge_available_Wh
                    daily_charge_Wh = min(daily_charge_required_Wh, daily_charge_available_Wh)

                    # If there is a charge needed, fill the charging profile according to the max charge power and energy to refill, starting from the first timestep that charging should start
                    if t_firstcharge >= 0:
                        # x = 0
                        for x in range(timesteps_per_day):
                            t = x + t_firstcharge
                            if t >= timesteps_per_day:
                                t = t - timesteps_per_day
                                
                            if df_diurnal_car_schedule.at[t, 'can_charge'] == 1: # type: ignore
                                charge_required_Wh = daily_charge_Wh
                                charge_possible_Wh = int(df_diurnal_car_schedule.at[t, 'E_charge_max_Wh']) # type: ignore
                                charge_Wh = min(charge_required_Wh, charge_possible_Wh)
                                df_diurnal_car_schedule.at[t, 'E_charge_Wh'] = charge_Wh
                                daily_charge_Wh = daily_charge_Wh - charge_Wh
                            else:
                                df_diurnal_car_schedule.at[t, 'E_charge_Wh'] = 0

                    # Fill the annual charging profile for this car by repeating the diurnal profile for each day of the year
                    df_ev_charging_Wh_annualts[car_column] = np.tile(df_diurnal_car_schedule['E_charge_Wh'].values, DAYS_PER_YEAR) # type: ignore
                    transport_public_charging_energy_kWh += int(np.round(diurnal_public_charge_Wh * DAYS_PER_YEAR / 1000))
                    
                # ====================================================================================================
                # Try and charge from midday (11am) onwards. Also charge as much as you can to cover the whole day's 
                # electricity demand.
                # ====================================================================================================
                elif charging_strategy == "Midday only (11am-1pm)":
                    
                    # Initialise
                    df_diurnal_car_schedule['can_charge'] = 0
                    df_diurnal_car_schedule['E_charge_Wh'] = 0
                    diurnal_public_charge_Wh = 0
                    
                    # Figure out when the EV can actually charge
                    for t in range(timesteps_per_day):
                        if t * (24 / timesteps_per_day) >= 11 and t * (24 / timesteps_per_day) < 13:
                            df_diurnal_car_schedule.loc[t, 'can_charge'] = int(df_diurnal_car_schedule.loc[t, 'at_home']) # type: ignore
                        else:
                            df_diurnal_car_schedule.loc[t, 'can_charge'] = int(0) # type: ignore

                    # Total charging required
                    daily_charge_required_Wh = df_diurnal_car_schedule['E_to_refill_Wh'].sum()
                    daily_charge_available_Wh = (df_diurnal_car_schedule['E_charge_max_Wh'] * df_diurnal_car_schedule['can_charge']).sum()
                    if daily_charge_required_Wh > daily_charge_available_Wh:
                        diurnal_public_charge_Wh = daily_charge_required_Wh - daily_charge_available_Wh
                    daily_charge_Wh = min(daily_charge_required_Wh, daily_charge_available_Wh)
                    
                    # Figure out the first timestep you can charge (after 11am)
                    t_firstcharge = -1
                    t_11am = int(11 / 24 * timesteps_per_day)
                    for x in range(timesteps_per_day):
                        t = x + t_11am
                        if t >= timesteps_per_day:
                            t = t - timesteps_per_day
                        if df_diurnal_car_schedule['can_charge'][t] > 0:
                            t_firstcharge = t
                            break
                        
                    if t_firstcharge >= 0:
                        # x = 0
                        for x in range(timesteps_per_day):
                            t = x + t_firstcharge
                            if t >= timesteps_per_day:
                                t = t - timesteps_per_day
                                
                            if df_diurnal_car_schedule.at[t, 'can_charge'] == 1: # type: ignore
                                charge_required_Wh = daily_charge_Wh
                                charge_possible_Wh = int(df_diurnal_car_schedule.at[t, 'E_charge_max_Wh']) # type: ignore
                                charge_Wh = min(charge_required_Wh, charge_possible_Wh)
                                df_diurnal_car_schedule.at[t, 'E_charge_Wh'] = charge_Wh
                                daily_charge_Wh = daily_charge_Wh - charge_Wh
                            else:
                                df_diurnal_car_schedule.at[t, 'E_charge_Wh'] = 0

                    # Fill the annual charging profile for this car by repeating the diurnal profile for each day of the year
                    df_ev_charging_Wh_annualts[car_column] = np.tile(df_diurnal_car_schedule['E_charge_Wh'].values, DAYS_PER_YEAR) # type: ignore
                    transport_public_charging_energy_kWh += int(np.round(diurnal_public_charge_Wh * DAYS_PER_YEAR / 1000))
                    
                # ====================================================================================================
                # Essentially charge the car last minute so that it is full just before the first departure
                # ====================================================================================================                    
                elif charging_strategy == "Just before departure":
                    
                    # Initialise
                    df_diurnal_car_schedule['can_charge'] = df_diurnal_car_schedule['at_home']
                    df_diurnal_car_schedule['E_charge_Wh'] = 0
                    diurnal_public_charge_Wh = 0
                    
                    # Total charging required
                    daily_charge_required_Wh = df_diurnal_car_schedule['E_to_refill_Wh'].sum()
                    daily_charge_available_Wh = df_diurnal_car_schedule['E_charge_max_Wh'].sum()
                    if daily_charge_required_Wh > daily_charge_available_Wh:
                        diurnal_public_charge_Wh = daily_charge_required_Wh - daily_charge_available_Wh
                    daily_charge_Wh = min(daily_charge_required_Wh, daily_charge_available_Wh)
                    
                    # Find the first timestep that charging could start
                    t_firstcharge = -1
                    for t in range(timesteps_per_day):
                        if df_diurnal_car_schedule['away'][t] > 0:
                            t_firstcharge = t - 1
                            if t_firstcharge < 0:
                                t_firstcharge = timesteps_per_day - 1
                            break
                    
                    if t_firstcharge >= 0:
                        # x = 0
                        for x in range(timesteps_per_day):
                            t = t_firstcharge - x
                            if t < 0:
                                t = t + timesteps_per_day

                            if df_diurnal_car_schedule.at[t, 'can_charge'] == 1: # type: ignore
                                charge_required_Wh = daily_charge_Wh
                                charge_possible_Wh = int(df_diurnal_car_schedule.at[t, 'E_charge_max_Wh']) # type: ignore
                                charge_Wh = min(charge_required_Wh, charge_possible_Wh)
                                df_diurnal_car_schedule.at[t, 'E_charge_Wh'] = charge_Wh
                                daily_charge_Wh = daily_charge_Wh - charge_Wh
                            else:
                                df_diurnal_car_schedule.at[t, 'E_charge_Wh'] = 0

                    # Fill the annual charging profile for this car by repeating the diurnal profile for each day of the year
                    df_ev_charging_Wh_annualts[car_column] = np.tile(df_diurnal_car_schedule['E_charge_Wh'].values, DAYS_PER_YEAR) # type: ignore
                    transport_public_charging_energy_kWh += int(np.round(diurnal_public_charge_Wh * DAYS_PER_YEAR / 1000))
                    
                # ====================================================================================================
                # For solar self-charging, setup the parameters as all solar charging needs to be processed together
                # in order to share the excess solar generation 
                # ====================================================================================================                    
                elif charging_strategy == "Solar self-charging first":                
                    num_solarcharging_EVs += 1
                    solarcharging_cars.append(car_column)
                    solarcharging_schedules.append(df_diurnal_car_schedule)
  
    # Solar self-charging preferred algorithm
    # ---------------------------------------
    # Determine how to charge the set of solar charging cars with the shared resource of excess solar generation
    if num_solarcharging_EVs > 0:
        
        # Figure out the excess solar generation available at this household
        underlying_load_kWh_annualts = df_opt_exogenous_timeseries['E_netload_kWh'].clip(lower = 0)
        solar_generation_kWh_annualts = df_opt_exogenous_timeseries['E_pvunitgeneration_kWh'] * scenario_param['solar_capacity']
        excess_solar_generation_kWh_annualts = (solar_generation_kWh_annualts - underlying_load_kWh_annualts).clip(lower = 0)
        
        df_ev_solar_charging_Wh_annualts = pd.DataFrame(index=range(num_timesteps), dtype=int)
        df_ev_solar_charging_Wh_annualts['E_excess_solar_Wh'] = (excess_solar_generation_kWh_annualts * 1000).astype(int)
        df_ev_solar_charging_Wh_annualts['Num_EVs_AtHome'] = 0
        ev_solar_charging_torefill_Wh = [0] * num_solarcharging_EVs
        public_charging_required_Wh = 0
        
        #ev_index = 0    
        for ev_index in range(num_solarcharging_EVs):
            car_column = solarcharging_cars[ev_index]
            df_diurnal_car_schedule = solarcharging_schedules[ev_index]
            df_ev_solar_charging_Wh_annualts[car_column + '_AtHome'] = np.tile(df_diurnal_car_schedule['at_home'].values, DAYS_PER_YEAR)
            df_ev_solar_charging_Wh_annualts[car_column + '_ChargeMax_Wh'] = np.tile(df_diurnal_car_schedule['E_charge_max_Wh'].values, DAYS_PER_YEAR)
            df_ev_solar_charging_Wh_annualts[car_column + '_Charge_Wh'] = 0
            ev_solar_charging_torefill_Wh[ev_index] = int(df_diurnal_car_schedule['E_to_refill_Wh'].sum())
            
        # Figure out the number of solar charging EVs that are at home
        df_ev_solar_charging_Wh_annualts['Num_EVs_AtHome'] = df_ev_solar_charging_Wh_annualts[[col for col in df_ev_solar_charging_Wh_annualts.columns if col.endswith('_AtHome')]].sum(axis=1)

        # Tackle one day at a time to 
        # Firstly, try and fill up during the day using solar only; and 
        # Secondly, if you cannot, charge from the grid once it is night time (and loop back to the same morning)
        for day_index in range(DAYS_PER_YEAR): # day_index = 2
            df_ev_solar_charging_Wh_dailyts_dayindex = day_index*timesteps_per_day
            df_ev_solar_charging_Wh_dailyts = df_ev_solar_charging_Wh_annualts.iloc[day_index*timesteps_per_day:(day_index+1)*timesteps_per_day].copy()
            df_ev_solar_charging_Wh_dailyts = df_ev_solar_charging_Wh_dailyts.reset_index(drop=True)
            first_solar = -1
            last_solar = -1
            for t in range(timesteps_per_day): # t = 0
                if first_solar < 0 and df_ev_solar_charging_Wh_dailyts['E_excess_solar_Wh'].iloc[t] > 0:
                    first_solar = t
                if df_ev_solar_charging_Wh_dailyts['E_excess_solar_Wh'].iloc[t] > 0:
                    last_solar = t
            df_ev_solar_charging_Wh_dailyts['solar_available'] = (df_ev_solar_charging_Wh_dailyts.index >= first_solar) & (df_ev_solar_charging_Wh_dailyts.index <= last_solar)
            
            # Initialise the daily charging target
            ev_solar_charging_remaining_Wh = ev_solar_charging_torefill_Wh.copy()

            # Step 1: During solar hours, try and fill up each EV evenly
            # ----------------------------------------------------------
            df_solar_hours = df_ev_solar_charging_Wh_dailyts[df_ev_solar_charging_Wh_dailyts['solar_available']]
            for t in df_solar_hours.index: # t = 6
                df_ev_solar_charging_Wh_dailyts_timeindex = df_ev_solar_charging_Wh_dailyts_dayindex + t
                if (df_solar_hours.loc[t, 'E_excess_solar_Wh'] > 0) and (df_solar_hours.loc[t, 'Num_EVs_AtHome'] > 0): # type:ignore

                    # Determine how many EVs can be charged in this timestep
                    num_to_recharge = 0
                    solar_required_per_EV = [0] * num_solarcharging_EVs
                    for ev_index in range(num_solarcharging_EVs): # ev_index = 0
                        car_column = solarcharging_cars[ev_index]
                        if df_solar_hours.at[t, car_column + '_AtHome'] == 1 and ev_solar_charging_remaining_Wh[ev_index] > 0: # type: ignore
                            num_to_recharge += 1
                            charge_needed_Wh = ev_solar_charging_remaining_Wh[ev_index]
                            charge_possible_Wh = min(charge_needed_Wh, df_solar_hours.loc[t, car_column + '_ChargeMax_Wh']) # type: ignore
                            solar_required_per_EV[ev_index] = int(charge_possible_Wh)

                    if num_to_recharge > 0:
                        # Setup the charging decision object
                        df_charging_decision = pd.DataFrame(index = solarcharging_cars, columns = ['solar_required_per_EV'], data = solar_required_per_EV, dtype = int)
                        df_charging_decision['charge_allocated'] = 0
                        df_charging_decision['charge_given'] = 0                    
                        # and sort from smallest to biggest
                        df_charging_decision.sort_values(by='solar_required_per_EV', inplace=True)

                        # Start by assigning each EV an even allocation of the excess solar 
                        solar_energy_available_Wh = int(df_solar_hours.loc[t, 'E_excess_solar_Wh']) # type:ignore #solar_energy_available_Wh = 913
                        even_solar_Wh_per_EV = int(np.floor(solar_energy_available_Wh / num_to_recharge)) # type:ignore
                        for ev_index in range(num_solarcharging_EVs): # ev_index = 0
                            car_name = df_charging_decision.index[ev_index]
                            if ev_index == (num_solarcharging_EVs - 1):
                                df_charging_decision.loc[car_name, 'charge_allocated'] = solar_energy_available_Wh
                            else:
                                df_charging_decision.loc[car_name, 'charge_allocated'] = even_solar_Wh_per_EV
                                solar_energy_available_Wh -= even_solar_Wh_per_EV

                        # Iterate the charging decision - if any EV is allocated more solar than it requires to fill up, give it what it needs and then reallocate the excess solar to the remaining cars, and repeat until all excess solar is allocated or all cars are fully charged
                        for ev_index in range(num_solarcharging_EVs):
                            car_name = df_charging_decision.index[ev_index]
                            if df_charging_decision.loc[car_name, 'solar_required_per_EV'] <= df_charging_decision.loc[car_name, 'charge_allocated']: # type: ignore
                                df_charging_decision.loc[car_name, 'charge_given'] = df_charging_decision.loc[car_name, 'solar_required_per_EV']
                                extra_allocation = int(df_charging_decision.loc[car_name, 'charge_allocated'] - df_charging_decision.loc[car_name, 'solar_required_per_EV']) # type: ignore
                                cars_remaining = num_solarcharging_EVs - (ev_index + 1)
                                if cars_remaining > 0:
                                    even_extra_allocation = int(np.floor(extra_allocation / cars_remaining))
                                    for ev_index2 in range(ev_index + 1, num_solarcharging_EVs):                                    
                                        car_name2 = df_charging_decision.index[ev_index2]
                                        df_charging_decision.loc[car_name2, 'charge_allocated'] += even_extra_allocation # type: ignore
                                        extra_allocation -= even_extra_allocation
                                    # If there is any remaining extra allocation after giving each remaining car an even extra allocation, give it to the next car in line
                                    if extra_allocation > 0:
                                        car_name2 = df_charging_decision.index[ev_index + 1]
                                        df_charging_decision.loc[car_name2, 'charge_allocated'] += extra_allocation # type: ignore
                            else:
                                df_charging_decision.loc[car_name, 'charge_given'] = df_charging_decision.loc[car_name, 'charge_allocated']
                        
                        # Update the solar charging profile and remaining charge to fill for each EV based on the charging decision
                        for ev_index in range(num_solarcharging_EVs): # ev_index = 0
                            car_name = df_charging_decision.index[ev_index]
                            charge_given_Wh = int(df_charging_decision.loc[car_name, 'charge_given']) # type: ignore
                            df_ev_solar_charging_Wh_annualts.at[df_ev_solar_charging_Wh_dailyts_timeindex, car_name + '_Charge_Wh'] = charge_given_Wh # type: ignore
                            ev_solar_charging_remaining_Wh[ev_index] -= charge_given_Wh
                    #df_ev_solar_charging_Wh_annualts[48:60]

            # Step 2: During evening and night hours, if there are EVs that still need to be charged, charge the EV from the grid
            # -------------------------------------------------------------------------------------------------------------------
            if sum(ev_solar_charging_remaining_Wh) > 0:
                df_nonsolar_hours = df_ev_solar_charging_Wh_dailyts[~df_ev_solar_charging_Wh_dailyts['solar_available']]
                sunset_timestep = last_solar + 1
                for x in range(len(df_nonsolar_hours)): # x = 2
                    t = x + sunset_timestep
                    if t >= timesteps_per_day:
                        t = t - timesteps_per_day
                    df_ev_solar_charging_Wh_dailyts_timeindex = df_ev_solar_charging_Wh_dailyts_dayindex + t
                    for ev_index in range(num_solarcharging_EVs): # ev_index = 0
                        car_column = solarcharging_cars[ev_index]
                        if df_nonsolar_hours.at[t, car_column + '_AtHome'] == 1 and ev_solar_charging_remaining_Wh[ev_index] > 0: # type: ignore
                            charge_needed_Wh = ev_solar_charging_remaining_Wh[ev_index]
                            charge_possible_Wh = int(df_nonsolar_hours.loc[t, car_column + '_ChargeMax_Wh']) # type: ignore
                            charge_Wh = min(charge_needed_Wh, charge_possible_Wh)
                            df_ev_solar_charging_Wh_annualts.at[df_ev_solar_charging_Wh_dailyts_timeindex, car_column + '_Charge_Wh'] = charge_Wh # type: ignore
                            ev_solar_charging_remaining_Wh[ev_index] -= charge_Wh

            # Step 3: If there is still charging required, charge from public chargers            
            # ------------------------------------------------------------------------
            if sum(ev_solar_charging_remaining_Wh) > 0:
                public_charging_required_Wh += sum(ev_solar_charging_remaining_Wh)
                
        # Update the total public charging requirements
        transport_public_charging_energy_kWh += int(np.round(public_charging_required_Wh / 1000))
        
        # Update the charging timeseries for these solar charging EVs
        for ev_index in range(num_solarcharging_EVs):
            car_column = solarcharging_cars[ev_index]
            df_ev_charging_Wh_annualts[car_column] = df_ev_solar_charging_Wh_annualts[car_column + '_Charge_Wh']

        # df_ev_solar_charging_Wh_annualts.to_excel('df_ev_solar_charging_Wh_annualts.xlsx')
        # df_ev_charging_Wh_annualts[0:48]
        
    # END IF there are solar charging EVs
        
    # Obtain the overall EV charging timeseries
    ts_transport_electricity_Wh = df_ev_charging_Wh_annualts.sum(axis=1).values
   
    transport_fuel_energy_kWh = int(np.round(transport_fuel_energy_MJ * CONVERT_MJ_TO_KWH))

    return [transport_petrolfuel_energy_MJ, transport_dieselfuel_energy_MJ, transport_fuel_energy_kWh, transport_public_charging_energy_kWh, ts_transport_electricity_Wh]
    
@st.cache_data
def create_solar_distribution_statistical(df_solar, location_name, df_datetime=None, solar_capacity=0.0):
    """
    Create box and whisker plots showing the distribution of 24-hour solar generation profiles 
    for the full year and each season.
    
    Parameters:
    - df_solar: Series or DataFrame column with solar generation data (kWh/kWp)
    - location_name: Name of the solar location for the title
    - df_datetime: Optional datetime series for x-axis
    - solar_capacity: Solar panel capacity in kWp
    """
    if df_solar is None or solar_capacity <= 0:
        return None
    
    num_points = len(df_solar)
    
    # Create datetime index
    if df_datetime is not None and len(df_datetime) > 0:
        try:
            datetime_index = pd.to_datetime(df_datetime)
        except:
            datetime_index = pd.date_range(start='2021-01-01', periods=num_points, freq='h')
    else:
        datetime_index = pd.date_range(start='2021-01-01', periods=num_points, freq='h')
    
    # Create DataFrame with datetime and solar generation
    df = pd.DataFrame({
        'datetime': datetime_index,
        'solar': df_solar.values * solar_capacity
    })
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    
    # Define seasons (Southern Hemisphere)
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Summer'
        elif month in [3, 4, 5]:
            return 'Autumn'
        elif month in [6, 7, 8]:
            return 'Winter'
        else:
            return 'Spring'
    
    df['season'] = df['month'].apply(get_season)
    
    # Create the figure with subplots - Annual on top row, seasons on bottom
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=('Annual Distribution of PV Generation',
                       'Summer', 'Autumn', 'Winter', 'Spring'),
        vertical_spacing=0.2,
        horizontal_spacing=0.08,
        specs=[[{"colspan": 4}, None, None, None],
               [{}, {}, {}, {}]]
    )
    
    hours = list(range(24))
    
    # Annual solar distribution (box plot for each hour)
    for hour in hours:
        hour_data = df[df['hour'] == hour]['solar'].values
        fig.add_trace(go.Box(
            y=hour_data,
            name=str(hour),
            marker_color='#FFA500',
            showlegend=False,
            boxmean='sd'  # Show mean and standard deviation
        ), row=1, col=1)
    
    # Summer distribution
    df_summer = df[df['season'] == 'Summer']
    for hour in hours:
        hour_data = df_summer[df_summer['hour'] == hour]['solar'].values
        fig.add_trace(go.Box(
            y=hour_data,
            name=str(hour),
            marker_color='#FFA500',
            showlegend=False,
            boxmean='sd'
        ), row=2, col=1)
    
    # Autumn distribution
    df_autumn = df[df['season'] == 'Autumn']
    for hour in hours:
        hour_data = df_autumn[df_autumn['hour'] == hour]['solar'].values
        fig.add_trace(go.Box(
            y=hour_data,
            name=str(hour),
            marker_color='#FFA500',
            showlegend=False,
            boxmean='sd'
        ), row=2, col=2)
    
    # Winter distribution
    df_winter = df[df['season'] == 'Winter']
    for hour in hours:
        hour_data = df_winter[df_winter['hour'] == hour]['solar'].values
        fig.add_trace(go.Box(
            y=hour_data,
            name=str(hour),
            marker_color='#FFA500',
            showlegend=False,
            boxmean='sd'
        ), row=2, col=3)
    
    # Spring distribution
    df_spring = df[df['season'] == 'Spring']
    for hour in hours:
        hour_data = df_spring[df_spring['hour'] == hour]['solar'].values
        fig.add_trace(go.Box(
            y=hour_data,
            name=str(hour),
            marker_color='#FFA500',
            showlegend=False,
            boxmean='sd'
        ), row=2, col=4)
    
    # Calculate the maximum value across all data for consistent y-axis
    max_value = df['solar'].max()
    y_max = max_value * 1.1  # Add 10% margin
    
    # Update axes with consistent y-axis range
    # Row 1 (Annual Solar)
    fig.update_xaxes(title_text="Hour of Day", row=1, col=1, range=[-0.5, 23.5])
    fig.update_yaxes(title_text="PV Generation (kW)", row=1, col=1, range=[0, y_max])
    
    # Row 2 (Seasonal Solar)
    for j in range(1, 5):
        fig.update_xaxes(title_text="Hour of Day", row=2, col=j, range=[-0.5, 23.5])
        fig.update_yaxes(title_text="PV Generation (kW)", row=2, col=j, range=[0, y_max])
    
    fig.update_layout(
        title_text=f"{location_name} with a {solar_capacity:.1f} kWp solar panel",
        height=600,
        template='plotly_white',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig


#%% ===========================================================================
# Streamlit Dashboard
# =============================================================================
# scenario_param = SCENARIO_PARAM
def display_household_infographic(scenario_param):
    """Render an icon-card infographic summarising the household configuration."""

    household   = scenario_param.get('meter_name', '—')
    location    = scenario_param.get('solar_name', '—')
    occupants   = scenario_param.get('occupants', '—')
    solar_kw    = scenario_param.get('solar_capacity', 0)
    battery_kwh = scenario_param.get('battery_energy_capacity', 0)
    battery_kw  = scenario_param.get('battery_power_capacity', 0)
    heating     = scenario_param.get('heating', '—')
    heating_use = scenario_param.get('selected_heating_usage', '—')
    cooling     = scenario_param.get('cooling', '—')
    cooling_use = scenario_param.get('selected_cooling_usage', '—')
    hotwater    = scenario_param.get('hotwater', '—')
    cooking     = scenario_param.get('cooking', '—')
    num_cars    = scenario_param.get('num_cars', 0)
    car_types   = scenario_param.get('car_types', ())

    card_base = (
        "background:#1e2130;border-radius:12px;padding:14px 12px;"
        "min-width:100px;max-width:160px;text-align:center;flex:1;"
    )

    def card(icon, label, value, color="#aab4d0"):
        return (
            f'<div style="{card_base}">'
            f'  <div style="font-size:2rem;line-height:1;">{icon}</div>'
            f'  <div style="font-size:0.65rem;color:#7a8499;text-transform:uppercase;'
            f'       letter-spacing:0.06em;margin-top:6px;">{label}</div>'
            f'  <div style="font-size:0.9rem;font-weight:600;color:{color};margin-top:4px;">{value}</div>'
            f'</div>'
        )

    cards = []

    # Location / household
    cards.append(card("🏠", "Location", household + " in " + location, "#6baed6"))

    # Occupants
    cards.append(card("👥", "Occupants", str(occupants), "#74c476"))

    # Solar PV — only show if installed
    if solar_kw and solar_kw > 0:
        cards.append(card("☀️", "Solar PV", f"{solar_kw:.1f} kWp", "#fdd835"))

    # Battery — only show if installed
    if battery_kwh and battery_kwh > 0:
        cards.append(card("🔋", "Battery", f"{battery_kwh:.1f} kWh / {battery_kw:.1f} kW", "#4dd0e1"))

    # Heating
    h_icon = {'Gas': '🔥', 'Electric': '⚡', 'Air-Con': '⚡'}.get(str(heating), '🌡️')
    str_heating = '-'
    if heating != 'None':
        str_heating = str(heating + " (" + heating_use + ")")
    cards.append(card(h_icon, "Heating", str_heating, "#ef9a9a"))

    # Cooling
    c_icon = {'Air-Con': '⚡', 'Evaporative': '⚡', 'None': '—'}.get(str(cooling), '❄️')
    str_cooling = '-'
    if cooling != 'None':
        str_cooling = str(cooling + " (" + cooling_use + ")")
    cards.append(card(c_icon, "Cooling", str_cooling, "#90caf9"))

    # Hot water
    hw_icon = {'Gas': '🔥', 'Electric': '⚡', 'Heat-Pump': '⚡', 'Solar': '☀️'}.get(str(hotwater), '🚿')
    cards.append(card(hw_icon, "Hot Water", str(hotwater) if hotwater else '—', "#a5d6a7"))

    # Cooking
    ck_icon = {'Gas': '🔥', 'Electric': '⚡', 'Induction': '⚡'}.get(str(cooking), '🍳')
    cards.append(card(ck_icon, "Cooking", str(cooking) if cooking else '—', "#ffcc80"))

    # Cars
    if num_cars and num_cars > 0:
        fuel_icons = {'Petrol': '⛽', 'Diesel': '🛢️', 'Electric': '⚡'}
        car_str = ' '.join(fuel_icons.get(t, '🚗') for t in car_types) if car_types else '🚗' * num_cars
        cards.append(card("🚗", f"Cars ({num_cars})", car_str, "#ce93d8"))

    html = (
        '<div style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:28px;">'
        + ''.join(cards)
        + '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# scenario_param = SCENARIO_PARAM

@st.cache_data
def create_streamlit_dashboard(df_results, summary, df_opt_exogenous_timeseries, scenario_param):
    """
    Create an interactive Streamlit dashboard with sliders for PV and battery capacity.
    """
    st.title("Household Energy Flow Analysis")

    # Household configuration infographic
    display_household_infographic(scenario_param)

    pv_capacity = scenario_param['solar_capacity']
    batt_capacity = scenario_param['battery_energy_capacity']
    
   # Display Word Cloud
    st.subheader("Energy Outcomes")
    # Center the bubble chart using columns
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        wordcloud_fig = create_metrics_wordcloud(summary)
        st.plotly_chart(wordcloud_fig, width='content')
    
    # Display Sankey diagram
    fig = create_energy_sankey(df_results, summary)
    st.plotly_chart(fig, width='stretch')

    # Display environmental outcomes in a waterfall chart
    st.subheader("Environmental Outcomes")

    ghg_fig = create_ghg_stacked_bar(summary, scenario_param)
    _ghg_left, _ghg_mid, _ghg_right = st.columns([1, 2, 1])
    with _ghg_mid:
        st.plotly_chart(ghg_fig, use_container_width=True)

    # Display economic outcomes in a waterfall chart
    st.subheader("Bill Savings with PV + Batteries")
    original_bill_import = (df_results['E_underlyingload_Wh'] /1000 * df_opt_exogenous_timeseries['C_tariff_import_$/kWh']).sum()
    original_bill_fixed = (df_opt_exogenous_timeseries['C_tariff_fixed_$/day']).sum()
    original_bill = original_bill_import + original_bill_fixed
    bill_reduction_import = ((df_results['E_underlyingload_Wh'] - df_results['E_final_grid_import_Wh']) /1000 * df_opt_exogenous_timeseries['C_tariff_import_$/kWh']).sum()
    bill_value_export = (df_results['E_final_grid_export_Wh'] /1000 * df_opt_exogenous_timeseries['C_tariff_export_$/kWh']).sum()
    new_bill = (df_results['E_final_grid_import_Wh'] /1000 * df_opt_exogenous_timeseries['C_tariff_import_$/kWh'] - df_results['E_final_grid_export_Wh'] /1000 * df_opt_exogenous_timeseries['C_tariff_export_$/kWh'] + df_opt_exogenous_timeseries['C_tariff_fixed_$/day']).sum()
    waterfall_fig = create_bill_savings_waterfall(original_bill, bill_reduction_import, bill_value_export, new_bill, original_bill_import, original_bill_fixed)
    st.plotly_chart(waterfall_fig, use_container_width=True)
    
    
    # Additional details in expandable section
    with st.expander("🔍 Detailed Breakdown"):

        # Display key metrics
        st.subheader("Key Metrics")
        metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
        
        with metric_col1:
            st.metric("Total Load", f"{summary['total_underlyingload_kWh']:.0f} kWh")
        with metric_col2:
            st.metric("Grid Export", f"{summary['total_pv_grid_kWh']:.0f} kWh")
        with metric_col3:
            st.metric("Grid Import", f"{summary['total_grid_load_kWh']:.0f} kWh")
        with metric_col4:
            self_consumption = (summary['total_pv_load_kWh'] + summary['total_pv_batt_kWh']) / summary['total_pv_generation_kWh'] * 100 if summary['total_pv_generation_kWh'] > 0 else 0
            st.metric("PV Self-Consumption", f"{self_consumption:.1f}%")
        with metric_col5:
            grid_dependence = (summary['total_grid_load_kWh'] / summary['total_underlyingload_kWh'] * 100)
            st.metric("Grid Dependence", f"{grid_dependence:.1f}%")

        detail_col1, detail_col2 = st.columns(2)
        
        with detail_col1:
            st.write("**PV Energy Distribution:**")
            st.write(f"- Direct to Load: {summary['total_pv_load_kWh']:.0f} kWh")
            st.write(f"- To Battery: {summary['total_pv_batt_kWh']:.0f} kWh")
            st.write(f"- Exported to Grid: {summary['total_pv_grid_kWh']:.0f} kWh")
            st.write(f"- Spilled: {summary['total_pv_curtailed_kWh']:.0f} kWh")
        
        with detail_col2:
            st.write("**Battery Performance:**")
            st.write(f"- Energy Discharged: {summary['total_batt_load_kWh']:.0f} kWh")
            st.write(f"- Total Losses: {summary['total_batt_losses_kWh']:.0f} kWh")
            roundtrip_eff = (summary['total_batt_load_kWh'] / summary['total_pv_batt_kWh'] * 100) if summary['total_pv_batt_kWh'] > 0 else 0
            st.write(f"- Round-trip Efficiency: {roundtrip_eff:.1f}%")

    with st.expander("📈 Average Daily Profiles (from simulation)", expanded=False):
        sim_profiles_fig = create_average_daily_simulation_profiles(df_results, scenario_param, df_opt_exogenous_timeseries)
        st.plotly_chart(sim_profiles_fig, use_container_width=True)


#%% ===========================================================================
# Cached data loading functions
# =============================================================================
@st.cache_data
def load_input_data(input_file):
    """Load all input data from Excel file. Cached to avoid reloading on every rerun."""
    df_xls_default_param = pd.read_excel(input_file, sheet_name='Default', engine='openpyxl', index_col=0)
    df_xls_static_param = pd.read_excel(input_file, sheet_name='Static', engine='openpyxl', index_col=0)
    df_xls_timedependent_param = pd.read_excel(input_file, sheet_name='TimeDependent', engine='openpyxl', index_col=0)
    df_xls_datetime_ts = pd.read_excel(input_file, sheet_name='DateTime', engine='openpyxl', index_col=0)
    df_xls_meter_ts = pd.read_excel(input_file, sheet_name='Meter_TS', engine='openpyxl', index_col=0)
    df_xls_meter_to_datetime = pd.read_excel(input_file, sheet_name='Meter_to_DateTime', engine='openpyxl', index_col=0)
    df_xls_heating_ts = pd.read_excel(input_file, sheet_name='Heating_TS', engine='openpyxl', index_col=0)
    df_xls_cooling_ts = pd.read_excel(input_file, sheet_name='Cooling_TS', engine='openpyxl', index_col=0)
    df_xls_gas_usage = pd.read_excel(input_file, sheet_name='Gas_Usage', engine='openpyxl', index_col=0)
    df_xls_solar_ts = pd.read_excel(input_file, sheet_name='Solar_TS', engine='openpyxl', index_col=0)
    df_xls_tariff_ts = pd.read_excel(input_file, sheet_name='Tariff_TS', engine='openpyxl', index_col=0)
    return (df_xls_default_param, df_xls_static_param, df_xls_timedependent_param,
            df_xls_datetime_ts, df_xls_meter_ts, df_xls_meter_to_datetime, df_xls_heating_ts, df_xls_cooling_ts, df_xls_gas_usage, df_xls_solar_ts, df_xls_tariff_ts)


#%%
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# =============================================================================
# Main Streamlit Application
# =============================================================================
def main():
    """Main Streamlit application function."""
    # Load the input data (cached)
    (df_xls_initial_param, 
     df_xls_static_param, 
     df_xls_timedependent_param,
     df_xls_datetime_ts, 
     df_xls_meter_ts, 
     df_xls_meter_to_datetime, 
     df_xls_heating_ts,
     df_xls_cooling_ts,
     df_xls_gas_usage,
     df_xls_solar_ts, 
     df_xls_tariff_ts) = load_input_data(INPUT_FILE)

    df_xls_initial_param['Value'] = df_xls_initial_param['Value'].replace(np.nan, 'None')

    # Convert Excel columns to proper types
    df_xls_static_param['Value'] = df_xls_static_param['Value'].astype(float)

    # Setup the common static parameters of the analysis
    PARAM = {}
    l_attribute_names = df_xls_static_param.index.tolist()
    for i in range(len(l_attribute_names)):
        if df_xls_static_param.at[l_attribute_names[i], 'Unit'] == '%':
            PARAM[l_attribute_names[i]] = df_xls_static_param.at[l_attribute_names[i], 'Value'] / 100 # type: ignore
        else:
            PARAM[l_attribute_names[i]] = df_xls_static_param.at[l_attribute_names[i], 'Value']

    # Setup the sidebar configuration options
    st.sidebar.header("⚙️ Household Configuration")
    
    
    occupant_options = list(range(1, 11))  # Assuming 1 to 10 occupants
    try:
        default_occupants = int(str(df_xls_initial_param.at['Occupants', 'Value']))
    except (KeyError, ValueError):
        # Fallback to first option if default not found
        default_occupants = 3
    _col_occ, _ = st.columns(2)
    with _col_occ:
        selected_occupants = st.sidebar.number_input(
            "Occupants",
            min_value=1,
            value=default_occupants
        )

    with st.sidebar.expander("Geography/Region 🌍"):

        with st.expander("Region"):

            # Show the available solar locations options from the Excel file
            solar_location_options = df_xls_solar_ts.columns.tolist()
            try:
                default_solar_location = str(df_xls_initial_param.at['Solar', 'Value'])
                default_solar_location_index = solar_location_options.index(default_solar_location)
            except (KeyError, ValueError):
                default_solar_location_index = 0
            selected_location = st.selectbox(
                "Location",
                options=solar_location_options,
                index=default_solar_location_index
            )
            
        with st.expander("Household electricity profile"):

            # Show the household meter data options from the Excel file
            household_options = df_xls_meter_ts.columns.tolist()
            try:
                default_household = str(df_xls_initial_param.at['Household', 'Value'])
                default_household_index = household_options.index(default_household)        
            except (KeyError, ValueError):
                # Fallback to first column if default not found
                default_household_index = 0
            selected_household = st.selectbox(
                "Load Profile",
                options=household_options,
                index=default_household_index
            )

    with st.sidebar.expander("Technology Choices ⚡"):

        with st.expander("Heating and Cooling 🌡️"):

            # Show the heating type options from the Excel file
            heating_options = ['Electric', 'Air-Con', 'Gas', 'None']
            try:
                default_heating = str(df_xls_initial_param.at['Heating_Type', 'Value'])
                default_heating_index = heating_options.index(default_heating)        
            except (KeyError, ValueError):
                # Fallback to first option if default not found
                default_heating_index = 2
            selected_heating = st.selectbox(
                "Heating Type",
                options=heating_options,
                index=default_heating_index
            )

            HEATING_USAGE_SCALE = {'Frugal': 0.7, 'Typical': 1.0, 'Comfort': 1.4}
            selected_heating_usage = st.select_slider(
                "Usage Level",
                options=['Frugal (70%)', 'Typical (100%)', 'Comfort (140%)'],
                value='Typical (100%)',
                disabled=(selected_heating == 'None'),
                key=f'heating_usage_slider',
            )
            selected_heating_usage_scale = HEATING_USAGE_SCALE[selected_heating_usage.split(' ')[0]]

            # Cooling options
            cooling_options = ['Air-Con', 'Evaporative', 'None']
            try:
                default_cooling = str(df_xls_initial_param.at['Cooling_Type', 'Value'])
                default_cooling_index = cooling_options.index(default_cooling)        
            except (KeyError, ValueError):
                # Fallback to first option if default not found
                default_cooling_index = 2
            selected_cooling = st.selectbox(
                "Cooling Type",
                options=cooling_options,
                index=default_cooling_index
            )
            
            COOLING_USAGE_SCALE = {'Frugal': 0.7, 'Typical': 1.0, 'Comfort': 1.4}
            selected_cooling_usage = st.select_slider(
                "Usage Level",
                options=['Frugal (70%)', 'Typical (100%)', 'Comfort (140%)'],
                value='Typical (100%)',
                disabled=(selected_cooling == 'None'),
                key=f'cooling_usage_slider',
            )
            selected_cooling_usage_scale = COOLING_USAGE_SCALE[selected_cooling_usage.split(' ')[0]]
            
            # Show the hot water type options from the Excel file
            hotwater_options = ['Gas', 'Heat-Pump', 'Electric']
            try:
                default_hotwater = str(df_xls_initial_param.at['Hotwater_Type', 'Value'])
                default_hotwater_index = hotwater_options.index(default_hotwater)        
            except (KeyError, ValueError):
                # Fallback to first option if default not found
                default_hotwater_index = 0
            selected_hotwater = st.selectbox(
                "Hot Water Type",
                options=hotwater_options,
                index=default_hotwater_index
            )
            

        with st.expander("Cooking 🍳"):

            # Show the cooking type options from the Excel file
            cooking_options = ['Gas', 'Electric', 'Induction']
            try:
                default_cooking = str(df_xls_initial_param.at['Cooking_Type', 'Value'])
                default_cooking_index = cooking_options.index(default_cooking)        
            except (KeyError, ValueError):
                # Fallback to first option if default not found
                default_cooking_index = 0
            selected_cooking = st.selectbox(
                "Cooking Type",
                options=cooking_options,
                index=default_cooking_index
            )
            
        with st.expander("Transport 🚗"):

            default_numcars = 2
            _col_cars, _ = st.columns(2)
            with _col_cars:
                selected_numcars = st.number_input(
                    "Number of Cars",
                    min_value=0,
                    value=default_numcars
                )

            selected_car_types = []
            selected_car_distances = []
            selected_car_efficiencies = []
            selected_car_schedules = []  # dict with depart_hour/arrive_hour for EVs, None for ICE
            selected_car_charger_speeds = []
            selected_car_charging_strategies = []
            for i in range(1, selected_numcars + 1):
                with st.expander(f"Car {i}", expanded=True):
                    car_type_options = ['Petrol', 'Diesel', 'Electric']
                    try:
                        default_car_type = str(df_xls_initial_param.at[f'Car_{i}_Type', 'Value'])
                        default_car_distance = int(str(df_xls_initial_param.at[f'Car_{i}_Distance', 'Value']))
                        default_car_type_index = car_type_options.index(default_car_type)
                    except (KeyError, ValueError):
                        # Fallback to first option if default not found
                        default_car_type_index = 0
                        default_car_distance = 12500
                    car_type = st.selectbox(
                        f"Fuel Type",
                        options=car_type_options,
                        index=default_car_type_index,
                        key=f'car_{i}_type'
                    )
                    car_distance = st.number_input(
                        f"Distance Travelled (km)",
                        min_value=0,
                        value=default_car_distance,
                        key=f'car_{i}_distance',
                        step=100,
                    )
                    # Include car_type in the efficiency key so a new widget (with the correct
                    # default value) is created whenever the fuel type changes.
                    if car_type == 'Petrol':
                        car_efficiency = st.number_input(
                            f"Fuel Efficiency (L/100km)",
                            min_value=0.0,
                            value=PARAM['default_petrol_eff'],
                            key=f'car_{i}_efficiency_{car_type}',
                            step=0.1,
                        )
                        car_schedule = None
                        charger_speed = None
                        charging_strategy = None
                    elif car_type == 'Diesel':
                        car_efficiency = st.number_input(
                            f"Fuel Efficiency (L/100km)",
                            min_value=0.0,
                            value=PARAM['default_diesel_eff'],
                            key=f'car_{i}_efficiency_{car_type}',
                            step=0.1,
                        )
                        car_schedule = None
                        charger_speed = None
                        charging_strategy = None
                    else:
                        car_efficiency = st.number_input(
                            f"Energy Efficiency (kWh/100km)",
                            min_value=0.0,
                            value=PARAM['default_electr_eff'],
                            key=f'car_{i}_efficiency_{car_type}',
                            step=0.1,
                        )
                        st.markdown("*Typical Daily Schedule:*")
                        
                        # Check if using default car_type and car_distance
                        using_defaults = (car_type == default_car_type and car_distance == default_car_distance)
                        
                        schedule_key = f'car_{i}_saved_schedule'

                        if using_defaults:
                            # Try to load car_sequence from Excel to seed the initial schedule on first load
                            try:
                                car_sequence_str = str(df_xls_initial_param.at[f'Car_{i}_Sequence', 'Value'])
                                # Parse the sequence string (e.g., "0700-0800; 1000-1300")
                                excel_time_blocks = []
                                if car_sequence_str and car_sequence_str != 'None':
                                    
                                    time_splits = car_sequence_str.split(';')                                
                                    tr_index = 0
                                    for tr_index in range(len(time_splits)):                                    
                                        time_range = time_splits[tr_index]
                                        if '-' in time_range and len(time_range) >= 9:
                                            depart_str, arrive_str = time_range.split('-')
                                            depart_hour = int(depart_str[:2])
                                            depart_minute = int(depart_str[2:4]) if len(depart_str) >= 4 else 0
                                            arrive_hour = int(arrive_str[:2])
                                            arrive_minute = int(arrive_str[2:4]) if len(arrive_str) >= 4 else 0
                                            excel_time_blocks.append({
                                                'depart_hour': depart_hour,
                                                'depart_minute': depart_minute,
                                                'arrive_hour': arrive_hour,
                                                'arrive_minute': arrive_minute
                                            })
                                
                                # Seed session state with Excel values only on first load (don't overwrite user edits)
                                if excel_time_blocks and schedule_key not in st.session_state:
                                    st.session_state[schedule_key] = excel_time_blocks
                            except (KeyError, ValueError):
                                pass
                        
                        # Always show the widget-based schedule editor (Excel data seeds the initial values)
                        blocks_key = f'car_{i}_num_blocks'
                        saved_schedule = st.session_state.get(schedule_key, [])
                        if blocks_key not in st.session_state:
                            st.session_state[blocks_key] = len(saved_schedule) if saved_schedule else 1
                        time_blocks = []
                        for b in range(1, st.session_state[blocks_key] + 1):
                            depart_key = f'car_{i}_depart_{b}'
                            arrive_key = f'car_{i}_arrive_{b}'
                            # Initialise widget state if not already set (from saved schedule or PARAM defaults)
                            # Must set via session state only — do not also pass value= to the widget
                            if depart_key not in st.session_state:
                                if saved_schedule and b <= len(saved_schedule):
                                    st.session_state[depart_key] = datetime_time(saved_schedule[b-1]['depart_hour'], saved_schedule[b-1]['depart_minute'])
                                else:
                                    st.session_state[depart_key] = datetime_time(int(np.floor(PARAM['default_electr_stime'])), int((PARAM['default_electr_stime'] % 1) * 60))
                            if arrive_key not in st.session_state:
                                if saved_schedule and b <= len(saved_schedule):
                                    st.session_state[arrive_key] = datetime_time(saved_schedule[b-1]['arrive_hour'], saved_schedule[b-1]['arrive_minute'])
                                else:
                                    st.session_state[arrive_key] = datetime_time(int(np.floor(PARAM['default_electr_etime'])), int((PARAM['default_electr_etime'] % 1) * 60))
                            col_dep, col_arr, col_del = st.columns([2, 2, 1])
                            with col_del:
                                # Show delete button only when there is more than one block
                                if st.session_state[blocks_key] > 1:
                                    if st.button("✕", key=f'car_{i}_del_block_{b}', help="Remove this block"):
                                        st.session_state[blocks_key] -= 1
                                        st.rerun()
                            with col_dep:
                                dep_time = st.time_input(
                                    "Depart",
                                    step=3600,
                                    key=depart_key
                                )
                            with col_arr:
                                arr_time = st.time_input(
                                    "Arrive",
                                    step=3600,
                                    key=arrive_key
                                )
                            time_blocks.append({'depart_hour': dep_time.hour, 'depart_minute': dep_time.minute,
                                                'arrive_hour': arr_time.hour, 'arrive_minute': arr_time.minute})
                        if st.button("＋ Add time block", key=f'car_{i}_add_block'):
                            st.session_state[blocks_key] += 1
                            st.rerun()
                        # Save the completed schedule so it survives fuel type changes
                        st.session_state[schedule_key] = time_blocks

                        charging_strategy = st.selectbox(
                            "Charging Strategy",
                            options=[
                                "Immediately upon return",
                                "Just before departure",
                                "Overnight only (11pm-7am)",
                                "Midday only (11am-1pm)",
                                "Solar self-charging first",
                            ],
                            key=f'car_{i}_charging_strategy',
                        )

                        charger_speed = st.selectbox(
                            "Charging Speed",
                            options=[
                                "Level 1 (2.4 kW)",
                                "Level 2 single phase (7.2 kW)",
                                "Level 2 three phase (22 kW)",
                            ],
                            key=f'car_{i}_charger_speed'
                        )

                        car_schedule = time_blocks
                        
                    selected_car_types.append(car_type)
                    selected_car_distances.append(car_distance)
                    selected_car_efficiencies.append(car_efficiency)
                    selected_car_schedules.append(car_schedule)
                    selected_car_charger_speeds.append(charger_speed)
                    selected_car_charging_strategies.append(charging_strategy)

        with st.expander("Rooftop PV ☀️ "):

            # Solar panel capacity
            try:
                default_solar_capacity = float(str(df_xls_initial_param.at['SolarPV_capacity_kW', 'Value']))
            except (KeyError, ValueError):
                default_solar_capacity = 5.0
            selected_solar_capacity = st.number_input(
                "Rated Capacity (kWp)",
                min_value=0.0,
                max_value=25.0,
                value=default_solar_capacity,
                step=1.0,
                format="%.1f"
            )

        with st.expander("Batteries 🔋", expanded = True):
            # Battery storage capacity
            try:
                default_battery_capacity = float(str(df_xls_initial_param.at['Battery_capacity_kWh', 'Value']))
            except (KeyError, ValueError):
                default_battery_capacity = 5.0
            selected_battery_capacity = st.number_input(
                "Storage Capacity (kWh)",
                min_value=0.0,
                max_value=50.0,
                value=default_battery_capacity,
                step=1.0,
                format="%.1f"
            )

            # Battery inverter capacity
            default_battery_inverter_capacity = 5.0
            selected_battery_inverter_capacity = st.number_input(
                "Inverter Capacity (kW)",
                min_value=0.0,
                max_value=20.0,
                value=default_battery_inverter_capacity,
                step=1.0,
                format="%.0f"
            )
        
    with st.sidebar.expander("Financial 💲"):
        
        st.subheader("⚡ Electricity")
        # Import electricity tariff
        try:
            default_import_tariff_name = df_xls_initial_param.at['Import_Tariff', 'Value']
            default_import_tariff = float(str(df_xls_tariff_ts.at[0, default_import_tariff_name]))
        except (KeyError, ValueError):
            default_import_tariff = 0.34
        selected_import_tariff = st.number_input(
            "Usage Charge ($/kWh)",
            min_value=0.0,
            max_value=1.0,
            value=default_import_tariff,
            step=0.01,
            format="%.3f"
        )
        
        # Export electricity tariff
        try:
            default_export_tariff_name = df_xls_initial_param.at['Export_Tariff', 'Value']
            default_export_tariff = float(str(df_xls_tariff_ts.at[0, default_export_tariff_name]))
        except (KeyError, ValueError):
            default_export_tariff = 0.03
        selected_export_tariff = st.number_input(
            "Feed-in Tariff ($/kWh)",
            min_value=0.0,
            max_value=1.0,
            value=default_export_tariff,
            step=0.01,
            format="%.3f"
        )
        
        # Daily electricity charge
        try:
            default_daily_fixed_charges_name = df_xls_initial_param.at['Fixed_Charge', 'Value']
            default_daily_fixed_charges = float(str(df_xls_tariff_ts.at[0, default_daily_fixed_charges_name])) * 24
        except (KeyError, ValueError):
            default_daily_fixed_charges = 1.45
        selected_daily_fixed_charges = st.number_input(
            "Daily Fixed Charge ($/day)",
            min_value=0.0,
            max_value=4.0,
            value=default_daily_fixed_charges,
            step=0.01,
            format="%.3f"
        )

        st.subheader("🔥 Natural Gas")
        # Gas charges
        default_gas_usage_charge = 0.035
        selected_gas_usage_charge = st.number_input(
            "Usage Charge ($/MJ)",
            min_value=0.0,
            max_value=4.0,
            value=default_gas_usage_charge,
            step=0.005,
            format="%.3f"
        )

        default_gas_daily_charge = 0.80
        selected_gas_daily_charge = st.number_input(
            "Daily Fixed Charge ($/day)",
            min_value=0.0,
            max_value=4.0,
            value=default_gas_daily_charge,
            step=0.01,
            format="%.3f"
        )

        st.subheader("⛽ Transport Fuel")
        # Petrol charges
        default_petrol_charge = 2.0
        selected_petrol_charge = st.number_input(
            "Cost of Petrol ($/L)",
            min_value=0.0,
            max_value=8.0,
            value=default_petrol_charge,
            step=0.01,
            format="%.3f"
        )

        # Diesel charges
        default_diesel_charge = 2.4
        selected_diesel_charge = st.number_input(
            "Cost of Diesel ($/L)",
            min_value=0.0,
            max_value=8.0,
            value=default_diesel_charge,
            step=0.01,
            format="%.3f"
        )

    with st.sidebar.expander("Environment 🌱"):
        
        st.subheader("⚡ Electricity")
        default_emissionsfactor_electricity = 679
        selected_emissionsfactor_electricity = st.number_input(
            "Emissions factor (kgCO2e/MWh)",
            min_value=0,
            value=default_emissionsfactor_electricity,
            step=10,
            format="%d"
        )

    # selected_household = df_xls_initial_param.at['Household', 'Value']
    # selected_occupants = df_xls_initial_param.at['Occupants', 'Value']
    # selected_location = df_xls_initial_param.at['Solar', 'Value']
    # selected_solar_capacity = df_xls_initial_param.at['SolarPV_capacity_kW', 'Value']
    # selected_battery_capacity = df_xls_initial_param.at['Battery_capacity_kWh', 'Value']
    # selected_import_tariff = float(str(df_xls_tariff_ts.at[0, df_xls_initial_param.at['Import_Tariff', 'Value']]))
    # selected_export_tariff = float(str(df_xls_tariff_ts.at[0, df_xls_initial_param.at['Export_Tariff', 'Value']]))
    # selected_daily_fixed_charges = float(str(df_xls_tariff_ts.at[0, df_xls_initial_param.at['Fixed_Charge', 'Value']])) * 24
    # selected_battery_inverter_capacity = default_battery_inverter_capacity
    # selected_gas_usage_charge = default_gas_usage_charge
    # selected_gas_daily_charge = default_gas_daily_charge
    # selected_petrol_charge = default_petrol_charge
    # selected_diesel_charge = default_diesel_charge
    # selected_emissionsfactor_electricity = default_emissionsfactor_electricity

    # Display the parameters chosen on main page
    with st.expander("Parameters", expanded=False):
        # Household & Basic Info
        st.subheader("Household")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("**Household Type:**")
            st.write("**Occupants:**")
        with col2:
            st.write(selected_household)
            st.write(selected_occupants)
        
        # Energy Assets
        st.subheader("Energy Assets")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("**Solar Location:**")
            st.write("**Solar Capacity:**")
            st.write("**Battery Storage Capacity:**")
            st.write("**Battery Inverter Capacity:**")
        with col2:
            st.write(f"{selected_location}")
            st.write(f"{selected_solar_capacity:.1f} kWp")
            st.write(f"{selected_battery_capacity:.1f} kWh")
            st.write(f"{selected_battery_inverter_capacity:.0f} kW")
        
        # Energy Prices
        st.subheader("Cost of Energy")
        
        # Electricity
        st.write("**⚡ Electricity**")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("**Usage Charge:**")
            st.write("**Feed-in Tariff:**")
            st.write("**Daily Fixed Charge:**")
        with col2:
            st.write(f"{selected_import_tariff*100:.1f} c/kWh")
            st.write(f"{selected_export_tariff*100:.1f} c/kWh")
            st.write(f"${selected_daily_fixed_charges:.2f}/day")
        
        # Gas
        st.write("**🔥 Natural Gas**")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("**Usage Charge:**")
            st.write("**Daily Fixed Charge:**")
        with col2:
            st.write(f"{selected_gas_usage_charge*100:.1f} c/MJ")
            st.write(f"${selected_gas_daily_charge:.2f}/day")
        
        # Petrol/Diesel
        st.write("**⛽ Fuel**")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("**Petrol Cost:**")
            st.write("**Diesel Cost:**")
        with col2:
            st.write(f"{selected_petrol_charge*100:.1f} c/L")
            st.write(f"{selected_diesel_charge*100:.1f} c/L")
        

    # Refactor the household_consumption based on the number of occupants (linear scaling)
    default_occupants = PARAM['occupant_average_per_household']
    occupants_scalingfactor = PARAM['occupant_LR_scalingfactor']
    occupants_bias = PARAM['occupant_LR_bias']
    default_occupant_daily_demand = default_occupants * occupants_scalingfactor + occupants_bias
    selected_occupant_daily_demand = selected_occupants * occupants_scalingfactor + occupants_bias
    demand_scaling_factor = selected_occupant_daily_demand / default_occupant_daily_demand


    # Get consumption data for selected household
    v_household_consumption = df_xls_meter_ts[selected_household] * demand_scaling_factor

    # Get solar generation data for selected location
    solar_generation = df_xls_solar_ts[selected_location]
    
    # Try to get datetime data
    try:
        datetime_col = df_xls_datetime_ts.iloc[:, 0]
    except:
        datetime_col = None
    
    with st.expander("Raw energy traces across the year", expanded=False):

        # Create annual consumption profile chart
        consumption_fig = create_consumption_profile_chart(v_household_consumption, selected_household, datetime_col)
        st.plotly_chart(consumption_fig, width='stretch')

        solarpv_fig = create_solarpv_profile_chart(solar_generation, selected_solar_capacity, datetime_col)
        st.plotly_chart(solarpv_fig, width='stretch')


    # Create average 24-hour profiles with solar overlay
    with st.expander("Average daily consumption & solar profiles", expanded=False):

        daily_profiles_fig = create_average_daily_profiles(
            v_household_consumption, 
            selected_household, 
            datetime_col,
            solar_generation,
            selected_solar_capacity
        )
        st.plotly_chart(daily_profiles_fig, width='stretch')

    with st.expander("Distribution of daily consumption (Box & Whisker)", expanded=False):
    
        daily_profiles_fig_stat = create_average_daily_profiles_statistical(
            v_household_consumption, 
            selected_household, 
            datetime_col
        )
        st.plotly_chart(daily_profiles_fig_stat, width='stretch')
    
    with st.expander("Distribution of daily PV generation (Box & Whisker)", expanded=False):
        solar_profiles_fig_stat = create_solar_distribution_statistical(
            solar_generation,
            selected_location,
            datetime_col,
            selected_solar_capacity
        )
        if solar_profiles_fig_stat is not None:
            st.plotly_chart(solar_profiles_fig_stat, width='stretch')

    with st.expander("Simulation results", expanded=True):
#        st.write("Run the energy flow simulation for the selected configuration and explore the results in an interactive dashboard.")

        # Prepare the exogenous timeseries data for the simulation
        # household_consumption
        # solar_generation
        # selected_solar_capacity
        selected_analysis_year = int(str(df_xls_initial_param.at['Analysis_year', 'Value']))
        
        # Create the parameters for the selected scenario
        SCENARIO_PARAM = PARAM.copy()
        SCENARIO_PARAM['financial_horizon'] = int(SCENARIO_PARAM['financial_horizon'])
    
        # selected_household = 'HOUSE_B'
        # Setup the annual timestamp used in the analysis based on the datetime column name.
        colname_datetime = df_xls_meter_to_datetime.at[selected_household,'DateTime']        
        num_timesteps = df_xls_datetime_ts[colname_datetime].count()
        v_timestamp = np.repeat(0, num_timesteps)
        l_datetime = [None] * num_timesteps
        for time_index in range(num_timesteps):
            v_timestamp[time_index] = int(df_xls_datetime_ts.at[time_index, colname_datetime].timestamp()) # type: ignore
            l_datetime[time_index] = datetime.fromtimestamp(v_timestamp[time_index], tz=pytz.timezone('UTC')) # type: ignore
        time_step = v_timestamp[1] - v_timestamp[0]
        df_timestamp = pd.DataFrame({'Timestamp_begin':v_timestamp,
                                    'Datetime_begin':l_datetime,'Timestamp_end':v_timestamp+time_step})
        SCENARIO_PARAM['timestep_seconds'] = time_step
        BASELINE_YEAR = int(selected_analysis_year)
        SCENARIO_PARAM['meter_name'] = selected_household
        SCENARIO_PARAM['demand_scaling'] = demand_scaling_factor
        SCENARIO_PARAM['solar_name'] = selected_location
        SCENARIO_PARAM['solar_capacity'] = selected_solar_capacity
        SCENARIO_PARAM['battery_energy_capacity'] = selected_battery_capacity
        SCENARIO_PARAM['battery_power_capacity'] = selected_battery_inverter_capacity
        SCENARIO_PARAM['grid_emissionsfactor'] = selected_emissionsfactor_electricity

        # Obtain the meter profile (kWh timeseries)
        df_load_profile = v_household_consumption

        # Obtain the solar PV generation profile (kWh/kWp timeseries)
        colname_solarprofile = SCENARIO_PARAM['solar_name']
        df_solar_profile = df_xls_solar_ts[colname_solarprofile]
        
        # Obtain the import and export tariffs for each hour (use user-selected tariffs)
        v_tariff_import = np.repeat(selected_import_tariff, num_timesteps)
        v_tariff_export = np.repeat(selected_export_tariff, num_timesteps)
        v_tariff_fixed = np.repeat(selected_daily_fixed_charges/24, num_timesteps)
    #     b_consider_demandcharges = v_tariff_demand_charge.sum() > 0
        df_electricity_tariffs = pd.DataFrame({'Import':v_tariff_import,
                                               'Export': v_tariff_export,
                                               'Fixed': v_tariff_fixed})

        # Define the annual tariff adjustments (all 1.0)
        df_scalingfactors = pd.DataFrame(index = range(BASELINE_YEAR,BASELINE_YEAR+SCENARIO_PARAM['financial_horizon']))
        import_tariff_annualscale = df_xls_timedependent_param.loc["sf_tariff_import", BASELINE_YEAR:(BASELINE_YEAR+SCENARIO_PARAM['financial_horizon']-1)]
        df_scalingfactors['Tariff_Import'] = import_tariff_annualscale/import_tariff_annualscale.iat[0] # type: ignore
        export_tariff_annualscale = df_xls_timedependent_param.loc["sf_tariff_export", BASELINE_YEAR:(BASELINE_YEAR+SCENARIO_PARAM['financial_horizon']-1)]
        df_scalingfactors['Tariff_Export'] = export_tariff_annualscale/export_tariff_annualscale.iat[0] # type: ignore
        fixed_tariff_annualscale = df_xls_timedependent_param.loc["sf_tariff_fixed", BASELINE_YEAR:(BASELINE_YEAR+SCENARIO_PARAM['financial_horizon']-1)]
        df_scalingfactors['Tariff_Fixed'] = fixed_tariff_annualscale/fixed_tariff_annualscale.iat[0] # type: ignore

    #    selected_location = 'Sydney'
    #    selected_occupants = 2
    #    selected_heating = 'Gas'
    #    selected_cooling = 'None'
    #    selected_hotwater = 'Gas'
    #    selected_cooking = 'Gas'
        # Figure out how much of the gas demand has been replaced by electricity
        # ----------------------------------------------------------------------
        underlying_gas_demand_reference = df_xls_gas_usage[selected_location]
        
        # Scale the underlying gas demand to the number of occupants
        scaling_factor = selected_occupants / underlying_gas_demand_reference['Occupancy'] # XXX scale gas consumption to occupants
        underlying_gas_demand = underlying_gas_demand_reference.copy()
        underlying_gas_demand['Heating'] = int(underlying_gas_demand['Heating'] * scaling_factor)
        underlying_gas_demand['Cooking'] = int(underlying_gas_demand['Cooking'] * scaling_factor)
        underlying_gas_demand['Hotwater'] = int(underlying_gas_demand['Hotwater'] * scaling_factor)
        underlying_gas_demand['Occupancy'] = selected_occupants

        # Heating:
        gas_demand_heating_MJ = 0
        v_electrical_heating_demand_kWh = np.repeat(0, HOURS_PER_YEAR)
        if selected_heating == 'Electric':
            gas_heater_eff = 0.80 # XXX Assumption
            electrical_heater_eff = 1.0 # XXX Assumption
            heating_gas_to_electricity_factor = gas_heater_eff / electrical_heater_eff
            gas_heating_demand_kWh = underlying_gas_demand['Heating'] * CONVERT_MJ_TO_KWH
            electric_heating_demand_kWh = df_xls_heating_ts[selected_location].sum()
            scaling_factor = gas_heating_demand_kWh * heating_gas_to_electricity_factor / electric_heating_demand_kWh * selected_heating_usage_scale
            v_electrical_heating_demand_kWh = np.array(df_xls_heating_ts[selected_location] * scaling_factor)
        elif selected_heating == 'Air-Con':
            gas_heater_eff = 0.80 # XXX Assumption
            electrical_heater_eff = 4.0 # XXX Assumption
            heating_gas_to_electricity_factor = gas_heater_eff / electrical_heater_eff
            gas_heating_demand_kWh = underlying_gas_demand['Heating'] * CONVERT_MJ_TO_KWH
            electric_heating_demand_kWh = df_xls_heating_ts[selected_location].sum()
            scaling_factor = gas_heating_demand_kWh * heating_gas_to_electricity_factor / electric_heating_demand_kWh * selected_heating_usage_scale
            v_electrical_heating_demand_kWh = np.array(df_xls_heating_ts[selected_location] * scaling_factor)
        elif selected_heating == 'Gas':
            gas_demand_heating_MJ = underlying_gas_demand['Heating'] * selected_heating_usage_scale

        # Cooling:
        v_electrical_cooling_demand_kWh = np.repeat(0, HOURS_PER_YEAR)
        if selected_cooling == 'Air-Con':
            # Figure out the amount of cooling required for the number of occupants
            reference_electric_cooling_demand_kWh = df_xls_cooling_ts[selected_location].sum()
            electric_cooling_demand_kWh = selected_occupants / 4 * reference_electric_cooling_demand_kWh # XXX hardcoded reference cooling demand to 4 occupants
            v_electrical_cooling_demand_kWh = np.array(selected_occupants / 4 * df_xls_cooling_ts[selected_location]) * selected_cooling_usage_scale
        elif selected_cooling == 'Evaporative':
            # Figure out the amount of cooling required for the number of occupants
            evaporative_scalingfactor = 0.35 # XXX hard-coded to 35% of air-con
            reference_electric_cooling_demand_kWh = df_xls_cooling_ts[selected_location].sum()
            electric_cooling_demand_kWh = selected_occupants / 4 * evaporative_scalingfactor * reference_electric_cooling_demand_kWh # XXX hardcoded reference cooling demand to 4 occupants
            v_electrical_cooling_demand_kWh = np.array(selected_occupants / 4 * evaporative_scalingfactor * df_xls_cooling_ts[selected_location]) * selected_cooling_usage_scale

        # Hot water:
        gas_demand_hotwater_MJ = 0
        v_electrical_hotwater_demand_kWh = np.repeat(0, HOURS_PER_YEAR)
        if selected_hotwater == 'Heat-Pump':
            hotwater_gas_to_electricity_factor = 0.25 # XXX figure it out properly
            v_electrical_hotwater_demand_kWh = np.repeat(underlying_gas_demand['Hotwater'] * CONVERT_MJ_TO_KWH * hotwater_gas_to_electricity_factor / HOURS_PER_YEAR, HOURS_PER_YEAR)
        elif selected_hotwater == 'Electric':
            hotwater_gas_to_electricity_factor = 0.5 # XXX figure it out properly
            v_electrical_hotwater_demand_kWh = np.repeat(underlying_gas_demand['Hotwater'] * CONVERT_MJ_TO_KWH * hotwater_gas_to_electricity_factor / HOURS_PER_YEAR, HOURS_PER_YEAR)   
        elif selected_hotwater == 'Gas':
            gas_demand_hotwater_MJ = underlying_gas_demand['Hotwater']
            
        # Cooking:
        gas_demand_cooking_MJ = 0
        v_electrical_cooking_demand_kWh = np.repeat(0, HOURS_PER_YEAR)
        if selected_cooking == 'Electric' or selected_cooking == 'Induction':
            cooking_gas_to_electricity_factor = 0.5 # XXX figure it out properly
            v_electrical_cooking_demand_kWh = np.repeat(underlying_gas_demand['Cooking'] * CONVERT_MJ_TO_KWH * cooking_gas_to_electricity_factor / HOURS_PER_YEAR, HOURS_PER_YEAR)
        elif selected_cooking == 'Gas':
            gas_demand_cooking_MJ = underlying_gas_demand['Cooking']
        
        # Calculate the overall electricity demand profile by adding the electrical demand related to heating, cooling and hot water to the original load profile
        df_load_profile = df_load_profile + v_electrical_heating_demand_kWh + v_electrical_cooling_demand_kWh + v_electrical_hotwater_demand_kWh + v_electrical_cooking_demand_kWh
        gas_demand_MJ = gas_demand_heating_MJ + gas_demand_hotwater_MJ + gas_demand_cooking_MJ

        # Process the meter data
        # ----------------------
        # Declare constants
        timestep_seconds = SCENARIO_PARAM['timestep_seconds']
        convert_energy_to_power = int(SECONDS_PER_HOUR / timestep_seconds)
        convert_power_to_energy = 1 / convert_energy_to_power
        num_annual_steps = int(HOURS_PER_YEAR * SECONDS_PER_HOUR / timestep_seconds)
        ZERO_BAND = 1E-3 # Avoiding floating point arithmatic errors

        # Check that the input data is valid (timestamps, load, pv generation, tariffs)
        v_input_annual_load_kwh = np.asarray(df_load_profile.tolist(), dtype = float)
        v_input_annual_unitgeneration_kwh = np.asarray(df_solar_profile.tolist(), dtype = float)
        v_input_annual_timestamp = np.asarray(df_timestamp['Timestamp_begin'].tolist())
        v_input_annual_tariff_import = np.asarray(df_electricity_tariffs['Import'].tolist(), dtype = float)
        v_input_annual_tariff_export = np.asarray(df_electricity_tariffs['Export'].tolist(), dtype = float)
        v_input_annual_tariff_fixed = np.asarray(df_electricity_tariffs['Fixed'].tolist(), dtype = float)

        data_is_valid = len(v_input_annual_load_kwh) == num_annual_steps & \
                        len(v_input_annual_unitgeneration_kwh) == num_annual_steps & \
                        len(v_input_annual_timestamp) == num_annual_steps
    #     if not data_is_valid:
    #         if log_key not in st.session_state:
    #             logMessage("ERROR: Input data is invalid. Scenario analysis halted!", OUTPUT_DIRECTORY)
    #             st.session_state[log_key] = True  # Mark as logged

        if data_is_valid:

            # Initialise the household load and generation profile, along with the applicable tariffs
            financial_horizon = SCENARIO_PARAM['financial_horizon']
            v_overall_netload_kwh = np.tile(v_input_annual_load_kwh, financial_horizon)
            v_overall_pvunitgeneration_kwh = np.tile(v_input_annual_unitgeneration_kwh, financial_horizon)
            v_overall_tariff_export = np.tile(v_input_annual_tariff_export, financial_horizon)
            v_overall_tariff_import = np.tile(v_input_annual_tariff_import, financial_horizon)
            v_overall_tariff_fixed = np.tile(v_input_annual_tariff_fixed, financial_horizon)
            
            pv_capacity = SCENARIO_PARAM['solar_capacity']
            batt_capacity = SCENARIO_PARAM['battery_energy_capacity']
            batt_inverter_capacity = SCENARIO_PARAM['battery_power_capacity']

            df_opt_exogenous_timeseries = pd.DataFrame(index = range(len(v_overall_netload_kwh)), 
                                            columns = [
                                                't',
                                                'E_netload_kWh',
                                                'E_pvunitgeneration_kWh',
                                                'C_tariff_import_$/kWh',
                                                'C_tariff_export_$/kWh',
                                                'C_tariff_fixed_$/day',
                                            ])
            df_opt_exogenous_timeseries['t'] = np.tile(df_timestamp['Datetime_begin'], SCENARIO_PARAM['financial_horizon'])
            df_opt_exogenous_timeseries['E_netload_kWh'] = v_overall_netload_kwh
            df_opt_exogenous_timeseries['E_pvunitgeneration_kWh'] = v_overall_pvunitgeneration_kwh
            df_opt_exogenous_timeseries['C_tariff_import_$/kWh'] = v_overall_tariff_import
            df_opt_exogenous_timeseries['C_tariff_export_$/kWh'] = v_overall_tariff_export
            df_opt_exogenous_timeseries['C_tariff_fixed_$/day'] = v_overall_tariff_fixed

            # Consider energy requirements related to household climate control?
            # ...

            # Consider energy requirements related to travel
            selected_cars = [
                CarConfig(
                    fuel_type=selected_car_types[i],
                    annual_distance_km=selected_car_distances[i],
                    efficiency=selected_car_efficiencies[i],
                    schedule=selected_car_schedules[i],
                    charger_speed=selected_car_charger_speeds[i],
                    charging_strategy=selected_car_charging_strategies[i]
                )
                for i in range(selected_numcars)
            ]
    
            [transport_petrolfuel_energy_MJ, 
             transport_dieselfuel_energy_MJ, 
             transport_fuel_energy_kWh, 
             transport_public_charging_energy_kWh,
             ts_transport_electricity_Wh] = create_transport_electricity_profile(selected_cars, df_opt_exogenous_timeseries, SCENARIO_PARAM) # scenario_param = SCENARIO_PARAM
            df_opt_exogenous_timeseries['E_netload_kWh'] += ts_transport_electricity_Wh / 1000

            # Run the simulation
            i_static_param = SCENARIO_PARAM
            i_df_energy_ts = df_opt_exogenous_timeseries
            [df_results, summary] = simulatePVB(df_opt_exogenous_timeseries, SCENARIO_PARAM, False)            
            
            summary['fuel_imports_kWh'] = transport_fuel_energy_kWh
            summary['diesel_imports_MJ'] = transport_dieselfuel_energy_MJ
            summary['petrol_imports_MJ'] = transport_petrolfuel_energy_MJ
            summary['public_charging_kWh'] = transport_public_charging_energy_kWh
            
            gas_required_GJ = gas_demand_MJ / 1000
            gas_energy_kWh = int(np.round(gas_required_GJ * 1000 * CONVERT_MJ_TO_KWH))
            summary['gas_imports_kWh'] = gas_energy_kWh

            # Store household config for the infographic
            SCENARIO_PARAM['occupants']  = selected_occupants
            SCENARIO_PARAM['num_cars']   = selected_numcars
            SCENARIO_PARAM['car_types']  = tuple(selected_car_types)
            SCENARIO_PARAM['heating']    = selected_heating
            SCENARIO_PARAM['cooling']    = selected_cooling
            SCENARIO_PARAM['selected_heating_usage'] = selected_heating_usage.split(' ')[0]
            SCENARIO_PARAM['selected_cooling_usage'] = selected_cooling_usage.split(' ')[0]
            SCENARIO_PARAM['hotwater']   = selected_hotwater
            SCENARIO_PARAM['cooking']    = selected_cooking

            # Create the interactive Streamlit dashboard of the outputs
            create_streamlit_dashboard(df_results, summary, df_opt_exogenous_timeseries, SCENARIO_PARAM)
    
    #     s of the outputst.session_state[log_key] = True  # Mark as logged
    #     st.sidebar.json(dict(st.session_state))


#%% ===========================================================================

# import plotly.graph_objects as go

# # 1. Setup sample data (Months, 2 Consumption types, 2 Generation types)
# months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']

# # Consumption Data (To be styled with patterns)
# grid_consumption = [50, 55, 60, 45, 40, 35]
# ev_consumption = [30, 25, 20, 15, 10, 10]

# # Self-Generation Data (To be styled with 50% transparency)
# solar_exports = [20, 35, 50, 60, 65, 70]
# solar_selfconsumption = [15, 15, 10, 12, 18, 20]

# # 2. Initialize the Figure
# fig = go.Figure()

# # --- GROUP 1: CONSUMPTION (Stacked using patterns) ---
# fig.add_trace(go.Bar(
#     x=months,
#     y=grid_consumption,
#     name='Grid Consumption',
#     offsetgroup=0,  # Group 0 pushes these bars to the left side of the slot
#     marker=dict(
#         color='#1f77b4',  # Solid blue base
#         pattern=dict(shape='/', solidity=0.3)  # Diagonal lines
#     )
# ))

# fig.add_trace(go.Bar(
#     x=months,
#     y=ev_consumption,
#     name='EV Consumption',
#     offsetgroup=0,  # Keeps it stacked vertically on top of Grid Consumption
#     marker=dict(
#         color='#1f77b4',  # Same base color family
#         pattern=dict(shape='x', solidity=0.3)  # Crosshatch lines
#     )
# ))

# # --- GROUP 2: SELF-GENERATION (Stacked using 50% transparency) ---
# # Note: Colors use 'rgba' where the last number (0.5) sets 50% opacity
# fig.add_trace(go.Bar(
#     x=months,
#     y=solar_exports,
#     name='Solar Exports',
#     offsetgroup=1,  # Group 1 pushes these bars to the right side of the slot
#     marker=dict(
#         color='rgba(46, 204, 113, 0.5)',  # 50% transparent green
#         line=dict(color='rgba(46, 204, 113, 1.0)', width=1.5)  # Solid border for crispness
#     )
# ))

# fig.add_trace(go.Bar(
#     x=months,
#     y=solar_selfconsumption,
#     name='Solar Self-Consumption',
#     offsetgroup=1,  # Keeps it stacked vertically on top of Solar Generation
#     marker=dict(
#         color='rgba(155, 89, 182, 0.5)',  # 50% transparent purple
#         line=dict(color='rgba(155, 89, 182, 1.0)', width=1.5)  # Solid border
#     )
# ))

# # 3. Configure the Layout
# fig.update_layout(
#     title='Energy Consumption vs. Self-Generation Archetypes',
#     xaxis=dict(title='Months'),
#     yaxis=dict(title='Energy (kWh)'),
#     barmode='stack',  # Enables stacking behavior within each offsetgroup
#     legend=dict(
#         x=1.02, y=1,  # Moves legend slightly outside the plot area
#         traceorder='normal'
#     ),
#     # Optional: adjust the spacing between the grouped clusters
#     bargap=0.15,      # Spacing between monthly clusters
#     bargroupgap=0.05  # Spacing between the Consumption and Generation bars
# )

# # 4. Render the chart
# fig.show()


#%% ===========================================================================
# Entry Point
# =============================================================================
if __name__ == '__main__':
    # Configure Streamlit page (must be first Streamlit command)
    st.set_page_config(
        page_title="PV-Battery Energy Analysis",
        page_icon="🔋",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] { min-width: 410px; }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Run the main application
    main()