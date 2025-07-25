import functools
import multiprocessing
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import datetime
import plotly.colors as pc
import re
from plotly.subplots import make_subplots
from scipy.interpolate import griddata
import time
import numpy as np
import colorsys
from concurrent.futures import ThreadPoolExecutor

from calculate_hot_water_delivered import calculate_hot_water_delivered


L_TO_GAL_RATIO = 0.264172

def load_data(results_folder='../OCHRE_output/results/'):
    """Load CSV files from the results folder into a dictionary of DataFrames."""
    csv_files = [f for f in os.listdir(results_folder) if f.endswith('.csv')]
    if len(csv_files) < 2:
        raise ValueError("At least 2 CSV files are required in the 'results' folder for comparison.")
    return {file: pd.read_csv(os.path.join(results_folder, file)) for file in csv_files}

def find_matching_columns(df, patterns):
    """Find columns that match the given patterns and group them."""
    column_groups = {pattern: [] for pattern in patterns}
    
    for col in df.columns:
        for pattern in patterns:
            # Use regex to match pattern followed by a number
            match = re.match(f"{pattern}(\\d+)$", col)
            if match:
                column_groups[pattern].append(col)
    
    # Sort columns within each group by node number
    for pattern in patterns:
        column_groups[pattern].sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
    
    return column_groups

def adjust_lightness(color, factor):
    """
    Adjust the lightness of a color.
    
    Args:
        color: The color to adjust (hex string)
        factor: Factor to adjust lightness by (0-1)
    
    Returns:
        Adjusted color as hex string
    """
    # Convert hex to RGB
    color = color.lstrip('#')
    r, g, b = int(color[0:2], 16) / 255, int(color[2:4], 16) / 255, int(color[4:6], 16) / 255
    
    # Convert RGB to HSL
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    
    # Adjust lightness
    l = max(min(l * factor, 1.0), 0.0)
    
    # Convert back to RGB
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    
    # Convert back to hex
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

# Regex definitions
SETPOINT_RE = re.compile(r"setpoint-(\d+)F", re.IGNORECASE)
TANK_RE     = re.compile(r"(\d+)gal", re.IGNORECASE)
TYPE_RE     = re.compile(r"(Electric|Heat[Pp]ump)")
# SHIFT_RE    = re.compile(r"(cp_h-T_data_shifted_\d+F)", re.IGNORECASE)  # currently unused

def plot_draw_event_summary(draw_outputs, plot_energy=False):
    """
    Plot grouped bars by tank_size & setpoint, with Electric and HeatPump as separate bars.
    Colors encode tank_size, saturation encodes setpoint.
    Includes manufacturer reference value lines for comparison.
    
    Parameters:
    -----------
    draw_outputs : dict
        Dictionary containing the draw output data
    plot_energy : bool, default=False
        If True, plots energy used (kWh) instead of water volume (gal)
    """
    # 1) Parse filenames into metadata
    meta = {}
    for fname, data in draw_outputs.items():
        # Extract type
        ttype_match = TYPE_RE.search(fname)
        ttype = ttype_match.group(1) if ttype_match else 'Unknown'
        if ttype.lower() == "heatpump":
            ttype = "HeatPump"  # Normalize capitalization
            
        # Extract setpoint
        sp_match = SETPOINT_RE.search(fname)
        sp = int(sp_match.group(1)) if sp_match else 0
        
        # Extract tank size
        ts_match = TANK_RE.search(fname)
        ts = int(ts_match.group(1)) if ts_match else 0
        
        if 'pcm' in fname:
            pcm: bool = True
        else:
            pcm: bool = False
        
        meta[fname] = {
            'pcm': pcm,
            'type': ttype,
            'setpoint': sp,
            'tank_size': ts,
            'volume': data['total_water_volume_gal'],
            'energy_delivered': data['total_heat_delivered_kWh'],
            'energy_used': data['total_energy_used_kwh']
        }
    
    # Group by tank size, setpoint, and heater type
    grouped_data = {}
    for fname, info in meta.items():
        key = (info['tank_size'], info['setpoint'])
        if key not in grouped_data:
            grouped_data[key] = {}
        grouped_data[key][f"{info['type']} - {info['pcm']}"] = {
            'volume': info['volume'],
            'energy_delivered': info['energy_delivered'],
            'energy_used': info['energy_used'],
            'pcm': info['pcm']
        }
    
    # Sort keys by tank size then setpoint
    sorted_keys = sorted(grouped_data.keys())
    
    # 3) Build color scales per tank size
    tank_sizes = sorted({ts for ts, _ in sorted_keys})
    base_colors = pc.qualitative.Plotly
    hue_map = {ts: base_colors[i % len(base_colors)] for i, ts in enumerate(tank_sizes)}
    
    # Determine setpoint ranges per tank size
    sp_by_ts = {}
    for ts in tank_sizes:
        sps = sorted({sp for t, sp in sorted_keys if t == ts})
        sp_by_ts[ts] = (min(sps), max(sps))
    
    def color_for(ts, sp):
        base = hue_map.get(ts)
        lo, hi = sp_by_ts.get(ts, (sp, sp))
        sat = 0.3 + 0.7 * ((sp - lo) / (hi - lo) if hi > lo else 1)
        return adjust_lightness(base, 1 - sat)
    
        
    def color_for_mixed(ts, sp):
    # Neon base colors - shifted towards brighter, more vibrant hues
        base = hue_map.get(ts)
        lo, hi = sp_by_ts.get(ts, (sp, sp))
        
        # Increased saturation baseline to make colors more vivid
        sat = 0.3 + 0.7 * ((sp - lo) / (hi - lo) if hi > lo else 1)
        
        # Boost lightness to make colors appear more neon-like
        return adjust_lightness(base, 1.2 - 0.6 * sat)
    
    # Create figure
    fig = go.Figure()
    
    # Create x-axis labels
    x_labels = [f"{ts} gal\n{sp}°F" for ts, sp in sorted_keys]
    
    # Prepare data for plotting
    electric_volume = []
    electric_energy = []
    heatpump_volume = []
    heatpump_energy = []
    electric_pcm_volume = []
    electric_pcm_energy = []
    heatpump_pcm_volume = []
    heatpump_pcm_energy = []
    colors = [color_for(ts, sp) for ts, sp in sorted_keys]
    colors_mixed = [color_for_mixed(ts, sp) for ts, sp in sorted_keys]
    
    # Extract values for each heater type
    for key in sorted_keys:
        data = grouped_data[key]
        
        # Electric values
        if "Electric - False" in data:
            electric_volume.append(data["Electric - False"]["volume"])
            electric_energy.append(data["Electric - False"]["energy_delivered"])
        else:
            electric_volume.append(0)
            electric_energy.append(0)
            
        # HeatPump values
        if "HeatPump - False" in data:
            heatpump_volume.append(data["HeatPump - False"]["volume"])
            heatpump_energy.append(data["HeatPump - False"]["energy_delivered"])
        else:
            heatpump_volume.append(0)
            heatpump_energy.append(0)
            
        if "Electric - True" in data:
            electric_pcm_volume.append(data["Electric - True"]["volume"])
            electric_pcm_energy.append(data["Electric - True"]["energy_delivered"])
        else:
            electric_pcm_volume.append(0)
            electric_pcm_energy.append(0)
            
        if "HeatPump - True" in data:
            heatpump_pcm_volume.append(data["HeatPump - True"]["volume"])
            heatpump_pcm_energy.append(data["HeatPump - True"]["energy_delivered"])  
        else:
            heatpump_pcm_volume.append(0)
            heatpump_pcm_energy.append(0)
    
    # Determine which data to plot based on the plot_energy flag
    if plot_energy:
        # Energy data
        electric_data = electric_energy
        heatpump_data = heatpump_energy
        electric_pcm_data = electric_pcm_energy
        heatpump_pcm_data = heatpump_pcm_energy
        y_axis_title = 'Energy Delivered (kWh)'
        value_prefix = 'Energy'
        formatting = lambda x: f"{x:.3f}"
    else:
        # Volume data (default)
        electric_data = electric_volume
        heatpump_data = heatpump_volume
        electric_pcm_data = electric_pcm_volume
        heatpump_pcm_data = heatpump_pcm_volume
        y_axis_title = 'Delivered Water Volume (gal)'
        value_prefix = 'Water'
        formatting = lambda x: f"{x:.2f}"
    
    # Add Electric data bars (solid fill)
    fig.add_trace(go.Bar(
        name=f'Electric - {value_prefix} Only',
        x=x_labels,
        y=electric_data,
        marker=dict(color=colors, pattern=dict(shape='')),
        text=[formatting(val) for val in electric_data],
        textposition='auto',
        offsetgroup=0
    ))
    
    # Add HeatPump data bars (hatched)
    fig.add_trace(go.Bar(
        name=f'HeatPump - {value_prefix} Only',
        x=x_labels,
        y=heatpump_data,
        marker=dict(color=colors, pattern=dict(shape='/')),
        text=[formatting(val) for val in heatpump_data],
        textposition='auto',
        offsetgroup=1
    ))
    
    # Add Electric PCM data bars
    fig.add_trace(go.Bar(
        name=f'Electric - Internal PCM',
        x=x_labels,
        y=electric_pcm_data,
        marker=dict(color='rgba(0,0,0,0)', line=dict(color='black', width=1)),
        text=[formatting(val) for val in electric_pcm_data],
        textposition='auto',
        offsetgroup=2
    ))

    # Add HeatPump PCM data bars
    fig.add_trace(go.Bar(
        name=f'Heatpump - Internal PCM',
        x=x_labels,
        y=heatpump_pcm_data,
        marker=dict(color='rgba(0,0,0,0)', line=dict(color='black', width=1), pattern=dict(shape='/')),
        text=[formatting(val) for val in heatpump_pcm_data],
        textposition='auto',
        offsetgroup=3
    ))
    
    # Update layout to display bars side by side
    # fig.update_layout(
    #     title=f'125 °F mixing valve 110 °F Cut off Temp FHR Summary by Tank Size & Setpoint ({y_axis_title})', 
    #     xaxis_title='Tank Size & Setpoint',
    #     yaxis_title=y_axis_title,
    #     barmode='group',  # This ensures the bars are displayed side by side
    #     legend=dict(
    #         orientation="h",
    #         yanchor="bottom",
    #         y=1.02,
    #         xanchor="center",
    #         x=0.5
    #     )
    # )
    
    fig.update_layout(
        title=f'No Mixing valve 110 °F Cut off Temp FHR Summary by Tank Size & Setpoint ({y_axis_title})', 
        xaxis_title='Tank Size & Setpoint',
        yaxis_title=y_axis_title,
        barmode='group',  # This ensures the bars are displayed side by side
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    fig.show() 
    
    
    
def plot_draw_events(draw_outputs):
    """
    For each file, create grouped bar charts of draw events.
    zDefault files get black bars; all others use the default color cycle.
    """
    # Find max number of events and labels
    max_events = max(len(m['draw_events']) for m in draw_outputs.values())
    event_numbers = [f"Event {i+1}" for i in range(max_events)]

    # Prepare water and heat data
    water_data = {}
    heat_data = {}
    for fname, metrics in draw_outputs.items():
        vols, heats = [], []
        for i in range(max_events):
            if i < len(metrics['draw_events']):
                e = metrics['draw_events'][i]
                vols.append(round(e['water_volume_gal'], 2))
                heats.append(round(e['heat_delivered_kWh'], 3))
            else:
                vols.append(None)
                heats.append(None)
        water_data[fname] = vols
        heat_data[fname] = heats

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Water Volume (gal)", "Heat Delivered (kWh)"),
        shared_xaxes=True
    )

    # Add traces, only zDefault gets marker_color='black'
    for row, data_dict, fmt in [
        (1, water_data, "{:.2f}"),
        (2, heat_data, "{:.3f}")
    ]:
        for fname, vals in data_dict.items():
            params = dict(
                name=fname,
                x=event_numbers,
                y=vals,
                text=[fmt.format(v) if v is not None else "" for v in vals],
                textposition='auto'
            )
            if "zDefault" in fname:
                params["marker_color"] = "black"
            fig.add_trace(go.Bar(**params), row=row, col=1)

    fig.update_layout(
        barmode='group',
        title_text="Draw Events Grouped by Event Number Across Files",
        xaxis_title="Draw Event"
    )
    fig.show()


def plot_totals(draw_outputs):
    """
    For each file, plot:
      • Total Water Volume (gal)
      • Total Heat Delivered (kWh)
    zDefault files get black bars; all others use the default color cycle.
    """
    # extract filenames and metrics
    files = list(draw_outputs.keys())
    water_totals = [round(m['total_water_volume_gal'], 2) for m in draw_outputs.values()]
    heat_totals  = [round(m['total_heat_delivered_kWh'], 3) for m in draw_outputs.values()]

    # set up two-row subplot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Total Water Volume (gal)", "Total Heat Delivered (kWh)"),
        shared_xaxes=True
    )

    # row 1: water
    for fname, val in zip(files, water_totals):
        params = dict(
            name=fname,
            x=[fname],
            y=[val],
            text=[f"{val:.2f}"],
            textposition='auto'
        )
        if "zDefault" in fname:
            params["marker_color"] = "black"
        fig.add_trace(go.Bar(**params), row=1, col=1)

    # row 2: heat
    for fname, val in zip(files, heat_totals):
        params = dict(
            name=fname,
            x=[fname],
            y=[val],
            text=[f"{val:.3f}"],
            textposition='auto'
        )
        if "zDefault" in fname:
            params["marker_color"] = "black"
        fig.add_trace(go.Bar(**params), row=2, col=1)

    # layout tweaks
    fig.update_layout(
        barmode='group',
        title_text="Total Metrics by File",
    )
    fig.update_xaxes(showticklabels=False)
    fig.show()



def plot_comparison(dfs, draw_outputs):
    """
    Create two scatter plots comparing average h_value (W/m^2K) and average sa_ratio 
    for each file. The first plot colors the points by the total hot water energy delivered 
    (computed from draw_events), and the second plot by energy used (assumed to be stored in draw_outputs).
    
    Parameters:
    - dfs: dict
        Dictionary of dataframes keyed by file name. Each dataframe has columns with names like:
        "Water Tank PCM<number> h (W/m^2K)" and "Water Tank PCM<number> sa_ratio".
    - draw_outputs: dict
        Dictionary containing hot water energy metrics for each file. It is assumed that:
          - draw_outputs[file]['draw_events'] is a list of events, each with key 'heat_delivered_kWh'.
          - draw_outputs[file]['energy_used_kWh'] exists for the energy used metric.
    """
    data = []
    
    for file, df in dfs.items():
        # Extract values from columns matching the h_value and sa_ratio patterns
        h_values = []
        sa_ratios = []
        
        for col in df.columns:
            # Regex for h_value
            h_match = re.search(r"Water Tank PCM(\d+)\s*h\s*\(W/m\^?2K\)", col)
            if h_match:
                h_values.append(df[col].mean())
            
            # Regex for sa_ratio
            sa_match = re.search(r"Water Tank PCM(\d+)\s*sa_ratio", col)
            if sa_match:
                sa_ratios.append(df[col].mean())
        
        # Compute average values if we found at least one value from each group
        if h_values and sa_ratios:
            avg_h = sum(h_values) / len(h_values)
            avg_sa = sum(sa_ratios) / len(sa_ratios)
        else:
            avg_h = 0
            avg_sa = 0

        # Get energy metrics from draw_outputs
        total_gal_hot_water_delivered = draw_outputs[file].get('total_water_volume_gal', None)
        total_delivered = draw_outputs[file].get('total_heat_delivered_kWh', None)  # Fixed key name
        total_used = draw_outputs[file].get('total_energy_used_kwh', None)
        
        data.append({
            'file': file,
            'avg_h_value': avg_h,
            'avg_sa_ratio': avg_sa,
            'total_gal_hot_water_delivered': total_gal_hot_water_delivered,
            'total_heat_delivered_kWh': total_delivered,  # Added this field which was missing
            'total_energy_used': total_used
        })

    # Create a DataFrame from the collected data
    df_plot = pd.DataFrame(data)

    # Drop any rows with missing values for plotting
    df_plot_clean = df_plot.dropna(subset=['avg_h_value', 'avg_sa_ratio'])

    # Create interpolation grid
    # Create a grid of points to interpolate over
    grid_resolution = 100
    x_min, x_max = df_plot_clean['avg_sa_ratio'].min(), df_plot_clean['avg_sa_ratio'].max()
    y_min, y_max = df_plot_clean['avg_h_value'].min(), df_plot_clean['avg_h_value'].max()

    # Add a small buffer to avoid edge issues
    x_buffer = (x_max - x_min) * 0.05
    y_buffer = (y_max - y_min) * 0.05

    x_grid = np.linspace(x_min - x_buffer, x_max + x_buffer, grid_resolution)
    y_grid = np.linspace(y_min - y_buffer, y_max + y_buffer, grid_resolution)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

    # Function to create interpolated plot
    def create_interpolated_plot(df, z_column, title, z_label):
        if df.empty or df[z_column].isna().all():
            print(f"No valid data for {z_column}")
            return None
        
        # Points for interpolation
        points = df[['avg_sa_ratio', 'avg_h_value']].values
        values = df[z_column].values
        
        # Perform interpolation - using 'cubic' for smoother results
        grid_z = griddata(points, values, (x_mesh, y_mesh), method='linear')
        
        # Get min and max values for color scale normalization
        z_min = np.nanmin(values)
        z_max = np.nanmax(values)
        
        # Create a custom color scale with more gradations
        # You can add as many color points as needed for finer gradations
        custom_colorscale = [
            [0.0, 'rgb(68, 1, 84)'],       # Dark purple
            [0.1, 'rgb(72, 40, 120)'],     # Purple
            [0.2, 'rgb(62, 74, 137)'],     # Blue-purple
            [0.3, 'rgb(49, 104, 142)'],    # Dark blue
            [0.4, 'rgb(38, 130, 142)'],    # Teal
            [0.5, 'rgb(31, 158, 137)'],    # Turquoise
            [0.6, 'rgb(53, 183, 121)'],    # Green
            [0.7, 'rgb(109, 205, 89)'],    # Light green
            [0.8, 'rgb(180, 222, 44)'],    # Yellow-green
            [0.9, 'rgb(223, 205, 35)'],    # Yellow
            [1.0, 'rgb(253, 231, 37)']     # Bright yellow
        ]
        
        # Create the figure
        fig = go.Figure()
        
        # Add contour plot with more levels for finer gradations
        contour = go.Contour(
            z=grid_z,
            x=x_grid,
            y=y_grid,
            colorscale=custom_colorscale,
            colorbar=dict(
                title=z_label,
                ticks="outside",
                tickfont=dict(size=12),
                len=0.75
            ),
            # Increase number of contour levels for more gradations
            ncontours=20,
            contours=dict(
                showlabels=True,
                labelfont=dict(size=12, color='white')
            ),
            # Smooth the contours
            line=dict(width=0.5, smoothing=0.85)
        )
        fig.add_trace(contour)
        
        # Add scatter points for actual data
        scatter = go.Scatter(
            x=df['avg_sa_ratio'],
            y=df['avg_h_value'],
            mode='markers',
            marker=dict(
                size=10,
                color=df[z_column],
                colorscale=custom_colorscale,
                cmin=z_min,
                cmax=z_max,
                line=dict(width=1, color='black'),
                showscale=False
            ),
            text=[f"File: {file}<br>{z_label}: {val:.2f}" for file, val in zip(df['file'], df[z_column])],
            hoverinfo='text'
        )
        fig.add_trace(scatter)
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="SA Ratio",
            yaxis_title="h value (W/m²K)",
            height=600,
            width=800,
            # Add a color axis for more control
            coloraxis=dict(
                colorscale=custom_colorscale,
                colorbar=dict(title=z_label)
            )
        )
        
        return fig

    # Create plot for delivered energy
    fig_delivered = create_interpolated_plot(
        df_plot_clean.dropna(subset=['total_heat_delivered_kWh']),
        'total_heat_delivered_kWh',
        "h_value vs SA_ratio with Interpolated Hot Water Delivered Energy",
        "Total Hot Water Delivered (kWh)"
    )

    # Create plot for energy used
    fig_used = create_interpolated_plot(
        df_plot_clean.dropna(subset=['total_energy_used']),
        'total_energy_used',
        "h_value vs SA_ratio with Interpolated Energy Used",
        "Total Energy Used (kWh)"
    )
    
    fig_total_water = create_interpolated_plot(
        df_plot_clean.dropna(subset=['total_gal_hot_water_delivered']),
        'total_gal_hot_water_delivered',
        "h_value vs SA_ratio with Interpolated Total Water Delivered",
        "Total Water Delivered (gal)"
    )

    # Show the plots
    if fig_delivered:
        fig_delivered.show()
    if fig_used:
        fig_used.show()
    if fig_total_water:
        fig_total_water.show()

def c_to_f(c):
    return c * 9/5 + 32

def process_single_df_hot_water_delivered(df, first_hour_test=False):
    """
    Calculate the total hot water delivered (W) and gallons for each file,
    capturing individual draw events with their time ranges and metrics.
    
    A draw event is defined as any continuous period where water is being drawn (water_draw > 0).
    However, water volume and heat energy are only added when the outlet temperature is >= cutoff.
    
    If first_hour_test=True, samples all outlet temps during each draw and applies the
    “final draw” adjustment per the first-hour test procedure.
    """
    water_draw_col       = "Hot Water Delivered (L/min)"
    water_output_W_col   = "Hot Water Delivered (W)"
    water_outlet_temp    = "Hot Water Outlet Temperature (C)"
    energy_used_col      = "Water Heating Delivered (W)"
    
    water_temp_cutoff    =  43.333  # 110°F in °C
    L_TO_GAL_RATIO       = 0.264172    # liters → gallons
    
    output = {}
    
    for file_key, df in dfs.items():
        try:
            df_copy = df.copy()
            
            # --- prepare ---
            # use fixed max for first‐hour test consistency (as in process_single_df)
            max_water_volume_L   = 180  
            total_water_volume_L = 0
            total_heat_delivered_J = 0
            is_draw_active       = False
            
            draw_events = []
            current_event = None
            # start as false
            is_pcm = False
            
            # compute time deltas (in seconds)
            df_copy.index = pd.to_datetime(df_copy['Time'])
            df_copy['time_delta'] = (
                                    pd.to_datetime(df_copy.index).to_series()
                                    .diff()
                                    .dt.total_seconds()
                                    .fillna(0)
                                    )
            
            pcm_columns = [col for col in df_copy.columns if col.startswith('T_PCM')]
            if len(pcm_columns) > 0:
                is_pcm = True
                pcm_enthalpy_column = 'Total PCM Enthalpy (J)'
                starting_pcm_enthalpy = df_copy[pcm_enthalpy_column].iloc[0]
                
                # find the first instance of pcm temp lower than cutoff_temp
                df_copy['average_pcm_temp'] = df_copy[pcm_columns].mean(axis=1)
                df_copy['is_cutoff_temp'] = df_copy['average_pcm_temp'] < water_temp_cutoff
                try:
                    cutoff_index = df_copy[df_copy['is_cutoff_temp']].index[0]
                    baseline_enthalpy = df_copy[pcm_enthalpy_column][cutoff_index]
                except Exception as e:
                    print(f"Error finding cutoff index for file [{file_key}] minimum pcm temp {df_copy['average_pcm_temp'].min():.2f} C temperature exceeds cutoff {water_temp_cutoff} C")
                    m = re.search(r'setpoint-[^_]+_(.*?)_\d+gal', file_key)
                    LUT = "bin/" + m.group(1) + '.csv'
                    enthalpy_lut = np.loadtxt(LUT, delimiter=",", skiprows=1)
                    temps = enthalpy_lut[:,0]
                    enths = enthalpy_lut[:,2]

                    # interpolate at cutoff temperature
                    h_cutoff = np.interp(water_temp_cutoff, temps, enths)

                    # for logging, find bracketing temps
                    pos = np.searchsorted(temps, water_temp_cutoff)
                    if 0 < pos < len(temps):
                        t0, t1 = temps[pos-1], temps[pos]
                        print(f"Interpolated enthalpy from {LUT} between {t0:.2f}°C and {t1:.2f}°C for cutoff {water_temp_cutoff}°C")
                    else:
                        print(f"Using endpoint enthalpy from {LUT} LUT at {temps[pos if pos<len(temps) else -1]:.2f}°C")

                    # scale by PCM mass [kg] → J
                    baseline_enthalpy = h_cutoff * df_copy['PCM Mass (kg)'].iloc[-1] * 1000
                    df_copy['average_pcm_temp'] = df_copy[pcm_columns].mean(axis=1)
             
                
            
        
            # --- iterate rows ---
            for timestamp, row in df_copy.iterrows():
                flow = row[water_draw_col]
                temp = row[water_outlet_temp]
                heat_W = row[water_output_W_col]
                dt = row['time_delta']
                if is_pcm:
                    avg_end_pcm_temp = row['average_pcm_temp']
                    enthalpy = row['Total PCM Enthalpy (J)']
                
                if flow > 0:
                    if not is_draw_active:
                        is_draw_active = True
                        current_event = {
                            'start_time': timestamp,
                            'end_time': None,
                            'water_volume_L': 0,
                            'heat_delivered_J': 0,
                            'max_temp': temp,
                            'min_temp': temp,
                            'max_flow_rate': flow,
                            'temp_readings': [] if first_hour_test else None,
                        }
                    # update envelope
                    current_event['end_time'] = timestamp
                    current_event['max_flow_rate'] = max(current_event['max_flow_rate'], flow)
                    current_event['max_temp']      = max(current_event['max_temp'], temp)
                    current_event['min_temp']      = min(current_event['min_temp'], temp)
                    
                    if first_hour_test:
                        current_event['temp_readings'].append(temp)
                    
                    # only count “hot” volume & energy
                    if temp >= water_temp_cutoff:
                        # note: W * s → J, and (L/min)*(s)→L
                        water_L = 3 / L_TO_GAL_RATIO / (60/dt)
                        energy_J = heat_W / (60/dt)
                        total_water_volume_L += water_L
                        total_heat_delivered_J += energy_J
                        current_event['water_volume_L']  += water_L
                        current_event['heat_delivered_J'] += energy_J
                
                elif is_draw_active:
                    # draw ended
                    if current_event and current_event['water_volume_L'] > 0:
                        # finalize event
                        current_event['water_volume_gal']    = current_event['water_volume_L'] * L_TO_GAL_RATIO
                        current_event['heat_delivered_kWh']  = current_event['heat_delivered_J'] * 2.77778e-7
                        if first_hour_test and current_event['temp_readings']:
                            current_event['avg_temp'] = (
                                sum(current_event['temp_readings']) / len(current_event['temp_readings'])
                            )
                        if is_pcm:
                            current_event['pcm_enthalpy'] = enthalpy
                            current_event['avg_pcm_temp'] = avg_end_pcm_temp
                            current_event['pcm_soc'] = (enthalpy - baseline_enthalpy) / (baseline_enthalpy)
                        draw_events.append(current_event)
                    is_draw_active = False
                    current_event = None
            
            # close last event if still open
            if is_draw_active and current_event and current_event['water_volume_L'] > 0:
                current_event['water_volume_gal']    = current_event['water_volume_L'] * L_TO_GAL_RATIO
                current_event['heat_delivered_kWh']  = current_event['heat_delivered_J'] * 2.77778e-7
                if first_hour_test and current_event['temp_readings']:
                    current_event['avg_temp'] = (
                        sum(current_event['temp_readings']) / len(current_event['temp_readings'])
                    )
                    if is_pcm: 
                        current_event['pcm_enthalpy'] = enthalpy
                        current_event['avg_pcm_temp'] = avg_end_pcm_temp
                        current_event['pcm_soc'] = (enthalpy - baseline_enthalpy) / (baseline_enthalpy)
                draw_events.append(current_event)
            
            # aggregate totals
            total_water_volume_gal    = total_water_volume_L * L_TO_GAL_RATIO
            total_heat_delivered_kWh  = total_heat_delivered_J * 2.77778e-7
            total_energy_used_kwh     = df_copy[energy_used_col].sum() / 60 / 1000
            
            # --- first‐hour adjustment ---
            if first_hour_test and len(draw_events) >= 2:
                final = draw_events[-1]
                prev  = draw_events[-2]
                
                # duration in seconds
                dur = (
                    (final['end_time'] - final['start_time']).total_seconds()
                    if hasattr(final['end_time'], 'to_pydatetime')
                    else final['end_time'] - final['start_time']
                )
                if dur >= 30 and final['max_temp'] >= water_temp_cutoff:
                    # compute adjustment factor
                    denom = (prev.get('avg_temp', prev['min_temp']) - prev['min_temp']) or 1
                    adj = ((final.get('avg_temp', final['min_temp']) - prev['min_temp']) / denom)
                    # apply
                    adjusted_gal = (total_water_volume_gal - final['water_volume_gal']) + \
                                    (final['water_volume_gal'] * adj)
                    total_water_volume_gal = adjusted_gal
                    total_water_volume_L   = total_water_volume_gal / L_TO_GAL_RATIO
                    
                    
            # grab the temperature of each pcm layer at the end of the list
            try:
                pcm_temps = [df[col].iloc[-1] for col in pcm_columns]
                if len(pcm_temps) > 0:
                    average_pcm_end_temp = sum(pcm_temps) / len(pcm_temps)
                    pcm_soc = (df_copy[pcm_enthalpy_column].iloc[-1] - baseline_enthalpy) / (starting_pcm_enthalpy - baseline_enthalpy)
                else:
                    average_pcm_end_temp = 14.44
                    pcm_soc = -100
            except Exception as e:
                print(f"Error calculating average PCM temperature for file [{file_key}]: {e}")
                average_pcm_end_temp = 14.44
                pcm_soc = -100

            
            output[file_key] = {
                "average_pcm_end_temp": average_pcm_end_temp,
                "pcm_soc": pcm_soc,
                'total_water_volume_L': total_water_volume_L,
                'total_water_volume_gal': total_water_volume_gal,
                'total_energy_used_kwh': total_energy_used_kwh,
                'total_heat_delivered_J': total_heat_delivered_J,
                'total_heat_delivered_kWh': total_heat_delivered_kWh,
                'max_possible_hot_water': max_water_volume_L * L_TO_GAL_RATIO,
                'draw_events': draw_events,
                'num_draw_events': len(draw_events)
            }
        except Exception as e:
            print(f"Error processing file [{file_key}]: {e}")
    
    return output



def parallel_calculate_hot_water_delivered(dfs, first_hour_test=False, num_processes=None):
    """
    Runs calculate_hot_water_delivered on each (file_key, df) *as a* one-item dict,
    in parallel, and then merges all of the per-file outputs into one dict.
    """
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    # build a list of (one_file_dict, first_hour_test) tuples
    tasks = [
        ({file_key: df}, first_hour_test)
        for file_key, df in dfs.items()
    ]

    with multiprocessing.Pool(processes=min(num_processes, len(tasks))) as pool:
        # each call returns a dict of shape {file_key: result_for_that_file}
        per_file_dicts = pool.starmap(calculate_hot_water_delivered, tasks)

    # merge all of the single-entry dicts into one
    output = {}
    for d in per_file_dicts:
        output.update(d)

    return output


# Predefined lookup arrays for water properties at 1 atm.
_TEMPS = np.array([0, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
_DENSITIES = np.array([999.8, 1000.0, 999.7, 998.2, 995.7, 992.2, 988.1, 983.2, 977.8, 971.8, 965.3, 958.4], dtype=float)
_THERMALEXPANSIONS = np.array([-1.0e-4, 0.0, 1.1e-4, 2.1e-4, 2.6e-4, 3.1e-4, 3.6e-4, 4.1e-4, 4.7e-4, 5.3e-4, 5.9e-4, 6.6e-4], dtype=float)

def lookup_water_properties(T):
    """
    Returns interpolated water properties for a given temperature T (in °C).
    The returned properties are based on approximate data for pure water at 1 atm.
    
    Properties returned:
      - density: in kg/m³
      - specific_weight: in N/m³ (calculated as density * 9.81)
      - thermal_expansion: volumetric thermal expansion coefficient in 1/°C
    
    Parameters:
      T (float): Temperature in °C (must be within 0 to 100)
      
    Raises:
      ValueError: If the temperature is outside the 0 to 100°C range.
    """
    if T < _TEMPS[0] or T > _TEMPS[-1]:
        raise ValueError("Temperature out of range (must be between 0 and 100 °C)")
    
    density = np.interp(T, _TEMPS, _DENSITIES)
    thermal_expansion = np.interp(T, _TEMPS, _THERMALEXPANSIONS)
    specific_weight = density * 9.81
    
    return {
        "temperature": T,
        "density": density,
        "specific_weight": specific_weight,
        "thermal_expansion": thermal_expansion,
    }

def calculate_net_water_temp(df):
    water_temp_column = ["Hot Water Average Temperature (C)"]
    valid_columns = [col for col in water_temp_column if col in df.columns]

    if not valid_columns:
        return 0

    # Compute the average of the first and last row for all available columns
    average_start_temp = df[valid_columns].iloc[0].mean()
    average_end_temp = df[valid_columns].iloc[-1].mean()

    return average_end_temp - average_start_temp

COLOR_PALETTE = [
    "#377eb8",  # 1 - Blue
    "#e41a1c",  # 2 - Red
    "#4daf4a",  # 3 - Green
    "#984ea3",  # 4 - Purple
    "#ff7f00",  # 5 - Orange
    "#a65628",  # 6 - Brown
    "#f781bf",  # 7 - Pink
    "#999999",  # 8 - Grey
    "#dede00",  # 9 - Yellow
    "#17becf",  # 10 - Cyan
    "#bcbd22",  # 11 - Olive
    "#1f78b4",  # 12 - Deep Blue
]

def get_column_index(col):
    """Extract index from column name like 'T_WH_3' or 'T_PCM_7'."""
    match = re.search(r'(\d+)$', col)
    if match:
        return int(match.group(1)) - 1  # Make it 0-based
    return 0  # Fallback if no number is found

def create_temperature_plots(dfs, uef_values, patterns=['T_WH', 'T_PCM']):
    """Create temperature plots with static color palette for trace indices."""
    all_figs = []
    figure_metadata = []
    water_temp_cutoff = 43.3333  # 110 F

    for i, (file, df) in enumerate(dfs.items()):
        uef = uef_values[i]
        column_groups = find_matching_columns(df, patterns)

        # Extract additional parameters for title
        pcm_mass_col = "PCM Mass (kg)"
        pcm_h_col_pattern = re.compile(r"Water Tank PCM\d+ h \(W/m\^2K\)")
        pcm_sa_col_pattern = re.compile(r"Water Tank PCM\d+ sa_ratio") 
        if pcm_mass_col not in df.columns:
            pcm_mass = 0.0
        else:
            pcm_mass = df[pcm_mass_col].iloc[-1]
        matched_h_cols = next((col for col in df.columns if pcm_h_col_pattern.fullmatch(col)), None) 
        match_sa_col = next((col for col in df.columns if pcm_sa_col_pattern.fullmatch(col)), None) 
        pcm_h = df[matched_h_cols].iloc[-1] if matched_h_cols is not None else 0.0
        pcm_sa = df[match_sa_col].iloc[-1] if match_sa_col is not None else 0.0

        water_volume_col = "Water Volume (L)"
        L_TO_GAL_RATIO = 0.264172
        if water_volume_col not in df.columns:
            water_volume_gal = 50 * .9
        else:
            water_volume_gal = df[water_volume_col].iloc[-1] * L_TO_GAL_RATIO

        # Create a plot for each pattern
        for pattern, columns in column_groups.items():
            if not columns:
                continue
            fig = go.Figure()

            # Determine the temperature range
            temp_min = float('inf')
            temp_max = float('-inf')
            for col in columns:
                temp_min = min(temp_min, df[col].min())
                temp_max = max(temp_max, df[col].max())
            temp_range = temp_max - temp_min
            temp_padding = temp_range * 0.1

            # Add temperature traces with static color assignment
            for col in columns:
                col_idx = get_column_index(col)
                color = COLOR_PALETTE[col_idx % len(COLOR_PALETTE)]
                fig.add_trace(
                    go.Scatter(
                        x=df['Time'],
                        y=df[col],
                        mode='lines',
                        name=col,
                        line=dict(color=color)
                    )
                )

            # [Rest of your overlay/annotation code here, unchanged]
            if 'Water Heating Mode' in df.columns:
                upper_regions = []
                lower_regions = []
                heat_pump_regions = []
                mode_data = df['Water Heating Mode']
                time_data = df['Time']

                in_upper_segment = False
                for j in range(len(mode_data)):
                    mode_str = str(mode_data.iloc[j])
                    if 'Upper On' in mode_str and not in_upper_segment:
                        upper_segment_start = j
                        in_upper_segment = True
                    elif 'Upper On' not in mode_str and in_upper_segment:
                        upper_segment_end = j
                        in_upper_segment = False
                        start_time = time_data.iloc[upper_segment_start]
                        end_time = time_data.iloc[upper_segment_end]
                        upper_regions.append((start_time, end_time))
                if in_upper_segment:
                    upper_segment_end = len(mode_data) - 1
                    start_time = time_data.iloc[upper_segment_start]
                    end_time = time_data.iloc[upper_segment_end]
                    upper_regions.append((start_time, end_time))
                in_lower_segment = False
                for j in range(len(mode_data)):
                    mode_str = str(mode_data.iloc[j])
                    if 'Lower On' in mode_str and not in_lower_segment:
                        lower_segment_start = j
                        in_lower_segment = True
                    elif 'Lower On' not in mode_str and in_lower_segment:
                        lower_segment_end = j
                        in_lower_segment = False
                        start_time = time_data.iloc[lower_segment_start]
                        end_time = time_data.iloc[lower_segment_end]
                        lower_regions.append((start_time, end_time))
                if in_lower_segment:
                    lower_segment_end = len(mode_data) - 1
                    start_time = time_data.iloc[lower_segment_start]
                    end_time = time_data.iloc[lower_segment_end]
                    lower_regions.append((start_time, end_time))
                in_heat_pump_segment = False
                for j in range(len(mode_data)):
                    mode_str = str(mode_data.iloc[j])
                    if 'Heat Pump On' in mode_str and not in_heat_pump_segment:
                        heat_pump_segment_start = j
                        in_heat_pump_segment = True
                    elif 'Heat Pump On' not in mode_str and in_heat_pump_segment:
                        heat_pump_segment_end = j
                        in_heat_pump_segment = False
                        start_time = time_data.iloc[heat_pump_segment_start]
                        end_time = time_data.iloc[heat_pump_segment_end]
                        heat_pump_regions.append((start_time, end_time))
                if in_heat_pump_segment:
                    heat_pump_segment_end = len(mode_data) - 1
                    start_time = time_data.iloc[heat_pump_segment_start]
                    end_time = time_data.iloc[heat_pump_segment_end]
                    heat_pump_regions.append((start_time, end_time))

                # Add overlays
                for j, (start_time, end_time) in enumerate(upper_regions):
                    fig.add_trace(
                        go.Scatter(
                            x=[start_time, end_time, end_time, start_time, start_time],
                            y=[temp_min - temp_padding, temp_min - temp_padding,
                               temp_max + temp_padding, temp_max + temp_padding,
                               temp_min - temp_padding],
                            fill="toself",
                            fillcolor="rgba(255, 0, 0, 0.15)",
                            line=dict(width=0),
                            mode="none",
                            name="Upper Element On" if j == 0 else "",
                            showlegend=True if j == 0 else False,
                            legendgroup="upper_elements",
                            hoverinfo="skip"
                        )
                    )
                for j, (start_time, end_time) in enumerate(lower_regions):
                    fig.add_trace(
                        go.Scatter(
                            x=[start_time, end_time, end_time, start_time, start_time],
                            y=[temp_min - temp_padding, temp_min - temp_padding,
                               temp_max + temp_padding, temp_max + temp_padding,
                               temp_min - temp_padding],
                            fill="toself",
                            fillcolor="rgba(0, 0, 255, 0.15)",
                            line=dict(width=0),
                            mode="none",
                            name="Lower Element On" if j == 0 else "",
                            showlegend=True if j == 0 else False,
                            legendgroup="lower_elements",
                            hoverinfo="skip"
                        )
                    )
                for j, (start_time, end_time) in enumerate(heat_pump_regions):
                    fig.add_trace(
                        go.Scatter(
                            x=[start_time, end_time, end_time, start_time, start_time],
                            y=[temp_min - temp_padding, temp_min - temp_padding,
                               temp_max + temp_padding, temp_max + temp_padding,
                               temp_min - temp_padding],
                            fill="toself",
                            fillcolor="rgba(0, 255, 0, 0.15)",
                            line=dict(width=0),
                            mode="none",
                            name="Heat Pump On" if j == 0 else "",
                            showlegend=True if j == 0 else False,
                            legendgroup="heat_pump",
                            hoverinfo="skip"
                        )
                    )

            fig.update_layout(
                title=f'{pattern} Temperatures - {file}<br>'
                      f'UEF: {uef:.3f} | PCM h: {pcm_h:.2f} W/m^2K | PCM SA Ratio: {pcm_sa:.2f} | PCM Mass: {pcm_mass:.3f} kg | '
                      f'Water Volume: {water_volume_gal:.1f} gal',
                xaxis_title='Time',
                yaxis_title='Temperature (°C)',
                height=600,
                showlegend=True
            )
            fig.update_yaxes(range=[temp_min - temp_padding, temp_max + temp_padding])

            # pick your index window
            # start_idx = 1200
            # end_idx   = 1280
            # draw_end_idx = 1230
            # t0 = df['Time'].iloc[start_idx]
            # t1 = df['Time'].iloc[end_idx]
            # t_draw_end = df['Time'].iloc[draw_end_idx]
            # fig.update_xaxes(range=[t0, t1])

            fig.add_shape(
                type="line",
                xref="paper", x0=0, x1=1,
                yref="y",     y0=water_temp_cutoff, y1=water_temp_cutoff,
                line=dict(color="red", width=1, dash="dash")
            )
            fig.add_annotation(
                xref="paper", x=1, 
                y=water_temp_cutoff,
                xanchor="right", yanchor="bottom",
                text="110 °F Cutoff Temp",
                showarrow=False
            )

            figure_metadata.append({
                'pattern': pattern,
                'file': file,
                'type': 'temperature_pattern'
            })

            all_figs.append(fig)

    return all_figs, figure_metadata

def create_energy_output_plots(dfs, uef_values):
    """Create energy output plots showing instantaneous power and cumulative energy over time."""
    all_figs = []
    figure_metadata = []
    
    # Constants for energy calculation
    DENSITY = 991.53  # kg/m³
    CP = 4190  # J/kg·K
    INLET_TEMP_COL = 'Hot Water Mains Temperature (C)'
    OUTLET_TEMP_COL = 'Hot Water Outlet Temperature (C)'
    OUTLET_VOLUME_COL = 'Hot Water Delivered (L/min)'
    MAX_OUTLET_TEMP = 51.67  # 125°F in Celsius
    MIN_FLOW_RATE = 11.356  # 3 gpm in L/min (3 * 3.78541)

    for i, (file, df) in enumerate(dfs.items()):
        uef = uef_values[i]

        # Extract additional parameters for title
        pcm_mass_col = "PCM Mass (kg)"
        pcm_h_col_pattern = re.compile(r"Water Tank PCM\d+ h \(W/m\^2K\)")
        pcm_sa_col_pattern = re.compile(r"Water Tank PCM\d+ sa_ratio") 
        if pcm_mass_col not in df.columns:
            pcm_mass = 0.0
        else:
            pcm_mass = df[pcm_mass_col].iloc[-1]
        matched_h_cols = next((col for col in df.columns if pcm_h_col_pattern.fullmatch(col)), None) 
        match_sa_col = next((col for col in df.columns if pcm_sa_col_pattern.fullmatch(col)), None) 
        pcm_h = df[matched_h_cols].iloc[-1] if matched_h_cols is not None else 0.0
        pcm_sa = df[match_sa_col].iloc[-1] if match_sa_col is not None else 0.0

        water_volume_col = "Water Volume (L)"
        L_TO_GAL_RATIO = 0.264172
        if water_volume_col not in df.columns:
            water_volume_gal = 50 * .9
        else:
            water_volume_gal = df[water_volume_col].iloc[-1] * L_TO_GAL_RATIO

        # Calculate energy output
        instantaneous_power_kw = None
        cumulative_energy_kwh = None
        total_energy_kwh = 0.0
        
        if INLET_TEMP_COL in df.columns and OUTLET_TEMP_COL in df.columns and OUTLET_VOLUME_COL in df.columns:
            try:
                # Convert to numeric, handling any string values
                inlet_temp = pd.to_numeric(df[INLET_TEMP_COL], errors='coerce')
                outlet_temp = pd.to_numeric(df[OUTLET_TEMP_COL], errors='coerce')
                flow_rate = pd.to_numeric(df[OUTLET_VOLUME_COL], errors='coerce')
                # time_values = pd.to_numeric(df['TimeStamp'], errors='coerce')
                time_values = df['Time']
                
                # Drop any rows with NaN values
                valid_mask = ~(inlet_temp.isna() | outlet_temp.isna() | flow_rate.isna() | time_values.isna())
                if valid_mask.sum() > 1:  # Need at least 2 valid points
                    inlet_temp = inlet_temp[valid_mask]
                    outlet_temp = outlet_temp[valid_mask]
                    flow_rate = flow_rate[valid_mask]
                    time_values = time_values[valid_mask].reset_index(drop=True)
                    
                    # Get effective outlet temperature (min of 125°F and actual outlet temp)
                    effective_outlet_temp = outlet_temp
                    
                    # Get tank outlet flow
                    effective_flow_rate = flow_rate
                    
                    # Calculate delta T
                    delta_t = effective_outlet_temp - inlet_temp
                    
                    # Only calculate power when there's actually hot water being delivered (delta_t > 0)
                    delta_t = delta_t.clip(lower=0)
                    
                    # Calculate instantaneous power (W) = density * cp * flow_rate * delta_t
                    # Convert L/min to L/s by dividing by 60
                    # Convert L to m³ by dividing by 1000
                    flow_rate_m3_s = effective_flow_rate / (60 * 1000)
                    m_dot = DENSITY * flow_rate_m3_s
                    instantaneous_power = CP * m_dot * delta_t
                    instantaneous_power_kw = instantaneous_power / 1000  # Convert W to kW
                    
                    # dynamically compute time step
                    format_string = '%Y-%m-%d %H:%M:%S.%f'
                    time_step = datetime.datetime.strptime(time_values[1], format_string) - datetime.datetime.strptime(time_values[0], format_string)

                    time_step_seconds = time_step.total_seconds()
                    
                    # Calculate cumulative energy in kWh
                    energy_increment_J = (instantaneous_power * time_step_seconds) # Convert to J
                    cumulative_energy_kwh = energy_increment_J.cumsum()
                    total_energy_kwh = cumulative_energy_kwh.iloc[-1]
                    
            except Exception as e:
                print(f"Error calculating energy for {file}: {e}")
                total_energy_kwh = 0.0

        # Create the energy plot
        fig = go.Figure()
        
        if instantaneous_power_kw is not None and cumulative_energy_kwh is not None:
            # Add instantaneous power trace
            fig.add_trace(
                go.Scatter(
                    x=time_values,
                    y=instantaneous_power_kw,
                    mode='lines',
                    name='Instantaneous Power (kW)',
                    line=dict(color='blue'),
                    yaxis='y'
                )
            )
            
            # Add cumulative energy trace on secondary y-axis
            fig.add_trace(
                go.Scatter(
                    x=time_values,
                    y=cumulative_energy_kwh,
                    mode='lines',
                    name='Cumulative Energy (kWh)',
                    line=dict(color='red'),
                    yaxis='y2'
                )
            )
        else:
            # Add empty traces if no data available
            fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode='lines',
                    name='No Energy Data Available',
                    line=dict(color='gray')
                )
            )

        # Add heating mode overlays if available
        if 'Water Heating Mode' in df.columns and instantaneous_power_kw is not None:
            # Get y-axis ranges for overlays
            power_min = instantaneous_power_kw.min() if len(instantaneous_power_kw) > 0 else 0
            power_max = instantaneous_power_kw.max() if len(instantaneous_power_kw) > 0 else 1
            power_range = power_max - power_min
            power_padding = power_range * 0.1
            
            upper_regions = []
            lower_regions = []
            heat_pump_regions = []
            mode_data = df['Water Heating Mode']
            time_data = df['Time']

            # Extract heating mode regions (same logic as before)
            in_upper_segment = False
            for j in range(len(mode_data)):
                mode_str = str(mode_data.iloc[j])
                if 'Upper On' in mode_str and not in_upper_segment:
                    upper_segment_start = j
                    in_upper_segment = True
                elif 'Upper On' not in mode_str and in_upper_segment:
                    upper_segment_end = j
                    in_upper_segment = False
                    start_time = time_data.iloc[upper_segment_start]
                    end_time = time_data.iloc[upper_segment_end]
                    upper_regions.append((start_time, end_time))
            if in_upper_segment:
                upper_segment_end = len(mode_data) - 1
                start_time = time_data.iloc[upper_segment_start]
                end_time = time_data.iloc[upper_segment_end]
                upper_regions.append((start_time, end_time))
                
            in_lower_segment = False
            for j in range(len(mode_data)):
                mode_str = str(mode_data.iloc[j])
                if 'Lower On' in mode_str and not in_lower_segment:
                    lower_segment_start = j
                    in_lower_segment = True
                elif 'Lower On' not in mode_str and in_lower_segment:
                    lower_segment_end = j
                    in_lower_segment = False
                    start_time = time_data.iloc[lower_segment_start]
                    end_time = time_data.iloc[lower_segment_end]
                    lower_regions.append((start_time, end_time))
            if in_lower_segment:
                lower_segment_end = len(mode_data) - 1
                start_time = time_data.iloc[lower_segment_start]
                end_time = time_data.iloc[lower_segment_end]
                lower_regions.append((start_time, end_time))
                
            in_heat_pump_segment = False
            for j in range(len(mode_data)):
                mode_str = str(mode_data.iloc[j])
                if 'Heat Pump On' in mode_str and not in_heat_pump_segment:
                    heat_pump_segment_start = j
                    in_heat_pump_segment = True
                elif 'Heat Pump On' not in mode_str and in_heat_pump_segment:
                    heat_pump_segment_end = j
                    in_heat_pump_segment = False
                    start_time = time_data.iloc[heat_pump_segment_start]
                    end_time = time_data.iloc[heat_pump_segment_end]
                    heat_pump_regions.append((start_time, end_time))
            if in_heat_pump_segment:
                heat_pump_segment_end = len(mode_data) - 1
                start_time = time_data.iloc[heat_pump_segment_start]
                end_time = time_data.iloc[heat_pump_segment_end]
                heat_pump_regions.append((start_time, end_time))

            # Add overlays
            for j, (start_time, end_time) in enumerate(upper_regions):
                fig.add_trace(
                    go.Scatter(
                        x=[start_time, end_time, end_time, start_time, start_time],
                        y=[power_min - power_padding, power_min - power_padding,
                           power_max + power_padding, power_max + power_padding,
                           power_min - power_padding],
                        fill="toself",
                        fillcolor="rgba(255, 0, 0, 0.15)",
                        line=dict(width=0),
                        mode="none",
                        name="Upper Element On" if j == 0 else "",
                        showlegend=True if j == 0 else False,
                        legendgroup="upper_elements",
                        hoverinfo="skip",
                        yaxis='y'
                    )
                )
            for j, (start_time, end_time) in enumerate(lower_regions):
                fig.add_trace(
                    go.Scatter(
                        x=[start_time, end_time, end_time, start_time, start_time],
                        y=[power_min - power_padding, power_min - power_padding,
                           power_max + power_padding, power_max + power_padding,
                           power_min - power_padding],
                        fill="toself",
                        fillcolor="rgba(0, 0, 255, 0.15)",
                        line=dict(width=0),
                        mode="none",
                        name="Lower Element On" if j == 0 else "",
                        showlegend=True if j == 0 else False,
                        legendgroup="lower_elements",
                        hoverinfo="skip",
                        yaxis='y'
                    )
                )
            for j, (start_time, end_time) in enumerate(heat_pump_regions):
                fig.add_trace(
                    go.Scatter(
                        x=[start_time, end_time, end_time, start_time, start_time],
                        y=[power_min - power_padding, power_min - power_padding,
                           power_max + power_padding, power_max + power_padding,
                           power_min - power_padding],
                        fill="toself",
                        fillcolor="rgba(0, 255, 0, 0.15)",
                        line=dict(width=0),
                        mode="none",
                        name="Heat Pump On" if j == 0 else "",
                        showlegend=True if j == 0 else False,
                        legendgroup="heat_pump",
                        hoverinfo="skip",
                        yaxis='y'
                    )
                )

        # Create secondary y-axis for cumulative energy
        fig.update_layout(
            title=f'Energy Output - {file}<br>'
                  f'UEF: {uef:.3f} | PCM h: {pcm_h:.2f} W/m^2K | PCM SA Ratio: {pcm_sa:.2f} | PCM Mass: {pcm_mass:.3f} kg | '
                  f'Water Volume: {water_volume_gal:.1f} gal | Total Energy: {total_energy_kwh:.2e} J',
            xaxis_title='Time',
            yaxis=dict(
                title='Instantaneous Power (kW)',
                side='left'
            ),
            yaxis2=dict(
                title='Cumulative Energy (J)',
                side='right',
                overlaying='y'
            ),
            height=600,
            showlegend=True
        )

        figure_metadata.append({
            'file': file,
            'type': 'energy_output',
            'total_energy_kwh': total_energy_kwh
        })

        all_figs.append(fig)

    return all_figs, figure_metadata

def create_water_flow_temperature_plots(dfs, uef_values, outlet_gpm=3):
    """Create water flow and temperature mixing plots with power state shading."""
    all_figs = []
    figure_metadata = []
    water_temp_cutoff = 43.3333  # 110 F
    GAL_TO_L = 3.78541  # gallons to liters conversion
    outlet_lpm = outlet_gpm * GAL_TO_L  # Convert GPM to L/min

    for i, (file, df) in enumerate(dfs.items()):
        uef = uef_values[i]
        
        # Check if required columns exist
        required_columns = [
            'Hot Water Delivered (L/min)',
            'Hot Water Outlet Temperature (C)',
            'Hot Water Mains Temperature (C)'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Skipping {file}: Missing columns {missing_columns}")
            continue

        # Extract additional parameters for title
        pcm_mass_col = "PCM Mass (kg)"
        pcm_h_col_pattern = re.compile(r"Water Tank PCM\d+ h \(W/m\^2K\)")
        pcm_sa_col_pattern = re.compile(r"Water Tank PCM\d+ sa_ratio") 
        if pcm_mass_col not in df.columns:
            pcm_mass = 0.0
        else:
            pcm_mass = df[pcm_mass_col].iloc[-1]
        matched_h_cols = next((col for col in df.columns if pcm_h_col_pattern.fullmatch(col)), None) 
        match_sa_col = next((col for col in df.columns if pcm_sa_col_pattern.fullmatch(col)), None) 
        pcm_h = df[matched_h_cols].iloc[-1] if matched_h_cols is not None else 0.0
        pcm_sa = df[match_sa_col].iloc[-1] if match_sa_col is not None else 0.0

        water_volume_col = "Water Volume (L)"
        L_TO_GAL_RATIO = 0.264172
        if water_volume_col not in df.columns:
            water_volume_gal = 50 * .9
        else:
            water_volume_gal = df[water_volume_col].iloc[-1] * L_TO_GAL_RATIO

        # Calculate derived values
        df_calc = df.copy()
        
        # Filter for times when hot water is being delivered
        hot_water_mask = df_calc['Hot Water Delivered (L/min)'] > 0
        
        # Calculate cold water flow rate (mains water flow)
        # Calculate cold water flow rate (mains water flow)
        # Only when there's hot water flow, otherwise cold water flow should be 0
        df_calc['Cold Water Flow (L/min)'] = 0.0
        df_calc.loc[hot_water_mask, 'Cold Water Flow (L/min)'] = (
            outlet_lpm - df_calc.loc[hot_water_mask, 'Hot Water Delivered (L/min)']
        )
        
        # Calculate mixed temperature using energy balance
        # T_mixed = (m_hot * T_hot + m_cold * T_cold) / (m_hot + m_cold)
        # Where mass flow rates are proportional to volumetric flow rates (assuming constant density)
        hot_flow = df_calc['Hot Water Delivered (L/min)']
        cold_flow = df_calc['Cold Water Flow (L/min)']
        hot_temp = df_calc['Hot Water Outlet Temperature (C)']
        cold_temp = df_calc['Hot Water Mains Temperature (C)']
        
        # Only calculate mixed temperature when there's hot water flow
        df_calc['Mixed Temperature (C)'] = 0.0
        valid_flow_mask = (hot_flow > 0) & (cold_flow >= 0)
        df_calc.loc[valid_flow_mask, 'Mixed Temperature (C)'] = (
            (hot_flow[valid_flow_mask] * hot_temp[valid_flow_mask] + 
             cold_flow[valid_flow_mask] * cold_temp[valid_flow_mask]) / 
            (hot_flow[valid_flow_mask] + cold_flow[valid_flow_mask])
        )

        # Create the plot
        fig = go.Figure()

        # Define colors for different traces
        colors = {
            'outlet_flow': '#1f77b4',      # Blue
            'hot_delivered': '#ff7f0e',    # Orange
            'hot_temp': '#d62728',         # Red
            'mains_temp': '#2ca02c',       # Green
            'cold_flow': '#9467bd',        # Purple
            'mixed_temp': '#8c564b'        # Brown
        }

        # Only plot outlet flow rate when hot water is being delivered
        outlet_flow_data = pd.Series(0.0, index=df_calc.index)
        outlet_flow_data[hot_water_mask] = outlet_lpm
        
        # Add flow rate traces
        fig.add_trace(
            go.Scatter(
                x=df_calc['Time'],
                y=outlet_flow_data,
                mode='lines',
                name=f'Outlet Flow Rate ({outlet_gpm} GPM)',
                line=dict(color=colors['outlet_flow'], width=2),
                yaxis='y2'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_calc['Time'],
                y=df_calc['Hot Water Delivered (L/min)'],
                mode='lines',
                name='Hot Water Delivered (L/min)',
                line=dict(color=colors['hot_delivered'], width=2),
                yaxis='y2'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_calc['Time'],
                y=df_calc['Cold Water Flow (L/min)'],
                mode='lines',
                name='Cold Water Flow (L/min)',
                line=dict(color=colors['cold_flow'], width=2),
                yaxis='y2'
            )
        )

        # Add temperature traces
        fig.add_trace(
            go.Scatter(
                x=df_calc['Time'],
                y=df_calc['Hot Water Outlet Temperature (C)'],
                mode='lines',
                name='Hot Water Outlet Temperature (°C)',
                line=dict(color=colors['hot_temp'], width=2)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_calc['Time'],
                y=df_calc['Hot Water Mains Temperature (C)'],
                mode='lines',
                name='Water Mains Temperature (°C)',
                line=dict(color=colors['mains_temp'], width=2)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_calc['Time'],
                y=df_calc['Mixed Temperature (C)'],
                mode='lines',
                name='Mixed Output Temperature (°C)',
                line=dict(color=colors['mixed_temp'], width=2)
            )
        )

        # Determine temperature and flow ranges for shading
        temp_cols = ['Hot Water Outlet Temperature (C)', 'Hot Water Mains Temperature (C)', 'Mixed Temperature (C)']
        temp_data = df_calc[temp_cols].values.flatten()
        temp_data = temp_data[~pd.isna(temp_data)]
        temp_min = temp_data.min()
        temp_max = temp_data.max()
        temp_range = temp_max - temp_min
        temp_padding = temp_range * 0.1

        flow_cols = ['Hot Water Delivered (L/min)', 'Cold Water Flow (L/min)']
        flow_data = df_calc[flow_cols].values.flatten()
        flow_data = flow_data[~pd.isna(flow_data)]
        flow_max = max(flow_data.max(), outlet_lpm)
        flow_min = 0
        flow_padding = flow_max * 0.1

        # Add power state shading (same as original function)
        if 'Water Heating Mode' in df.columns:
            upper_regions = []
            lower_regions = []
            heat_pump_regions = []
            mode_data = df['Water Heating Mode']
            time_data = df['Time']

            # Extract upper element regions
            in_upper_segment = False
            for j in range(len(mode_data)):
                mode_str = str(mode_data.iloc[j])
                if 'Upper On' in mode_str and not in_upper_segment:
                    upper_segment_start = j
                    in_upper_segment = True
                elif 'Upper On' not in mode_str and in_upper_segment:
                    upper_segment_end = j
                    in_upper_segment = False
                    start_time = time_data.iloc[upper_segment_start]
                    end_time = time_data.iloc[upper_segment_end]
                    upper_regions.append((start_time, end_time))
            if in_upper_segment:
                upper_segment_end = len(mode_data) - 1
                start_time = time_data.iloc[upper_segment_start]
                end_time = time_data.iloc[upper_segment_end]
                upper_regions.append((start_time, end_time))

            # Extract lower element regions
            in_lower_segment = False
            for j in range(len(mode_data)):
                mode_str = str(mode_data.iloc[j])
                if 'Lower On' in mode_str and not in_lower_segment:
                    lower_segment_start = j
                    in_lower_segment = True
                elif 'Lower On' not in mode_str and in_lower_segment:
                    lower_segment_end = j
                    in_lower_segment = False
                    start_time = time_data.iloc[lower_segment_start]
                    end_time = time_data.iloc[lower_segment_end]
                    lower_regions.append((start_time, end_time))
            if in_lower_segment:
                lower_segment_end = len(mode_data) - 1
                start_time = time_data.iloc[lower_segment_start]
                end_time = time_data.iloc[lower_segment_end]
                lower_regions.append((start_time, end_time))

            # Extract heat pump regions
            in_heat_pump_segment = False
            for j in range(len(mode_data)):
                mode_str = str(mode_data.iloc[j])
                if 'Heat Pump On' in mode_str and not in_heat_pump_segment:
                    heat_pump_segment_start = j
                    in_heat_pump_segment = True
                elif 'Heat Pump On' not in mode_str and in_heat_pump_segment:
                    heat_pump_segment_end = j
                    in_heat_pump_segment = False
                    start_time = time_data.iloc[heat_pump_segment_start]
                    end_time = time_data.iloc[heat_pump_segment_end]
                    heat_pump_regions.append((start_time, end_time))
            if in_heat_pump_segment:
                heat_pump_segment_end = len(mode_data) - 1
                start_time = time_data.iloc[heat_pump_segment_start]
                end_time = time_data.iloc[heat_pump_segment_end]
                heat_pump_regions.append((start_time, end_time))

            # Add overlays for power states
            for j, (start_time, end_time) in enumerate(upper_regions):
                fig.add_trace(
                    go.Scatter(
                        x=[start_time, end_time, end_time, start_time, start_time],
                        y=[temp_min - temp_padding, temp_min - temp_padding,
                           temp_max + temp_padding, temp_max + temp_padding,
                           temp_min - temp_padding],
                        fill="toself",
                        fillcolor="rgba(255, 0, 0, 0.15)",
                        line=dict(width=0),
                        mode="none",
                        name="Upper Element On" if j == 0 else "",
                        showlegend=True if j == 0 else False,
                        legendgroup="upper_elements",
                        hoverinfo="skip"
                    )
                )
            for j, (start_time, end_time) in enumerate(lower_regions):
                fig.add_trace(
                    go.Scatter(
                        x=[start_time, end_time, end_time, start_time, start_time],
                        y=[temp_min - temp_padding, temp_min - temp_padding,
                           temp_max + temp_padding, temp_max + temp_padding,
                           temp_min - temp_padding],
                        fill="toself",
                        fillcolor="rgba(0, 0, 255, 0.15)",
                        line=dict(width=0),
                        mode="none",
                        name="Lower Element On" if j == 0 else "",
                        showlegend=True if j == 0 else False,
                        legendgroup="lower_elements",
                        hoverinfo="skip"
                    )
                )
            for j, (start_time, end_time) in enumerate(heat_pump_regions):
                fig.add_trace(
                    go.Scatter(
                        x=[start_time, end_time, end_time, start_time, start_time],
                        y=[temp_min - temp_padding, temp_min - temp_padding,
                           temp_max + temp_padding, temp_max + temp_padding,
                           temp_min - temp_padding],
                        fill="toself",
                        fillcolor="rgba(0, 255, 0, 0.15)",
                        line=dict(width=0),
                        mode="none",
                        name="Heat Pump On" if j == 0 else "",
                        showlegend=True if j == 0 else False,
                        legendgroup="heat_pump",
                        hoverinfo="skip"
                    )
                )

        # Update layout with dual y-axes
        fig.update_layout(
            title=f'Water Flow and Temperature Analysis - {file}<br>'
                  f'UEF: {uef:.3f} | PCM h: {pcm_h:.2f} W/m^2K | PCM SA Ratio: {pcm_sa:.2f} | PCM Mass: {pcm_mass:.3f} kg | '
                  f'Water Volume: {water_volume_gal:.1f} gal | Outlet: {outlet_gpm} GPM',
            xaxis_title='Time',
            yaxis=dict(
                title='Temperature (°C)',
                side='left',
                range=[temp_min - temp_padding, temp_max + temp_padding]
            ),
            yaxis2=dict(
                title='Flow Rate (L/min)',
                side='right',
                overlaying='y',
                range=[flow_min - flow_padding, flow_max + flow_padding]
            ),
            height=600,
            showlegend=True
        )

        # Add 110°F cutoff line
        fig.add_shape(
            type="line",
            xref="paper", x0=0, x1=1,
            yref="y", y0=water_temp_cutoff, y1=water_temp_cutoff,
            line=dict(color="red", width=1, dash="dash")
        )
        fig.add_annotation(
            xref="paper", x=1, 
            y=water_temp_cutoff,
            xanchor="right", yanchor="bottom",
            text="110 °F Cutoff Temp",
            showarrow=False
        )

        figure_metadata.append({
            'pattern': 'water_flow_temperature',
            'file': file,
            'type': 'water_flow_analysis',
            'outlet_gpm': outlet_gpm
        })

        all_figs.append(fig)

    return all_figs, figure_metadata

def create_heat_exchanger_plot_outlet_temp(dfs):
    """Create a simplified plot focusing only on outlet temperature
    with improved cutoff temperature visualization."""
    
    
    water_temp_cutoff = 43.3333  # 110 F 15 deg delta from 125 F for UEF test
    
    # Create a figure with a single plot
    fig = go.Figure()
    
    # Colors to differentiate between files
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    
    # For tracking min/max values to set axis ranges
    temp_min, temp_max = float('inf'), float('-inf')
    
    # Process datasets and add traces
    for i, (file, df) in enumerate(dfs.items()):
        color = colors[i % len(colors)]
        legendgroup = f"group_{file}"  # Create a unique legend group for this file
        
        # Add Outlet Temperature trace
        if 'Hot Water Outlet Temperature (C)' in df.columns:
            temp_data = df['Hot Water Outlet Temperature (C)']
            temp_min = min(temp_min, temp_data.min())
            temp_max = max(temp_max, temp_data.max() * 1.1)
            
            fig.add_trace(
                go.Scatter(
                    x=df['Time'],
                    y=temp_data,
                    mode='lines',
                    name=f"{file}",
                    line=dict(color=color),
                    legendgroup=legendgroup
                )
            )
    
    # Calculate the adjusted range for temperature axis
    temp_range = temp_max - temp_min
    temp_padding = temp_range * 0.1  # 10% padding
    
    # Set specified x-axis window
    # Note: This should be adjusted based on your actual data
    start_idx = 1200
    end_idx = 1280
    draw_end_idx = 1230
    
    # Apply to each dataset (assuming they have enough points)
    for file, df in dfs.items():
        if len(df) > end_idx:
            # Look up the actual Time values at those positions
            t0 = df['Time'].iloc[start_idx]
            t1 = df['Time'].iloc[end_idx]
            t_draw_end = df['Time'].iloc[draw_end_idx]
            
            # Clamp the x-axis to that slice
            fig.update_xaxes(range=[t0, t1])
            
            # Add vertical marker for draw end
            # fig.add_shape(
            #     type="line",
            #     x0=t_draw_end, x1=t_draw_end,
            #     y0=temp_min - temp_padding, y1=temp_max + temp_padding,
            #     line=dict(color="black", dash="dash"),
            #     xref="x", yref="y"
            # )
            
            # fig.add_annotation(
            #     x=t_draw_end,
            #     y=temp_max + temp_padding,
            #     text="2 GPM Draw End",
            #     showarrow=False,
            #     yshift=10,
            #     xanchor="left"
            # )
            
            # Add horizontal line for cutoff temperature
            fig.add_shape(
                type="line",
                xref="paper", x0=0, x1=1,  # span full width
                yref="y", y0=water_temp_cutoff, y1=water_temp_cutoff,
                line=dict(color="red", width=1, dash="dash")
            )
            
            # Label the cutoff temperature
            fig.add_annotation(
                xref="paper", x=1,
                y=water_temp_cutoff,
                xanchor="right", yanchor="bottom",
                text="110 °F Cutoff Temp",
                showarrow=False
            )
            
            # Only need to do this once
            break
    
    # Update layout and axes titles
    fig.update_layout(
        height=700,
        showlegend=True,
        title_text="PCM Heat Transfer Rate Performance Analysis",
    )
    
    # Update y-axis title and range
    fig.update_yaxes(
        title_text="Outlet Temperature (°C)",
        range=[temp_min - temp_padding, temp_max + temp_padding],
    )
    
    # Update x-axis label
    fig.update_xaxes(title_text="Time")
    
    return fig

# Helper function (assumed to be defined elsewhere in the original code)
def c_to_f(celsius):
    """Convert Celsius to Fahrenheit"""
    return celsius * 9/5 + 32


def create_capacitance_plots(dfs, uef_values):
    """Create separate capacitance plots for each file,
    with upper and lower heating element overlays on the capacitance curves."""
    all_figs = []
    figure_metadata = []  # List to store metadata separately
    
    # Define the pattern to search for capacitance columns
    capacitance_pattern = "Capacitance"
    # Find matching columns for this file

        
        # Find all capacitance columns for this file
    # Extract additional parameters for title
    
    for i, (file, df) in enumerate(dfs.items()):
        # Get UEF value for this file
        uef = uef_values[i]
        capacitance_columns = [col for col in df.columns if capacitance_pattern in col]
        
        if not capacitance_columns:  # Skip if no matching columns found
            continue
            

        pcm_mass_col = "PCM Mass (kg)"
        pcm_h_col_pattern = re.compile(r"Water Tank PCM\d+ h \(W/m\^2K\)")
        pcm_sa_col_pattern = re.compile(r"Water Tank PCM\d+ sa_ratio") 
        if pcm_mass_col not in df.columns:
            pcm_mass = 0.0
        else:
            pcm_mass = df[pcm_mass_col].iloc[-1] # in kg
            
        matched_h_cols = next((col for col in df.columns if pcm_h_col_pattern.fullmatch(col)), None) 
        match_sa_col = next((col for col in df.columns if pcm_sa_col_pattern.fullmatch(col)), None) 
        
        if matched_h_cols is not None:
            pcm_h = df[matched_h_cols].iloc[-1]

        if match_sa_col is not None:
            pcm_sa = df[match_sa_col].iloc[-1]
            
        # Create a new figure
        fig = go.Figure()
        
        # Extract additional parameters for title
        pcm_mass_col = "PCM Mass (kg)"
        if pcm_mass_col not in df.columns:
            pcm_mass = 0.0
        else:
            pcm_mass = df[pcm_mass_col].iloc[-1]  # in kg
            
        water_volume_col = "Water Volume (L)"
        if water_volume_col not in df.columns:
            water_volume_gal = 45.0
        else:
            water_volume_gal = df[water_volume_col].iloc[-1] * L_TO_GAL_RATIO
            
        # Determine the overall capacitance range across all columns
        cap_min = float('inf')
        cap_max = float('-inf')
        for col in capacitance_columns:
            cap_min = min(cap_min, df[col].min())
            cap_max = max(cap_max, df[col].max())
        cap_range = cap_max - cap_min
        cap_padding = cap_range * 0.1
        
        # Add capacitance traces for each matching column
        for col in capacitance_columns:
            fig.add_trace(
                go.Scatter(
                    x=df['Time'],
                    y=df[col],
                    mode='lines',
                    name=col
                )
            )
            
        # Add heating element overlays if the "Water Heating Mode" column is present
        if 'Water Heating Mode' in df.columns:
            upper_regions = []
            lower_regions = []
            mode_data = df['Water Heating Mode']
            time_data = df['Time']
            
            # Identify regions when the upper element is on
            in_upper_segment = False
            for j in range(len(mode_data)):
                mode_str = str(mode_data.iloc[j])
                if 'Upper On' in mode_str and not in_upper_segment:
                    upper_segment_start = j
                    in_upper_segment = True
                elif 'Upper On' not in mode_str and in_upper_segment:
                    upper_segment_end = j
                    in_upper_segment = False
                    start_time = time_data.iloc[upper_segment_start]
                    end_time = time_data.iloc[upper_segment_end]
                    upper_regions.append((start_time, end_time))
            if in_upper_segment:
                upper_segment_end = len(mode_data) - 1
                start_time = time_data.iloc[upper_segment_start]
                end_time = time_data.iloc[upper_segment_end]
                upper_regions.append((start_time, end_time))
                
            # Identify regions when the lower element is on
            in_lower_segment = False
            for j in range(len(mode_data)):
                mode_str = str(mode_data.iloc[j])
                if 'Lower On' in mode_str and not in_lower_segment:
                    lower_segment_start = j
                    in_lower_segment = True
                elif 'Lower On' not in mode_str and in_lower_segment:
                    lower_segment_end = j
                    in_lower_segment = False
                    start_time = time_data.iloc[lower_segment_start]
                    end_time = time_data.iloc[lower_segment_end]
                    lower_regions.append((start_time, end_time))
            if in_lower_segment:
                lower_segment_end = len(mode_data) - 1
                start_time = time_data.iloc[lower_segment_start]
                end_time = time_data.iloc[lower_segment_end]
                lower_regions.append((start_time, end_time))
                
            # Add red overlay for upper element on regions
            for j, (start_time, end_time) in enumerate(upper_regions):
                fig.add_trace(
                    go.Scatter(
                        x=[start_time, end_time, end_time, start_time, start_time],
                        y=[cap_min - cap_padding, cap_min - cap_padding,
                           cap_max + cap_padding, cap_max + cap_padding,
                           cap_min - cap_padding],
                        fill="toself",
                        fillcolor="rgba(255, 0, 0, 0.3)",
                        line=dict(width=0),
                        mode="none",
                        name="Upper Element On" if j == 0 else "",
                        showlegend=True if j == 0 else False,
                        legendgroup="upper_elements",
                        hoverinfo="skip"
                    )
                )
            # Add blue overlay for lower element on regions
            for j, (start_time, end_time) in enumerate(lower_regions):
                fig.add_trace(
                    go.Scatter(
                        x=[start_time, end_time, end_time, start_time, start_time],
                        y=[cap_min - cap_padding, cap_min - cap_padding,
                           cap_max + cap_padding, cap_max + cap_padding,
                           cap_min - cap_padding],
                        fill="toself",
                        fillcolor="rgba(0, 0, 255, 0.3)",
                        line=dict(width=0),
                        mode="none",
                        name="Lower Element On" if j == 0 else "",
                        showlegend=True if j == 0 else False,
                        legendgroup="lower_elements",
                        hoverinfo="skip"
                    )
                )
                
        # Update the layout with UEF and other file-specific info in the title
        fig.update_layout(
            title=f'PCM Capacitances - {file}<br>'
                      f'UEF: {uef:.3f} | PCM h: {pcm_h:.2f} W/m^2K | PCM SA Ratio: {pcm_sa:.2f} | PCM Mass: {pcm_mass:.3f} kg | '
                      f'Water Volume: {water_volume_gal:.1f} gal',
            xaxis_title='Time',
            yaxis_title='Capacitance (J/K)',
            height=600,
            showlegend=True
        )
        fig.update_yaxes(range=[cap_min - cap_padding, cap_max + cap_padding])
        
        # Store metadata for later reference
        figure_metadata.append({
            'pattern': 'PCM_Capacitance',
            'file': file,
            'type': 'capacitance'
        })
        
        all_figs.append(fig)
        
    return all_figs, figure_metadata

def calculate_net_water_energy(volume, temperature, temperature_difference):
    
    water_properties = lookup_water_properties(temperature)
    density = water_properties["density"]
    # volume is in Liters
    water_weight = density * volume / 1e3 # in kg
    
    
    return water_weight * temperature_difference * 4184 # J

def calculate_uef(dfs):
    # calculate UEF of the water tank
    uef_values = []      # make sure in W*min                            
    
    for df in dfs.values():
        uef = calculate_single_uef(df)
        uef_values.append(uef)
    
    return uef_values

def calculate_single_uef(df):
    # calculate UEF of the water tank
    Q_cons = (
        df["Water Heating Electric Power (kW)"].sum() * 1000
    )  # not sure if this is the correct term that I should be pulling
    Q_load = df[
        "Hot Water Delivered (W)"
    ].sum()  # not sure if this is the correct term that I should be pulling
    
    PCM_Q_Heat_to_Water= calculate_net_PCM_heat(df)     # make sure in W*min
    PCM_net_enthalpy = calculate_net_PCM_enthalpy(df)                               
    PCM_net_heat_loss = PCM_net_enthalpy / 60           # make sure in W*min
    water_net_temp_delta = calculate_net_water_temp(df)
    
    water_volume_col = "Water Volume (L)"
    if water_volume_col not in df.columns:
        
        Q_cons_total = Q_cons - PCM_net_heat_loss           # make sure in W*min
        # water_net_energy = calculate_net_water_energy(102.2, df['Hot Water Average Temperature (C)'].iloc[-1], water_net_temp_delta) / 60                            
        # UEF = Q_load / Q_cons_total
        UEF = Q_load / Q_cons
    else:
        water_volume = df[water_volume_col].iloc[-1]
        water_net_energy = calculate_net_water_energy(water_volume, df['Hot Water Average Temperature (C)'].iloc[-1], water_net_temp_delta) / 60 # make sure in W*min
        Q_cons_total = Q_cons - PCM_net_heat_loss - water_net_energy          # make sure in W*min                            
        # UEF = Q_load / Q_cons_total
        UEF = Q_load / Q_cons
        
    
    return UEF

def calculate_net_PCM_heat(df):

    # Dynamically find all PCM enthalpy columns.
    pcm_column = 'Total PCM Heat Injected (W)'
    
    if pcm_column not in df.columns:
        return 0
    
    # Sum the PCM enthalpy columns row-wise.

    net_PCM_to_water_heat_Transfer = df[pcm_column].sum()
    
    return net_PCM_to_water_heat_Transfer

def calculate_net_PCM_enthalpy(df):
    # Dynamically find all PCM enthalpy columns.
    pcm_column = 'Total PCM Enthalpy (J)'
    
    if pcm_column not in df.columns:
        return 0

    net_PCM_enthalpy = df[pcm_column].iloc[-1] - df[pcm_column].iloc[0]
    
    return net_PCM_enthalpy


def plot_individual_energy_values(dfs):
    """
    Create a separate plot for each individual H_ value (H_WH1, H_WH2, etc.) comparing across files.
    
    Args:
        dfs (dict): Dictionary of DataFrames with filenames as keys
        
    Returns:
        tuple: (list of figures, list of metadata)
    """
    all_figures = []
    all_metadata = []
    
    # First, find all unique H_ columns across all files
    all_h_columns = set()
    for df in dfs.values():
        all_h_columns.update([col for col in df.columns if col.startswith('T_PCM')])
    
    # Sort columns to ensure consistent order
    all_h_columns = sorted(all_h_columns)
    
    # Create a separate plot for each H_ column
    for column in all_h_columns:
        # Create figure
        fig = go.Figure()
        
        # Add a trace for each file that has this column
        for file, df in dfs.items():
            if column in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['Time'],
                        y=df[column],
                        mode='lines',
                        name=f'{file}'
                    )
                )
        
        # Update layout
        fig.update_layout(
            title=f'{column} Comparison Across Files',
            xaxis_title='Time',
            yaxis_title='Energy',
            height=600,
            showlegend=True,
            legend_title_text='Files'
        )
        
        # Store metadata
        metadata = {
            'type': 'individual_energy',
            'column': column
        }
        
        all_figures.append(fig)
        all_metadata.append(metadata)
    
    return all_figures, all_metadata

def create_heat_exchanger_plots(dfs):
    """Create plots with dual y-axes for temperature and power data,
    with the power y-axis scaled larger and grouped power > 0 overlays for each dataset.
    The third subplot displays heating element status overlays alongside temperature data."""
    water_temp_cutoff = 43.3333  # 110 F 15 deg delta from 125 F for UEF test
    # water_temp_cutoff = 40.5556  # 105
        
    # Create a figure with 3 subplot rows and 1 column
    fig = make_subplots(
        rows=3, 
        cols=1,
        subplot_titles=(
            'Hot Water Outlet Temperature and Heating Power',
            'Hot Water Average Temperature and Heating Power',
            'Hot Water Outlet Temperature and Heating Element Status'
        ),
        specs=[
            [{"secondary_y": True}], 
            [{"secondary_y": True}],
            [{"secondary_y": True}]
        ],
        vertical_spacing=0.13,
        row_heights=[0.33, 0.33, 0.34]  # Make bottom plot slightly larger
    )
    
    # Colors to differentiate between files
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    
    # For tracking min/max values to set axis ranges
    temp_min, temp_max = float('inf'), float('-inf')
    power_min, power_max = float('inf'), float('-inf')
    
    # Process datasets and add traces
    for i, (file, df) in enumerate(dfs.items()):
        color = colors[i % len(colors)]
        legendgroup = f"group_{file}"  # Create a unique legend group for this file
        
        # 1. First subplot: Outlet Temperature and Power
        
        # Add Outlet Temperature trace
        if 'Hot Water Outlet Temperature (C)' in df.columns:
            temp_data = df['Hot Water Outlet Temperature (C)']
            temp_min = min(temp_min, temp_data.min())
            temp_max = max(temp_max, temp_data.max()*1.1)
            
            fig.add_trace(
                go.Scatter(
                    x=df['Time'],
                    y=temp_data,
                    mode='lines',
                    name=f"{file} - Outlet Temp",
                    line=dict(color=color),
                    legendgroup=legendgroup
                ),
                row=1, col=1, secondary_y=False
            )
            
            # Add hot water cutoff line
            fig.add_trace(
                go.Scatter(
                    x=df['Time'], y=[water_temp_cutoff for _ in df['Time']],  # invisible point 
                    mode='lines',
                    line=dict(color='red', width=1),
                    name=f'Hot Water Cutoff Temp ({c_to_f(water_temp_cutoff)}°F)', 
                    showlegend=True
                ),
                row=1, col=1
            )

        
        #
        # 2. Second subplot: Average Temperature and Power
        
        # Add Average Temperature trace
        if 'Hot Water Average Temperature (C)' in df.columns:
            avg_temp_data = df['Hot Water Average Temperature (C)']
            temp_min = min(temp_min, avg_temp_data.min())
            temp_max = max(temp_max, avg_temp_data.max()*1.1)
            
            fig.add_trace(
                go.Scatter(
                    x=df['Time'],
                    y=avg_temp_data,
                    mode='lines',
                    name=f"{file} - Avg Temp",
                    line=dict(color=color),
                    legendgroup=legendgroup,
                    showlegend=True
                ),
                row=2, col=1, secondary_y=False
            )
        
        
        # 3. Third subplot: Outlet Temperature and Water Heating Element Status with Overlays
        if 'Hot Water Outlet Temperature (C)' in df.columns:
            # Add the outlet temperature again in the third subplot
            fig.add_trace(
                go.Scatter(
                    x=df['Time'],
                    y=temp_data,
                    mode='lines',
                    name=f"{file} - Outlet Temp (Heater Mode)",
                    line=dict(color=color),
                    legendgroup=legendgroup,
                    showlegend=True
                ),
                row=3, col=1, secondary_y=False
            )
            # Add hot water cutoff line
            fig.add_trace(
                go.Scatter(
                    x=df['Time'], y=[water_temp_cutoff for _ in df['Time']],  # invisible point 
                    mode='lines',
                    line=dict(color='red', width=1),
                    name='Hot Water Cutoff Temp (110°F / 43.33°C)',
                    showlegend=True
                ),
                row=3, col=1
            )
        
        # Calculate the adjusted ranges for both axes
        temp_range = temp_max - temp_min
        temp_padding = temp_range * 0.1  # 10% padding
        
        power_range = power_max - power_min
        power_padding = power_range * 0.1  # 10% padding
        scaled_power_max = (power_max + power_padding) * 5
        scaled_power_min = power_min 
    
        # Add Water Heating Mode overlays
        if 'Water Heating Mode' in df.columns:
            # Process each mode value to identify regions where elements are on
            upper_regions = []
            lower_regions = []
            
            if len(df) > 0:
                mode_data = df['Water Heating Mode']
                time_data = df['Time']
                
                # Find regions where upper element is on
                in_upper_segment = False
                for j in range(len(mode_data)):
                    mode_str = str(mode_data.iloc[j])
                    upper_on = 'Upper On' in mode_str
                    
                    if upper_on and not in_upper_segment:  # Start of a segment
                        upper_segment_start = j
                        in_upper_segment = True
                    elif not upper_on and in_upper_segment:  # End of a segment
                        upper_segment_end = j
                        in_upper_segment = False
                        start_time = time_data.iloc[upper_segment_start]
                        end_time = time_data.iloc[upper_segment_end]
                        upper_regions.append((start_time, end_time))
                
                if in_upper_segment:  # If the last segment extends to the end
                    upper_segment_end = len(mode_data)
                    start_time = time_data.iloc[upper_segment_start]
                    end_time = time_data.iloc[upper_segment_end]
                    upper_regions.append((start_time, end_time))
                
                # Find regions where lower element is on
                in_lower_segment = False
                for j in range(len(mode_data)):
                    mode_str = str(mode_data.iloc[j])
                    lower_on = 'Lower On' in mode_str
                    
                    if lower_on and not in_lower_segment:  # Start of a segment
                        lower_segment_start = j
                        in_lower_segment = True
                    elif not lower_on and in_lower_segment:  # End of a segment
                        lower_segment_end = j
                        in_lower_segment = False
                        start_time = time_data.iloc[lower_segment_start]
                        end_time = time_data.iloc[lower_segment_end]
                        lower_regions.append((start_time, end_time))
                
                if in_lower_segment:  # If the last segment extends to the end
                    lower_segment_end = len(mode_data)-1
                    start_time = time_data.iloc[lower_segment_start]
                    end_time = time_data.iloc[lower_segment_end]
                    lower_regions.append((start_time, end_time))
            
            # Create overlays for upper element (blue)
            temp_padding = 0
            for j, (start_time, end_time) in enumerate(upper_regions):
                fig.add_trace(
                    go.Scatter(
                        x=[start_time, end_time, end_time, start_time, start_time],
                        y=[temp_min - temp_padding, temp_min - temp_padding, 
                           temp_max + temp_padding, temp_max + temp_padding, 
                           temp_min - temp_padding],
                        fill="toself",
                        fillcolor="rgba(0, 0, 255, 0.3)",  # Blue with transparency
                        line=dict(width=0),
                        mode="none",
                        name=f"{file} - Upper Element On",
                        legendgroup=legendgroup,
                        showlegend=True if j == 0 else False,
                        hoverinfo="skip"
                    ),
                    row=3, col=1, secondary_y=False
                )
            
            # Create overlays for lower element (red)
            for j, (start_time, end_time) in enumerate(lower_regions):
                fig.add_trace(
                    go.Scatter(
                        x=[start_time, end_time, end_time, start_time, start_time],
                        y=[temp_min - temp_padding, temp_min - temp_padding, 
                           temp_max + temp_padding, temp_max + temp_padding, 
                           temp_min - temp_padding],
                        fill="toself",
                        fillcolor="rgba(255, 0, 0, 0.3)",  # Red with transparency
                        line=dict(width=0),
                        mode="none",
                        name=f"{file} - Lower Element On",
                        legendgroup=legendgroup,
                        showlegend=True if j == 0 else False,
                        hoverinfo="skip"
                    ),
                    row=3, col=1, secondary_y=False
                )
    
    
    
    # Update layout and axes titles
    fig.update_layout(
        height=1400,
        showlegend=True,
        title_text="PCM Performance Analysis",
        legend=dict(
            groupclick="togglegroup"
        )
    )
    
    # Update y-axis titles and ranges for first subplot
    fig.update_yaxes(
        title_text="Outlet Temperature (°C)",
        range=[temp_min - temp_padding, temp_max + temp_padding],
        row=1, col=1, secondary_y=False
    )
    fig.update_yaxes(
        title_text="Power (W)",
        range=[scaled_power_min, scaled_power_max],
        row=1, col=1, secondary_y=True
    )
    
    # Update y-axis titles and ranges for second subplot
    fig.update_yaxes(
        title_text="Average Temperature (°C)",
        range=[temp_min - temp_padding, temp_max + temp_padding],
        row=2, col=1, secondary_y=False
    )
    fig.update_yaxes(
        title_text="Power (W)",
        range=[scaled_power_min, scaled_power_max],
        row=2, col=1, secondary_y=True
    )
    
    # Update y-axis titles and ranges for third subplot (Heating Element Status)
    fig.update_yaxes(
        title_text="Outlet Temperature (°C)",
        range=[temp_min - temp_padding, temp_max + temp_padding],
        row=3, col=1, secondary_y=False
    )
    fig.update_yaxes(
        title_text="Heating Element Status",
        visible=False,  # Hide the secondary y-axis since we're using overlays
        row=3, col=1, secondary_y=True
    )
    
    # Update x-axis labels
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=1)
    
    return fig

def save_plots(figures, metadata, output_folder='plots'):
    """
    Save all plots in multiple formats.
    
    Args:
        figures: List of plotly figures
        metadata: List of metadata dictionaries corresponding to figures
        output_folder: Folder to save the plots in
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Save each figure
    for fig, meta in zip(figures, metadata):
        # Generate base filename based on metadata
        if meta['type'] == 'temperature_pattern':
            base_name = f"{meta['pattern']}_{meta['file'].replace('.csv', '')}"
        else:
            base_name = "outlet_temperature_comparison"
        
        # Save in different formats
        # HTML (interactive)
        # fig.write_html(os.path.join(output_folder, f"{base_name}.html"))
        
        # Static images
        fig.write_image(os.path.join(output_folder, f"{base_name}.png"))
        # fig.write_image(os.path.join(output_folder, f"{base_name}.pdf"))

def find_energy_columns(df):
    """
    Find all columns that start with 'H_' pattern.
    
    Args:
        df (pandas.DataFrame): Input DataFrame
        
    Returns:
        list: List of column names matching the pattern
    """
    return [col for col in df.columns if col.startswith('H_')]

def calculate_energy_sums(dfs):
    """
    Calculate the sum of all energy columns (starting with 'H_') for each DataFrame.
    
    Args:
        dfs (dict): Dictionary of DataFrames with filenames as keys
        
    Returns:
        dict: Dictionary with filenames as keys and dictionaries of energy sums as values
    """
    energy_sums = {}
    
    for file, df in dfs.items():
        # Find all energy columns
        energy_cols = find_energy_columns(df)
        
        net_enthalpy = calculate_net_PCM_enthalpy(df)
        net_enthalpy = net_enthalpy/60 # conver to W*min
        
        # Calculate sums and store in nested dictionary
        sums = {
            'total_energy': sum(df[energy_cols].sum()) - net_enthalpy,  # Total sum across all H_ columns
            'column_sums': {col: df[col].sum() - net_enthalpy for col in energy_cols},  # Individual column sums
            'timestep_sums': df[energy_cols].sum(axis=1) - net_enthalpy # Sum at each timestep
        }
        
        
        energy_sums[file] = sums
    
    return energy_sums

def plot_energy_comparison(dfs, energy_sums):
    """
    Create plots comparing energy values across different files.
    
    Args:
        dfs (dict): Dictionary of DataFrames
        energy_sums (dict): Dictionary of energy sums from calculate_energy_sums
        
    Returns:
        tuple: (figure, metadata)
    """
    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=('Cumulative Energy by File', 
                                     'Energy Rate over Time'))
    
    # Plot cumulative energy for each file
    for file, sums in energy_sums.items():
        timestep_sums = sums['timestep_sums']
        cumulative_energy = timestep_sums.cumsum()
        
        fig.add_trace(
            go.Scatter(
                x=dfs[file]['Time'],
                y=cumulative_energy,
                name=f'{file} (Cumulative)',
                mode='lines'
            ),
            row=1, col=1
        )
        
        # Plot energy rate over time
        fig.add_trace(
            go.Scatter(
                x=dfs[file]['Time'],
                y=timestep_sums,
                name=f'{file} (Rate)',
                mode='lines'
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=900,
        title='Energy Analysis Comparison',
        showlegend=True
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="Cumulative Energy", row=1, col=1)
    fig.update_yaxes(title_text="Energy Rate", row=2, col=1)
    
    # Update x-axes labels
    fig.update_xaxes(title_text="Time", row=2, col=1)
    
    return fig, {'type': 'energy_comparison'}

def plot_pcm_enthalpies(profiles, names=None):
    """
    Plot cp and Enthalpy vs Temperature for multiple PCM profiles on the same plot,
    using solid lines for cp and dashed lines for enthalpy, with distinct colors per PCM.

    Args:
        profiles (list of np.ndarray): Each entry is an array with columns
                                       [Temp (C), cp (J/g-C), Enthalpy (J/kg)].
        names (list of str, optional): Labels for each PCM. Defaults to ['PCM 1', 'PCM 2', …].

    Returns:
        go.Figure: Plotly figure object.
    """
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    n = len(profiles)
    if names is None:
        names = [f'PCM {i+1}' for i in range(n)]
        
    for i, df in enumerate(profiles):
        color = colors[i % len(colors)]
        label = names[i]
        # cp curve: solid line, no markers
        fig.add_trace(go.Scatter(
            x=df[:, 0],
            y=df[:, 1],
            name=f'{label} cp',
            mode='lines',
            line=dict(color=color, dash='solid'),
            yaxis='y1'
        ))
        # enthalpy curve: dashed line, no markers
        fig.add_trace(go.Scatter(
            x=df[:, 0],
            y=df[:, 2],
            name=f'{label} enthalpy',
            mode='lines',
            line=dict(color=color, dash='dash'),
            yaxis='y2'
        ))
        
        # add vertical line at 125 F
        # compute full cp‑axis range
        all_cp = np.hstack([df[:,1] for df in profiles])
        cp_min, cp_max = all_cp.min(), all_cp.max()

        setpoint_C = (125 - 32) / 1.8  # 125 °F in °C
        setpoint_C2 = (110 -32) / 1.8  # 110 °F in °C
        
        fig.add_trace(go.Scatter(
            x=[setpoint_C, setpoint_C],
            y=[cp_min,   cp_max],
            mode='lines',
            line=dict(color='red', dash='solid'),
            name='125°F Setpoint',
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=[setpoint_C2, setpoint_C2],
            y=[cp_min,   cp_max],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='110°F Cutoff Temp',
            showlegend=True
        ))

    fig.update_layout(
        title='PCM cp and Enthalpy vs Temperature',
        xaxis=dict(title='Temperature (°C)'),
        yaxis=dict(title='cp (J/g-°C)', showgrid=False),
        yaxis2=dict(
            title='Enthalpy (J/kg)',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        # legend=dict(x=0.05, y=0.95),
        height=600
    )
    return fig



# ANSI color codes
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"

graphing_results_folder = "../OCHRE_output/results/"


def plot_worker(args):
    df, uef = args
    temp_plots, _ = create_temperature_plots({None: df}, uef_values=[uef], patterns=['T_WH', 'T_PCM'])
    return temp_plots

def process_dataset(file_key, df, uef_value, patterns=['T_WH', 'T_PCM']):
    """Process a single dataset and return the temperature plots"""
    # This assumes create_temperature_plots returns plots for a single dataframe
    df = {df[0]: df[1]}
    plots, _ = create_temperature_plots(df, uef_values=[uef_value], patterns=patterns)
    return plots

def process_energy_dataset(file_key, df, uef_value, patterns=['T_WH', 'T_PCM']):
    """Process a single dataset and return the temperature plots"""
    # This assumes create_temperature_plots returns plots for a single dataframe
    df = {df[0]: df[1]}
    plots, _ = create_energy_output_plots(df, uef_values=[uef_value])
    return plots

def parallel_create_temperature_plots(dfs, uef_values, patterns=['T_WH', 'T_PCM'], num_processes=None):
    """
    Create temperature plots in parallel using multiprocessing
    
    Args:
        dfs: Dictionary of dataframes (key: file_name, value: dataframe)
        uef_values: Dictionary of UEF values corresponding to each dataframe key
        patterns: List of temperature patterns to plot
        num_processes: Number of processes to use (defaults to CPU count)
        
    Returns:
        all_plots: List of all plots from all dataframes
    """
    # if set(dfs.keys()) != set(uef_values.keys()):
    
    # Default to number of CPUs if not specified
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    # Create a pool of workers
    pool = multiprocessing.Pool(processes=min(num_processes, len(dfs)))
    
    # Create a partial function with fixed patterns argument
    process_func = functools.partial(process_dataset, patterns=patterns)
    
    # Create task arguments - one tuple for each file
    # zip uef with a tuple of the file_key and df
    tasks = [(file_key, df, uef_value) for file_key, df, uef_value in zip(dfs.keys(), dfs.items(), uef_values)]
    
    # Process each dataframe and UEF value pair in parallel
    results = pool.starmap(process_func, tasks)
    
    # Close the pool and wait for all processes to complete
    pool.close()
    pool.join()
    
    # Flatten the list of lists into a single list of plots
    all_plots = [plot for sublist in results for plot in sublist]
    
    return all_plots


def parallel_create_energy_output_plots(dfs, uef_values, patterns=['T_WH', 'T_PCM'], num_processes=None):
    """
    Create temperature plots in parallel using multiprocessing
    
    Args:
        dfs: Dictionary of dataframes (key: file_name, value: dataframe)
        uef_values: Dictionary of UEF values corresponding to each dataframe key
        patterns: List of temperature patterns to plot
        num_processes: Number of processes to use (defaults to CPU count)
        
    Returns:
        all_plots: List of all plots from all dataframes
    """
    # if set(dfs.keys()) != set(uef_values.keys()):
    
    # Default to number of CPUs if not specified
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    # Create a pool of workers
    pool = multiprocessing.Pool(processes=min(num_processes, len(dfs)))
    
    # Create a partial function with fixed patterns argument
    process_func = functools.partial(process_energy_dataset, patterns=patterns)
    
    # Create task arguments - one tuple for each file
    # zip uef with a tuple of the file_key and df
    tasks = [(file_key, df, uef_value) for file_key, df, uef_value in zip(dfs.keys(), dfs.items(), uef_values)]
    
    # Process each dataframe and UEF value pair in parallel
    results = pool.starmap(process_func, tasks)
    
    # Close the pool and wait for all processes to complete
    pool.close()
    pool.join()
    
    # Flatten the list of lists into a single list of plots
    all_plots = [plot for sublist in results for plot in sublist]
    
    return all_plots



def display_plot(plot, delay=0):
    """
    Display a single plot.
    
    Args:
        plot: The plot object to display
        delay: Optional delay in seconds before showing the plot
    """
    if delay > 0:
        time.sleep(delay)
    plot.show()
    return True


def parallel_display_plots(plots, stagger_delay=0, num_processes=None):
    """
    Display multiple plots in parallel using multiprocessing
    
    Args:
        plots: List of plot objects to display
        stagger_delay: Delay between plot displays in seconds (0 for simultaneous)
        num_processes: Number of processes to use (defaults to CPU count)
        
    Returns:
        True if all plots were displayed successfully
    """
    if not plots:
        return True
    
    # Default to number of CPUs if not specified
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    # Create a pool of workers
    pool = multiprocessing.Pool(processes=min(num_processes, len(plots)))
    
    # Calculate delays if staggering is requested
    if stagger_delay > 0:
        delays = [i * stagger_delay for i in range(len(plots))]
    else:
        delays = [0] * len(plots)
    
    # Create a partial function for displaying plots
    display_func = display_plot
    
    # Display plots in parallel
    results = pool.starmap(display_func, zip(plots, delays))
    
    # Close the pool and wait for all processes to complete
    pool.close()
    pool.join()
    
    return all(results)


# Example usage:
if __name__ == "__main__":
    # Load data
    _start_time = time.perf_counter()
    _start_time_plot_results = time.perf_counter()
    print(os.getcwd())
    
    dfs  = load_data(results_folder=graphing_results_folder)
    print(f"Data loading time: {time.perf_counter() - _start_time:.2f} seconds")
    
    # _uef_time = time.perf_counter()
    # uef = calculate_uef(dfs)
    # print(f"UEF calculation time: {time.perf_counter() - _uef_time:.2f} seconds")

    # _pool_time = time.perf_counter()
    # all_plots = parallel_create_temperature_plots(dfs, uef_values=uef, patterns=['T_WH', 'T_PCM'])
    # print(f"Temp chart processing pool time: {time.perf_counter() - _pool_time:.2f} seconds")
    
    # # # # # Display all plots
    # _plot_time = time.perf_counter()
    # parallel_display_plots(all_plots, stagger_delay=0.1)  # 0.1 second delay between plots
    # print(f"Temp chart display pool time: {time.perf_counter() - _plot_time:.2f} seconds")
    
    # _plot_time = time.perf_counter()
    # all_plots = parallel_create_energy_output_plots(dfs, uef_values=uef, patterns=['T_WH', 'T_PCM'])
    # print(f"Energy output processing pool time: {time.perf_counter() - _plot_time:.2f} seconds")
    
    # _plot_time = time.perf_counter()
    # parallel_display_plots(all_plots, stagger_delay=0.1)  # 0.1 second delay between plots
    # print(f"Energy output display pool time: {time.perf_counter() - _plot_time:.2f} seconds")

    # Create water flow and temperature plots
    # _pool_time = time.perf_counter()
    # figures, figure_metadata = create_water_flow_temperature_plots(dfs, uef_values=uef, outlet_gpm=3)
    # print(f"Water flow and temperature processing pool time: {time.perf_counter() - _pool_time:.2f} seconds")
    # for fig in figures:
    #     fig.show()

    # # Draw data summary
    _hot_water_delivered_pool_time = time.perf_counter()
    output = parallel_calculate_hot_water_delivered(dfs)
    print(f"Hot water delivered pool time: {time.perf_counter() - _hot_water_delivered_pool_time:.2f} seconds")
    
    # _hot_water_plot_time = time.perf_counter()
    # plot_draw_event_summary(output)
    # print(f"Hot water plot time: {time.perf_counter() - _hot_water_plot_time:.2f} seconds")
    
    
    plot_draw_events(output)
    plot_totals(output)
    
    # draw 2d matrix plot
    # plot_comparison(dfs, output)
    
    # pcms = [np.loadtxt(os.path.join(os.path.dirname(__file__), "..", "ochre", "Models", f"cp_h-T_data_shifted_{i}F.csv"), delimiter=",", skiprows=1) for i in range(110, 142, 2)]
    pcms = [np.loadtxt(os.path.join(os.path.dirname(__file__), "..", "ochre", "Models", "cp_h-T_data_shifted_120F.csv"), delimiter=",", skiprows=1)]
    pcms.append(np.loadtxt(os.path.join(os.path.dirname(__file__), "..", "ochre", "Models", "90-cp_h-T_data_shifted_120F.csv"), delimiter=",", skiprows=1))
    pcms.append(np.loadtxt(os.path.join(os.path.dirname(__file__), "..", "ochre", "Models", "60-40_PCM55-TPU_cp-h-T_data_shifted_120F.csv"), delimiter=",", skiprows=1))
    
    # pcms_names = [f'90% Infiltrated Graphite PCM {i}F' for i in range(110, 142, 2)]
    pcms_names = ['Bulk PCM', '90% Graphite infiltrated PCM', '60-40 PCM polymer mix PCM 120F'] 
    
    fig = plot_pcm_enthalpies(pcms, pcms_names)
    fig.show()
    
    print(f"{BOLD}{GREEN}All Plots created in {time.perf_counter() - _start_time_plot_results:.2f} seconds{RESET}")
    
    _end_time = time.perf_counter()
    total_time = _end_time - _start_time
    print(f"\n{BOLD}{RED}Total execution time: {total_time:.2f} seconds{RESET}")
    
    # # Combine all figures and metadata
    # all_figures = pattern_figures + [outlet_temp_fig]
    # all_metadata = pattern_metadata + [outlet_metadata]
    
    # # Save all plots
    # # save_plots(all_figures, all_metadata)
    
    # # Show all figures
    # for fig in all_figures:
    #     fig.show()
    
    # # Print UEF values
    # for file, value in uef_values.items():
    #     print(f"UEF for {file}:\t {value:.3f}")