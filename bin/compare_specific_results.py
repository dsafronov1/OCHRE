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
from create_output_csv import export_draw_outputs_csv


L_TO_GAL_RATIO = 0.264172

def load_data(results_folder='../OCHRE_output/results/'):
    csv_files = [f for f in os.listdir(results_folder) if f.endswith('.csv')]
    if len(csv_files) < 2:
        raise ValueError("At least 2 CSV files are required in the 'results' folder for comparison.")

    def _load_one(name):
        return name, pd.read_csv(os.path.join(results_folder, name))

    with ThreadPoolExecutor() as ex:
        out = dict(ex.map(_load_one, csv_files))

    return out

def find_matching_columns(df, patterns):
    """Find columns that match the given patterns and group them.

    Matches:
      T_WH1, T_WH2, …
      T_PCM7-1-1-1, T_PCM12-3-4-2, …
    """
    column_groups = {pattern: [] for pattern in patterns}
    # build a regex once per pattern
    regexes = {
        pattern: re.compile(rf"^{pattern}\d+(?:-\d+)*$")
        for pattern in patterns
    }

    for col in df.columns:
        for pattern, rx in regexes.items():
            if rx.match(col):
                column_groups[pattern].append(col)

    # sort by all numeric parts (so PCM7-1-1-1 comes after PCM7-1-1-0, etc.)
    for pattern, cols in column_groups.items():
        column_groups[pattern] = sorted(
            cols,
            key=lambda name: tuple(int(n) for n in re.findall(r"\d+", name))
        )

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
    
    
    
def _extract_gallons(fname: str):
    """
    Extract the last 'NNgal' token near the end of the filename.
    Supports endings like: ..._40gal.csv, ..._40gal_1, ...-40gal
    """
    # Match 'Default' (case-insensitive)
    default_match = re.search(r'(default)', fname, re.IGNORECASE)
    default_str = "Default" if default_match else ""

    # Match 'HeatPump', 'Gas', etc. after the PCM or No_PCM tokens
    system_match = re.search(r'(HeatPump|Gas|Electric|Hybrid|Resistance)', fname, re.IGNORECASE)
    system_str = system_match.group(1) if system_match else ""

    # Match setpoint (e.g., setpoint-125F)
    setpoint_match = re.search(r'setpoint[-_]?(\d+)F', fname, re.IGNORECASE)
    setpoint_str = f"{setpoint_match.group(1)}F" if setpoint_match else ""

    # Match gallons (e.g., 50gal)
    gal_match = re.search(r'(\d+)\s*gal', fname, re.IGNORECASE)
    gal_str = f"{gal_match.group(1)}gal" if gal_match else ""

    # Combine with spacing and remove any extra whitespace
    parts = [default_str, system_str, setpoint_str, gal_str]
    return " ".join(p for p in parts if p)

def _extract_heatpump_thickness(fname: str):
    """
    Prefer 'Heatpump_thickness-N.NN' token; if missing, fall back to the first 'thickness-N.NN'.
    """
    m = re.search(r'Heatpump_thickness-([0-9]*\.?[0-9]+)', fname, flags=re.IGNORECASE)
    if not m:
        m = re.search(r'\bthickness-([0-9]*\.?[0-9]+)', fname, flags=re.IGNORECASE)
    return float(m.group(1)) if m else float('inf')

def _sort_key(fname: str):
    """
    Sorting strategy:
      Group 0 = zDefault (sorted by gallons ascending)
      Group 1 = others   (sorted by Heatpump_thickness ascending, then gallons, then name)
    """
    is_default = "zDefault" in fname
    gal = _extract_gallons(fname)
    thk = _extract_heatpump_thickness(fname)
    if is_default:
        return (0, 10**9 if gal is None else gal, 0.0)
    else:
        return (1, thk, 10**9 if gal is None else gal, fname.lower())

def plot_draw_events(draw_outputs):
    """
    For each file, create grouped bar charts of draw events.
    Legend order requirement:
      - All legends for the TOP chart first (Water Volume), then all legends for the BOTTOM chart (Heat Delivered).
      - Within each chart:
          • zDefault first, sorted by tail gallons (e.g., 40gal, 50gal, 65gal)
          • then others, sorted by Heatpump_thickness ascending
    zDefault files use black bars; others use default color cycle.
    Also adds a draw-count number above each event group.
    """
    # Canonical file order
    files_sorted = sorted(draw_outputs.keys(), key=_sort_key)

    # Determine legend rank offsets so top chart legends appear before bottom chart legends
    # (Avoid interleaving by giving bottom traces a large base offset.)
    TOP_LEGEND_BASE = 0
    BOTTOM_LEGEND_BASE = 10_000

    # Max number of events and labels
    max_events = max(len(m['draw_events']) for m in draw_outputs.values())
    event_numbers = [f"Event {i+1}" for i in range(max_events)]

    # Prepare water and heat data in sorted order
    water_data = {}
    heat_data = {}
    for fname in files_sorted:
        metrics = draw_outputs[fname]
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

    # ---- Row 1: Water Volume ----
    for rank, fname in enumerate(files_sorted):
        vals = water_data[fname]
        is_default = "zDefault" in fname
        gal = _extract_gallons(fname)
        label = f"{gal}" if is_default and gal is not None else fname
        params = dict(
            name=label,
            legendrank=TOP_LEGEND_BASE + rank,
            legendgroup="water",
            x=event_numbers,
            y=vals,
            text=[f"{v:.2f}" if v is not None else "" for v in vals],
            textposition='auto'
        )
        if is_default:
            params["marker_color"] = "black"
        bar = go.Bar(**params)
        # Give the first trace in the group a group title
        if rank == 0:
            bar.legendgrouptitle = dict(text="Water Volume")
        fig.add_trace(bar, row=1, col=1)

    # Add draw count number above each group in first row
    for i, label in enumerate(event_numbers):
        col_max = max([v[i] for v in water_data.values() if v[i] is not None], default=0)
        fig.add_annotation(
            x=label,
            y=col_max * 1.05 if col_max else 0,
            text=str(i + 1),
            showarrow=False,
            font=dict(size=12, color="red"),
            row=1, col=1
        )

    # ---- Row 2: Heat Delivered ----
    for rank, fname in enumerate(files_sorted):
        val_list = heat_data[fname]
        is_default = "zDefault" in fname
        gal = _extract_gallons(fname)
        label = f"{gal}" if is_default and gal is not None else fname
        params = dict(
            name=label,
            legendrank=BOTTOM_LEGEND_BASE + rank,  # ensure all heat legends come after water legends
            legendgroup="heat",
            x=event_numbers,
            y=val_list,
            text=[f"{v:.3f}" if v is not None else "" for v in val_list],
            textposition='auto'
        )
        if is_default:
            params["marker_color"] = "black"
        bar = go.Bar(**params)
        if rank == 0:
            bar.legendgrouptitle = dict(text="Heat Delivered")
        fig.add_trace(bar, row=2, col=1)

    fig.update_layout(
        barmode='group',
        title_text="Draw Events Grouped by Event Number Across Files",
        xaxis_title="Draw Event",
        showlegend=True
    )
    fig.show()


def plot_totals(draw_outputs):
    """
    For each file, plot:
      • Total Water Volume (gal)  (top chart)
      • Total Heat Delivered (kWh) (bottom chart)
    Legend order requirement:
      - All legends for the TOP chart first, then all legends for the BOTTOM chart.
      - Within each chart:
          • zDefault first, sorted by gallons ascending
          • then others, sorted by Heatpump_thickness ascending
    zDefault files use black bars; others use default color cycle.
    """
    files_sorted = sorted(draw_outputs.keys(), key=_sort_key)
    TOP_LEGEND_BASE = 0
    BOTTOM_LEGEND_BASE = 10_000

    water_totals = [round(draw_outputs[f]['total_water_volume_gal'], 2) for f in files_sorted]
    heat_totals  = [round(draw_outputs[f]['total_heat_delivered_kWh'], 3) for f in files_sorted]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Total Water Volume (gal)", "Total Heat Delivered (kWh)"),
        shared_xaxes=True
    )

    # Row 1: water totals
    for rank, (fname, val) in enumerate(zip(files_sorted, water_totals)):
        is_default = "zDefault" in fname
        gal = _extract_gallons(fname)
        label = f"{gal}" if is_default and gal is not None else fname
        params = dict(
            name=label,
            legendrank=TOP_LEGEND_BASE + rank,
            legendgroup="water_total",
            x=[fname],
            y=[val],
            text=[f"{val:.2f}"],
            textposition='auto'
        )
        if is_default:
            params["marker_color"] = "black"
        bar = go.Bar(**params)
        if rank == 0:
            bar.legendgrouptitle = dict(text="Total Water Volume")
        fig.add_trace(bar, row=1, col=1)

    # Row 2: heat totals
    for rank, (fname, val) in enumerate(zip(files_sorted, heat_totals)):
        is_default = "zDefault" in fname
        gal = _extract_gallons(fname)
        label = f"{gal}" if is_default and gal is not None else fname
        params = dict(
            name=label,
            legendrank=BOTTOM_LEGEND_BASE + rank,
            legendgroup="heat_total",
            x=[fname],
            y=[val],
            text=[f"{val:.3f}"],
            textposition='auto'
        )
        if is_default:
            params["marker_color"] = "black"
        bar = go.Bar(**params)
        if rank == 0:
            bar.legendgrouptitle = dict(text="Total Heat Delivered")
        fig.add_trace(bar, row=2, col=1)

    fig.update_layout(
        barmode='group',
        title_text="Total Metrics by File",
        showlegend=True
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


# Assumes these helpers/constants are defined elsewhere:
#   - find_matching_columns(df, patterns)
#   - get_column_index(column_name)
#   - COLOR_PALETTE for non‑PCM traces

def create_temperature_plots(dfs, uef_values, patterns=['T_WH', 'T_PCM']):
    """Create temperature plots with static colors plus distinct hues & opacities for each PCM layer & node."""
    all_figs = []
    figure_metadata = []
    water_temp_cutoff = 43.3333  # 110 °F in °C

    # Precompile PCM regex
    pcm_regex_full = re.compile(r"T_PCM(\d+)-(\d+)-(\d+)-(\d+)")

    for i, (file, df) in enumerate(dfs.items()):
        uef = uef_values[i]
        column_groups = find_matching_columns(df, patterns)

        # Title parameters
        pcm_mass = float(df.get("PCM Mass (kg)", pd.Series([0.0])).iloc[-1])
        pcm_h_col = next((c for c in df.columns
                          if re.fullmatch(r"Water Tank PCM\d+ h \(W/m\^2K\)", c)),
                         None)
        pcm_sa_col = next((c for c in df.columns
                           if re.fullmatch(r"Water Tank PCM\d+ sa_ratio", c)),
                          None)
        pcm_h  = float(df[pcm_h_col].iloc[-1]) if pcm_h_col else 0.0
        pcm_sa = float(df[pcm_sa_col].iloc[-1]) if pcm_sa_col else 0.0
        
        name_water_volume, isDefault = parse_tank_volume_from_name(file)
        name_water_volume *= 0.9

        water_volume_gal = df["Water Volume (L)"].iloc[0] * L_TO_GAL_RATIO if "Water Volume (L)" in df else name_water_volume

        # Precompute per‑layer info for T_PCM
        pcm_cols = column_groups.get('T_PCM', [])
        if pcm_cols:
            # unique layers
            layers = sorted({
                int(m.group(1))
                for col in pcm_cols
                if (m := pcm_regex_full.match(col))
            })
            # assign a distinct hue per layer
            palette = px.colors.qualitative.Dark24
            layer_colors = {
                layer: palette[idx % len(palette)]
                for idx, layer in enumerate(layers)
            }
            # find max index along each dimension for each layer
            max_indices = {}
            for layer in layers:
                rad = [
                    int(m.group(2))
                    for col in pcm_cols
                    if (m := pcm_regex_full.match(col)) and int(m.group(1)) == layer
                ]
                cir = [
                    int(m.group(3))
                    for col in pcm_cols
                    if (m := pcm_regex_full.match(col)) and int(m.group(1)) == layer
                ]
                axi = [
                    int(m.group(4))
                    for col in pcm_cols
                    if (m := pcm_regex_full.match(col)) and int(m.group(1)) == layer
                ]
                max_indices[layer] = {
                    'radial':        max(rad) if rad else 1,
                    'circumference': max(cir) if cir else 1,
                    'axial':         max(axi) if axi else 1
                }
            dash_map = {
                'radial':       'dash',
                'circumference':'dot',
                'axial':        'dashdot'
            }

        # One figure per pattern
        for pattern, cols in column_groups.items():
            if not cols:
                continue
            fig = go.Figure()

            # y‑range + padding
            tmin = min(df[c].min() for c in cols)
            tmax = max(df[c].max() for c in cols)
            pad  = (tmax - tmin) * 0.1

            for col in cols:
                if pattern == 'T_PCM' and (m := pcm_regex_full.match(col)):
                    layer, radial, circ, axial = map(int, m.groups())
                    # base hex → RGB
                    hexcol = layer_colors[layer].lstrip('#')
                    r, g, b = int(hexcol[0:2], 16), int(hexcol[2:4], 16), int(hexcol[4:6], 16)
                    # determine which dim varies
                    if   radial > 1 and circ == 1 and axial == 1:
                        dim, idx = 'radial', radial
                    elif circ   > 1 and radial == 1 and axial == 1:
                        dim, idx = 'circumference', circ
                    elif axial  > 1 and radial == 1 and circ == 1:
                        dim, idx = 'axial', axial
                    else:
                        dim, idx = None, 1

                    # opacity from 1.0 (idx=1) down to 0.5 (idx=max)
                    if dim:
                        max_idx = max_indices[layer][dim]
                        opacity = 1 - ((idx - 1) / max(max_idx - 1, 1) * 0.5)
                        dash = dash_map[dim]
                    else:
                        opacity, dash = 1, 'solid'

                    color = f'rgba({r},{g},{b},{opacity:.2f})'
                    line_style = dict(color=color, dash=dash)

                else:
                    idx = get_column_index(col)
                    color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
                    line_style = dict(color=color)

                fig.add_trace(
                    go.Scatter(
                        x=df['Time'],
                        y=df[col],
                        mode='lines',
                        name=col,
                        line=line_style
                    )
                )

            # Overlays (unchanged)…
            if 'Water Heating Mode' in df.columns:
                mode = df['Water Heating Mode'].astype(str)
                times = df['Time']
                def regions(key):
                    R, in_seg = [], False
                    for j, mstr in enumerate(mode):
                        if key in mstr and not in_seg:
                            start, in_seg = j, True
                        elif key not in mstr and in_seg:
                            R.append((times.iloc[start], times.iloc[j]))
                            in_seg = False
                    if in_seg:
                        R.append((times.iloc[start], times.iloc[len(mode)-1]))
                    return R
                def add(R, name, clr, grp):
                    for k, (t0, t1) in enumerate(R):
                        fig.add_trace(go.Scatter(
                            x=[t0, t1, t1, t0, t0],
                            y=[tmin-pad]*2 + [tmax+pad]*2 + [tmin-pad],
                            fill='toself',
                            fillcolor=clr,
                            line=dict(width=0),
                            mode='none',
                            name=name if k==0 else None,
                            showlegend=(k==0),
                            legendgroup=grp,
                            hoverinfo='skip'
                        ))
                add(regions('Upper On'),     "Upper Element On",    "rgba(255,0,0,0.15)", "upper")
                add(regions('Lower On'),     "Lower Element On",    "rgba(0,0,255,0.15)", "lower")
                add(regions('Heat Pump On'), "Heat Pump On",        "rgba(0,255,0,0.15)", "heat_pump")

            # Cutoff line & annotation
            fig.add_shape(
                type='line', xref='paper', x0=0, x1=1,
                yref='y', y0=water_temp_cutoff, y1=water_temp_cutoff,
                line=dict(color='red', width=1, dash='dash')
            )
            fig.add_annotation(
                xref='paper', x=1, y=water_temp_cutoff,
                xanchor='right', yanchor='bottom',
                text='110 °F Cutoff Temp', showarrow=False
            )

            # Layout
            fig.update_layout(
                title=(
                    f"{pattern} Temperatures – {file}<br>"
                    f"UEF: {uef:.3f} | PCM h: {pcm_h:.2f} W/m²K | "
                    f"PCM SA: {pcm_sa:.2f} | PCM Mass: {pcm_mass:.3f} kg | "
                    f"Water Volume: {water_volume_gal:.1f} gal"
                ),
                xaxis_title='Time',
                yaxis_title='Temperature (°C)',
                height=600,
                showlegend=True
            )
            fig.update_yaxes(range=[tmin - pad, tmax + pad])

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
            water_volume_gal, isDefault = parse_tank_volume_from_name(file)
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

def parse_tank_volume_from_name(filename):
    """
    Parses a filename to extract tank volume in gallons.
    Returns a tuple: (volume_in_gal, is_default)
    """
    is_default = "zDefault" in filename
    match = re.search(r'_([0-9]+(?:\.[0-9]+)?)gal_', filename)
    volume = float(match.group(1)) if match else None
    return volume, is_default

def calculate_uef(dfs):
    # calculate UEF of the water tank
    uef_values = []      # make sure in W*min                            
    
    for name, df in dfs.items():
        uef = calculate_single_uef(df, name)
        uef_values.append(uef)
    
    return uef_values

def calculate_single_uef(df, name):
    """
    Calculates two UEF values:
      - 'uef_all': UEF over the entire dataframe
      - 'uef_last_day': UEF using only rows from the last calendar day present in df['Time']
    Returns a dict: {'uef_all': float|np.nan, 'uef_last_day': float|np.nan}
    """
    import pandas as pd
    import numpy as np

    def _parse_time(series: pd.Series) -> pd.Series:
        # Support mixed timestamp formats (with/without fractional seconds).
        return pd.to_datetime(series, errors="coerce", format="mixed")

    def _compute_uef_for_df(local_df: pd.DataFrame) -> float:
        if local_df.empty:
            return np.nan

        local = local_df.copy()
        local['Time'] = _parse_time(local['Time'])
        local = local.dropna(subset=['Time'])
        if local.empty:
            return np.nan
        local = local.sort_values('Time')
        local['dt_s'] = local['Time'].diff().dt.total_seconds().fillna(0)

        # Electric heating energy (kW → W, then multiply by seconds)
        Q_cons = ((local["Water Heating Electric Power (kW)"] * 1000) * local['dt_s']).sum()

        # Hot water delivered energy (already W, multiply by seconds)
        Q_load = (local["Hot Water Delivered (W)"] * local['dt_s']).sum()

        # PCM and water state adjustments (based on your existing helpers)
        PCM_Q_Heat_to_Water = calculate_net_PCM_heat(local)  # kept for parity; not directly used below
        PCM_net_enthalpy = calculate_net_PCM_enthalpy(local)
        PCM_net_heat_loss = PCM_net_enthalpy
        water_net_temp_delta = calculate_net_water_temp(local)

        water_volume_col = "Water Volume (L)"
        if water_volume_col in local.columns:
            water_volume_L = local[water_volume_col].iloc[-1]
        else:
            tank_volume, _is_default_case = parse_tank_volume_from_name(name)
            tank_volume *= 0.9  # effective water fraction
            water_volume_L = tank_volume / L_TO_GAL_RATIO

        # Net energy to change bulk water temperature over the window
        water_net_energy = calculate_net_water_energy(
            water_volume_L,
            local['Hot Water Average Temperature (C)'].iloc[-1],
            water_net_temp_delta
        )

        Q_cons_adjusted = Q_cons - PCM_net_heat_loss - water_net_energy
        if not np.isfinite(Q_cons_adjusted) or Q_cons_adjusted == 0:
            return np.nan

        return Q_load / Q_cons_adjusted

    # ---- Whole-dataset UEF ----
    df_all = df.copy()
    df_all['Time'] = _parse_time(df_all['Time'])
    df_all = df_all.dropna(subset=['Time'])
    df_all = df_all.sort_values('Time')
    uef_all = _compute_uef_for_df(df_all)

    # ---- Last calendar day UEF ----
    if df_all.empty:
        uef_last_day = np.nan
    else:
        last_day = df_all['Time'].max().normalize()  # calendar day of the last timestamp
        df_last_day = df_all[df_all['Time'].dt.normalize() == last_day]
        # Recompute on the sliced day to get correct dt_s within the day
        uef_last_day = _compute_uef_for_df(df_last_day)

    return {"uef_all": uef_all, "uef_last_day": uef_last_day}

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

def plot_pcm_reference_and_deciles_plotly_F(
    pcms,
    pcms_names,
    ref_f=127,
    x_min_f=85,
    x_max_f=155,
    show=True,
):
    if len(pcms) != len(pcms_names):
        raise ValueError("pcms and pcms_names must have the same length")

    # ----------------------------
    # Locate reference PCM
    # ----------------------------
    ref_matches = [i for i, n in enumerate(pcms_names) if str(ref_f) in str(n)]
    if len(ref_matches) != 1:
        raise ValueError(f"Expected exactly one PCM containing '{ref_f}', found {ref_matches}")
    ref_idx = ref_matches[0]

    # ----------------------------
    # Helpers
    # ----------------------------
    def K_to_F(Tk):
        return (Tk - 273.15) * 9.0 / 5.0 + 32.0

    def prepare(arr):
        a = np.asarray(arr)[:, :3]
        a = a[np.isfinite(a).all(axis=1)]
        a = a[np.argsort(a[:, 0])]
        return (
            K_to_F(a[:, 0]),
            a[:, 1] / 1.8,  # Cp -> J/g-F
            a[:, 2],        # enthalpy
        )

    def decile_indices(n):
        step = max(1, int(round(n / 10)))
        idx = np.arange(0, n, step)
        if idx[-1] != n - 1:
            idx = np.append(idx, n - 1)
        return np.unique(idx)

    dec_idx = [i for i in decile_indices(len(pcms)) if i != ref_idx]

    # ----------------------------
    # Common layout (VALID ONLY)
    # ----------------------------
    common_layout = dict(
        template="plotly_white",
        font=dict(size=20),
        margin=dict(l=90, r=40, t=90, b=80),
        legend=dict(font=dict(size=16)),
        xaxis=dict(
            range=[x_min_f, x_max_f],
            title=dict(text="Temperature (°F)", font=dict(size=22)),
            tickfont=dict(size=18),
        ),
    )

    # ----------------------------
    # Enthalpy figure
    # ----------------------------
    fig_h = go.Figure()
    fig_h.update_layout(
        **common_layout,
        title=dict(
            text=f"PCM Enthalpy vs Temperature (Reference {ref_f}F)",
            font=dict(size=28),
        ),
        yaxis=dict(
            title=dict(text="Enthalpy (J/g)", font=dict(size=22)),
            tickfont=dict(size=18),
        ),
    )

    # ----------------------------
    # Cp figure
    # ----------------------------
    fig_cp = go.Figure()
    fig_cp.update_layout(
        **common_layout,
        title=dict(
            text=f"PCM Heat Capacity vs Temperature (Reference {ref_f}F)",
            font=dict(size=28),
        ),
        yaxis=dict(
            title=dict(text="Cp (J/g-°F)", font=dict(size=22)),
            tickfont=dict(size=18),
        ),
    )

    # ----------------------------
    # Plot reference (black)
    # ----------------------------
    T_F, cp_F, h = prepare(pcms[ref_idx])

    fig_h.add_trace(go.Scatter(
        x=T_F,
        y=h,
        mode="lines",
        name=f"{pcms_names[ref_idx]} (reference)",
        line=dict(color="black", width=6),
    ))

    fig_cp.add_trace(go.Scatter(
        x=T_F,
        y=cp_F,
        mode="lines",
        name=f"{pcms_names[ref_idx]} (reference)",
        line=dict(color="black", width=6),
    ))

    # ----------------------------
    # Plot decile curves
    # ----------------------------
    palette = pc.qualitative.Dark24

    for k, i in enumerate(dec_idx):
        T_F, cp_F, h = prepare(pcms[i])
        color = palette[k % len(palette)]

        fig_h.add_trace(go.Scatter(
            x=T_F,
            y=h,
            mode="lines",
            name=pcms_names[i],
            line=dict(color=color, width=3),
            opacity=0.55,
        ))

        fig_cp.add_trace(go.Scatter(
            x=T_F,
            y=cp_F,
            mode="lines",
            name=pcms_names[i],
            line=dict(color=color, width=3),
            opacity=0.55,
        ))

    if show:
        fig_h.show()
        fig_cp.show()

    return fig_h, fig_cp, ref_idx, dec_idx



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


def create_deltaT_over_film_coeff_plots(
    dfs,
    water_regex=r"^T_WH\d+$",
    enamel_regex=r"^T_ENM\d+$",
    film_regex=r"^Film Tank Heat Transfer Coefficient \(W/m\^2-K\)$",
    hot_water_col="Hot Water Delivered (L/min)",
    use_abs_deltaT=True,
    title_prefix="ΔT / h over time"
):
    """
    Iterate over {file: df}. For each df (skipping ones without water/enamel/film HTC):
      - Find water & enamel node columns by regex.
      - Average them to get T_water_avg and T_enamel_avg.
      - ΔT = |T_water_avg - T_enamel_avg| (or signed if use_abs_deltaT=False).
      - Compute ratio R = ΔT / h using film coefficient column (W/m^2-K).
      - Plot R vs Time (left y-axis) and hot water draw (converted to gal/min) on right y-axis.
    """
    all_figs = []
    figure_metadata = []

    water_rx  = re.compile(water_regex)
    enamel_rx = re.compile(enamel_regex)
    film_rx   = re.compile(film_regex)

    for file, df in dfs.items():
        # Ensure Time column present and usable
        if "Time" not in df.columns:
            if "time" in df.columns:
                df = df.rename(columns={"time": "Time"})
            else:
                continue  # skip: no time axis

        dfi = df.copy()
        dfi["Time"] = pd.to_datetime(dfi["Time"], errors="coerce")
        dfi = dfi.dropna(subset=["Time"])

        # Find columns by regex
        water_cols  = [c for c in dfi.columns if water_rx.search(c)]
        enamel_cols = [c for c in dfi.columns if enamel_rx.search(c)]
        film_cols   = [c for c in dfi.columns if film_rx.search(c)]

        # Skip if missing required sets
        if not water_cols or not enamel_cols or not film_cols:
            continue
        film_col = film_cols[0]

        # Compute averages
        water_avg  = dfi[water_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        enamel_avg = dfi[enamel_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

        # ΔT
        deltaT = water_avg - enamel_avg
        if use_abs_deltaT:
            deltaT = deltaT.abs()

        # Film coefficient h and ratio ΔT / h
        h = pd.to_numeric(dfi[film_col], errors="coerce")
        # avoid division by zero or invalid h
        h = h.replace(0, pd.NA)
        ratio = (deltaT / h).dropna()

        if ratio.empty:
            continue

        # Optional hot water draw (gal/min) on secondary axis
        hot_gpm = None
        if hot_water_col in dfi.columns:
            hot_gpm = pd.to_numeric(dfi[hot_water_col], errors="coerce") * L_TO_GAL_RATIO

        # Build figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=dfi.loc[ratio.index, "Time"],
                y=ratio,
                mode="lines",
                name="ΔT / h",
                line=dict(width=2)
            ),
            secondary_y=False
        )

        if hot_gpm is not None:
            valid_hw = hot_gpm.notna() & dfi["Time"].notna()
            if valid_hw.any():
                fig.add_trace(
                    go.Scatter(
                        x=dfi.loc[valid_hw, "Time"],
                        y=hot_gpm.loc[valid_hw],
                        mode="lines",
                        name="Hot Water Draw (gpm)",
                        line=dict(width=1.6, dash="dot")
                    ),
                    secondary_y=True
                )

        # Axes & layout
        fig.update_yaxes(title_text="ΔT / h  [K²·m²/W]", secondary_y=False)
        fig.update_yaxes(title_text="Hot Water (gal/min)", secondary_y=True)
        fig.update_layout(
            title=f"{title_prefix} – {file}",
            xaxis_title="Time",
            height=520,
            showlegend=True
        )

        all_figs.append(fig)
        figure_metadata.append({
            "file": file,
            "type": "deltaT_over_h_with_hotwater",
            "water_regex": water_regex,
            "enamel_regex": enamel_regex,
            "film_regex": film_regex,
            "hot_water_col": hot_water_col
        })

    return all_figs, figure_metadata

def create_deltaT_vs_film_coeff_scatter(
    dfs,
    water_regex=r"^T_WH\d+$",
    enamel_regex=r"^T_ENM\d+$",
    # Per-layer film HTC columns like: "Film Tank T_WH1 Heat Transfer Coefficient (W/m^2-K)"
    film_regex=r"^Film Tank T_WH\d+ Heat Transfer Coefficient \(W/m\^2-K\)$",
    use_abs_deltaT=True,
    title_prefix="ΔT vs Film Heat Transfer Coefficient"
):
    """
    Iterate over {file: df}. For each df (skipping ones without water/enamel/film HTC):
      - Discover *all* per-layer film HTC columns (e.g., 'Film Tank T_WH1 Heat Transfer Coefficient (W/m^2-K)').
      - For each layer N found in a film-HTC column '... T_WHN ...':
          * Prefer ΔT_N = T_WHN - T_ENMN if both per-layer nodes exist.
          * Else fallback to ΔT_avg = (mean water nodes) - (mean enamel nodes).
        (Apply absolute value if use_abs_deltaT=True.)
      - Plot one scatter trace per layer (ΔT on x, h on y) in a single figure per file.
      - Return figures and detailed metadata, including the exact film columns used and the ΔT source (pair vs average).
    """
    all_figs = []
    figure_metadata = []

    water_rx  = re.compile(water_regex)
    enamel_rx = re.compile(enamel_regex)
    film_rx   = re.compile(film_regex)

    # Helper to coerce numeric DataFrame columns
    def _num(df_sub):
        return df_sub.apply(pd.to_numeric, errors="coerce")

    for file, df in dfs.items():
        if "Time" not in df.columns:
            if "time" in df.columns:
                df = df.rename(columns={"time": "Time"})
            else:
                # Time not present; we don't strictly need it for this plot, but keep the original behavior.
                pass

        dfi = df.copy()
        if "Time" in dfi.columns:
            dfi["Time"] = pd.to_datetime(dfi["Time"], errors="coerce")
            dfi = dfi.dropna(subset=["Time"])

        # Find node columns
        water_cols_all  = [c for c in dfi.columns if water_rx.search(c)]
        enamel_cols_all = [c for c in dfi.columns if enamel_rx.search(c)]
        film_cols_all   = [c for c in dfi.columns if film_rx.search(c)]

        # Must have at least one water, one enamel, and at least one film column
        if not water_cols_all or not enamel_cols_all or not film_cols_all:
            continue

        # Pre-compute averages for fallback
        water_avg  = _num(dfi[water_cols_all]).mean(axis=1)
        enamel_avg = _num(dfi[enamel_cols_all]).mean(axis=1)

        # Build the figure with a trace per film layer
        fig = go.Figure()
        trace_count = 0
        per_layer_metadata = []
        used_film_columns = []

        # Regex to pull layer number from a film column: look for 'T_WH(\d+)'
        layer_rx = re.compile(r"T_WH(\d+)")

        for film_col in film_cols_all:
            m = layer_rx.search(film_col)
            if not m:
                # If no layer number found, skip this column
                continue
            layer_id = m.group(1)

            # Try to find matching water/enamel columns for this layer
            water_col_layer = f"T_WH{layer_id}"
            enamel_col_layer = f"T_ENM{layer_id}"

            have_layer_pair = (water_col_layer in dfi.columns) and (enamel_col_layer in dfi.columns)

            if have_layer_pair:
                w = pd.to_numeric(dfi[water_col_layer], errors="coerce")
                e = pd.to_numeric(dfi[enamel_col_layer], errors="coerce")
                deltaT = w - e
                deltaT_source = "layer_pair"
            else:
                # Fallback to averages
                deltaT = water_avg - enamel_avg
                deltaT_source = "avg_fallback"

            if use_abs_deltaT:
                deltaT = deltaT.abs()

            h = pd.to_numeric(dfi[film_col], errors="coerce")

            mask = deltaT.notna() & h.notna() & (h > 0)
            if not mask.any():
                # Nothing valid to plot for this layer
                per_layer_metadata.append({
                    "film_column": film_col,
                    "layer": layer_id,
                    "deltaT_source": deltaT_source,
                    "water_col_used": water_col_layer if have_layer_pair else sorted(water_cols_all),
                    "enamel_col_used": enamel_col_layer if have_layer_pair else sorted(enamel_cols_all),
                    "points_plotted": 0
                })
                continue

            fig.add_trace(
                go.Scatter(
                    x=deltaT[mask],
                    y=h[mask],
                    mode="markers",
                    name=f"Layer {layer_id} (T_WH{layer_id})",
                    marker=dict(size=5, opacity=0.6)
                )
            )
            trace_count += 1
            used_film_columns.append(film_col)
            per_layer_metadata.append({
                "film_column": film_col,
                "layer": layer_id,
                "deltaT_source": deltaT_source,
                "water_col_used": water_col_layer if have_layer_pair else sorted(water_cols_all),
                "enamel_col_used": enamel_col_layer if have_layer_pair else sorted(enamel_cols_all),
                "points_plotted": int(mask.sum())
            })

        # If no valid traces, skip the figure
        if trace_count == 0:
            continue

        # Axes & layout
        fig.update_xaxes(title_text="ΔT [K]")
        fig.update_yaxes(title_text="Film Heat Transfer Coefficient h [W/m²·K]")
        fig.update_layout(
            title=f"{title_prefix} – {file}",
            height=560,
            showlegend=True,
            legend_title_text="Tank Layer"
        )

        all_figs.append(fig)
        figure_metadata.append({
            "file": file,
            "type": "deltaT_vs_h_scatter_per_layer",
            "water_regex": water_regex,
            "enamel_regex": enamel_regex,
            "film_regex": film_regex,
            "used_film_columns": used_film_columns,
            "layer_details": per_layer_metadata
        })

    return all_figs, figure_metadata

def create_deltaT_over_time_plots(
    dfs,
    water_regex=r"^T_WH\d+$",
    enamel_regex=r"^T_ENM\d+$",
    use_abs_deltaT=True,
    title_prefix="ΔT over time"
):
    """
    For each df:
      - Find water & enamel node columns by regex.
      - Average them to get T_water_avg and T_enamel_avg.
      - ΔT = |T_water_avg - T_enamel_avg| (or signed if use_abs_deltaT=False).
      - Plot ΔT vs Time (line plot).
    """
    all_figs = []
    figure_metadata = []

    water_rx  = re.compile(water_regex)
    enamel_rx = re.compile(enamel_regex)

    for file, df in dfs.items():
        # Normalize time
        if "Time" not in df.columns:
            if "time" in df.columns:
                df = df.rename(columns={"time": "Time"})
            else:
                continue
        dfi = df.copy()
        dfi["Time"] = pd.to_datetime(dfi["Time"], errors="coerce")
        dfi = dfi.dropna(subset=["Time"])

        # Find columns
        water_cols  = [c for c in dfi.columns if water_rx.search(c)]
        enamel_cols = [c for c in dfi.columns if enamel_rx.search(c)]
        if not water_cols or not enamel_cols:
            continue

        # Compute ΔT
        water_avg  = dfi[water_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        enamel_avg = dfi[enamel_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        deltaT = water_avg - enamel_avg
        if use_abs_deltaT:
            deltaT = deltaT.abs()

        if deltaT.notna().sum() == 0:
            continue

        # Plot ΔT over time
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dfi["Time"],
                y=deltaT,
                mode="lines",
                name="ΔT",
                line=dict(width=2)
            )
        )
        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="ΔT [K]")
        fig.update_layout(
            title=f"{title_prefix} – {file}",
            height=480,
            showlegend=False
        )

        all_figs.append(fig)
        figure_metadata.append({
            "file": file,
            "type": "deltaT_over_time",
            "water_regex": water_regex,
            "enamel_regex": enamel_regex
        })

    return all_figs, figure_metadata



def create_film_htc_plots(
    dfs,
    # Per-layer columns like: "Film Tank T_WH1 Heat Transfer Coefficient (W/m^2-K)"
    film_regex=r"^Film Tank T_WH\d+ Heat Transfer Coefficient \(W/m\^2-K\)$",
    hot_water_col="Hot Water Delivered (L/min)",
    title_prefix="Film HTC and Hot Water Delivered"
):
    """
    Iterate over {file: df}. For each df:
      - Discover *all* per-layer film HTC columns (e.g., 'Film Tank T_WH1 Heat Transfer Coefficient (W/m^2-K)').
      - Plot one line per layer (Film HTC vs Time) on the left y-axis.
      - Plot hot water delivered (converted to gal/min) vs Time on the right y-axis.
      - Overlay semi-transparent regions where 'Water Heating Mode' contains 'Heat Pump On'.
    Skips files missing either at least one film HTC column or the hot water column.
    Returns (figs, metadata) where metadata records which film columns/layers were used and any overlay spans.
    """
    all_figs = []
    figure_metadata = []

    film_rx = re.compile(film_regex)
    layer_rx = re.compile(r"T_WH(\d+)")  # extract layer number

    for file, df in dfs.items():
        # Ensure Time column present
        if "Time" not in df.columns:
            if "time" in df.columns:
                df = df.rename(columns={"time": "Time"})
            else:
                continue

        dfi = df.copy()
        dfi["Time"] = pd.to_datetime(dfi["Time"], errors="coerce")
        dfi = dfi.dropna(subset=["Time"])

        # Find per-layer film HTC columns
        film_cols = [c for c in dfi.columns if film_rx.search(c)]
        if not film_cols or hot_water_col not in dfi.columns:
            continue

        # Convert series to numeric
        hot_lpm = pd.to_numeric(dfi[hot_water_col], errors="coerce")
        try:
            hot_gpm = hot_lpm * L_TO_GAL_RATIO  # assume defined upstream
        except NameError:
            hot_gpm = hot_lpm * 0.2641720524

        # Build subplot with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        per_layer_metadata = []
        used_film_columns = []
        trace_count = 0

        # Track left-axis range for overlay band sizing
        left_vals_min = None
        left_vals_max = None

        for film_col in film_cols:
            m = layer_rx.search(film_col)
            layer_id = m.group(1) if m else "?"

            h = pd.to_numeric(dfi[film_col], errors="coerce")
            if h.dropna().empty:
                per_layer_metadata.append({
                    "film_column": film_col,
                    "layer": layer_id,
                    "points_plotted": 0
                })
                continue

            # Update left-axis min/max for overlay sizing
            col_min = h.min(skipna=True)
            col_max = h.max(skipna=True)
            if pd.notna(col_min):
                left_vals_min = col_min if left_vals_min is None else min(left_vals_min, col_min)
            if pd.notna(col_max):
                left_vals_max = col_max if left_vals_max is None else max(left_vals_max, col_max)

            fig.add_trace(
                go.Scatter(
                    x=dfi["Time"],
                    y=h,
                    mode="lines",
                    name=f"Layer {int(layer_id):02d} (T_WH{int(layer_id):02d})",
                    line=dict(width=2)
                ),
                secondary_y=False
            )
            trace_count += 1
            used_film_columns.append(film_col)
            per_layer_metadata.append({
                "film_column": film_col,
                "layer": layer_id,
                "points_plotted": int(h.notna().sum())
            })

        # Add hot water draw on secondary y-axis
        if not hot_gpm.dropna().empty:
            fig.add_trace(
                go.Scatter(
                    x=dfi["Time"],
                    y=hot_gpm,
                    mode="lines",
                    name="Hot Water Draw (gpm)",
                    line=dict(width=2, dash="dot")
                ),
                secondary_y=True
            )
        else:
            if trace_count == 0:
                continue

        if trace_count == 0:
            continue

        # Compute padding for overlay polygons on left y-axis
        if left_vals_min is None or left_vals_max is None:
            # Fallback if somehow not set
            left_vals_min, left_vals_max = 0.0, 1.0
        y_pad = (left_vals_max - left_vals_min) * 0.1
        y0_band = left_vals_min - y_pad
        y1_band = left_vals_max + y_pad

        # Overlay: "Heat Pump On" regions (from Water Heating Mode)
        overlay_spans = []
        if "Water Heating Mode" in dfi.columns:
            mode = dfi["Water Heating Mode"].astype(str)
            times = dfi["Time"]

            def heat_pump_regions():
                spans, in_seg = [], False
                start = None
                for j, mstr in enumerate(mode):
                    if ("Heat Pump On" in mstr) and not in_seg:
                        start, in_seg = j, True
                    elif ("Heat Pump On" not in mstr) and in_seg:
                        spans.append((times.iloc[start], times.iloc[j]))
                        in_seg = False
                if in_seg:
                    spans.append((times.iloc[start], times.iloc[len(mode) - 1]))
                return spans

            hp_regions = heat_pump_regions()
            overlay_spans = [(str(t0), str(t1)) for (t0, t1) in hp_regions]

            # Add semi-transparent green bands as filled polygons on left axis
            for k, (t0, t1) in enumerate(hp_regions):
                fig.add_trace(
                    go.Scatter(
                        x=[t0, t1, t1, t0, t0],
                        y=[y0_band, y0_band, y1_band, y1_band, y0_band],
                        fill="toself",
                        fillcolor="rgba(0,255,0,0.15)",
                        line=dict(width=0),
                        mode="none",
                        name="Heat Pump On" if k == 0 else None,
                        showlegend=(k == 0),
                        hoverinfo="skip",
                        legendgroup="heat_pump"
                    ),
                    secondary_y=False
                )

        # Axes & layout
        fig.update_yaxes(title_text="Film HTC (W/m²·K)", secondary_y=False)
        fig.update_yaxes(title_text="Hot Water (gal/min)", secondary_y=True)
        fig.update_layout(
            title=f"{title_prefix} – {file}",
            xaxis_title="Time",
            height=560,
            showlegend=True,
            legend_title_text="Series"
        )

        all_figs.append(fig)
        figure_metadata.append({
            "file": file,
            "type": "film_htc_and_hotwater_per_layer",
            "film_regex": film_regex,
            "used_film_columns": used_film_columns,
            "layer_details": per_layer_metadata,
            "hot_water_col": hot_water_col,
            "heat_pump_on_spans": overlay_spans  # stringified time spans for ease of serialization
        })

    return all_figs, figure_metadata


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



def _label_from_filename(fname):
    m = re.search(r'_([0-9]{2,3})F', fname)
    return f"{m.group(1)}°F" if m else fname

def _label_from_filename(fname):
    m = re.search(r'_([0-9]{2,3})F', fname)
    return f"{m.group(1)}°F" if m else fname

def pcm_enthalpy_integral_bar(profiles, filenames=None, T_low_F=110, T_high_F=125):
    """
    Compute integral of enthalpy over [T_low_F, T_high_F]°F for each PCM profile
    and return a Plotly bar chart plus the raw integrals dict.

    Args:
        profiles (list of np.ndarray): Each array with columns [Temp (°C), cp, Enthalpy (J/kg)].
        filenames (list of str], optional): Used to label bars by extracting _XXXF.
        T_low_F (float): Lower Fahrenheit bound.
        T_high_F (float): Upper Fahrenheit bound.

    Returns:
        fig: Plotly bar figure.
        integrals: dict label -> integral value (J/kg * °C).
    """
    T_low_C = (T_low_F - 32) / 1.8
    T_high_C = (T_high_F - 32) / 1.8

    labels = []
    if filenames:
        labels = [_label_from_filename(f) for f in filenames]
    else:
        labels = [f'PCM {i+1}' for i in range(len(profiles))]
    # pad/truncate
    if len(labels) < len(profiles):
        labels += [f'PCM {i+1}' for i in range(len(labels), len(profiles))]
    labels = labels[: len(profiles)]

    integrals = {}
    values = []
    for label, df in zip(labels, profiles):
        temp_C = df[:, 0]
        enthalpy = df[:, 1]
        mask = (temp_C >= T_low_C) & (temp_C <= T_high_C)
        x_sel = temp_C[mask]
        y_sel = enthalpy[mask]
        if len(x_sel) == 0:
            integral = 0.0
        else:
            if not np.isclose(x_sel[0], T_low_C):
                y_low = np.interp(T_low_C, temp_C, enthalpy)
                x_sel = np.insert(x_sel, 0, T_low_C)
                y_sel = np.insert(y_sel, 0, y_low)
            if not np.isclose(x_sel[-1], T_high_C):
                y_high = np.interp(T_high_C, temp_C, enthalpy)
                x_sel = np.append(x_sel, T_high_C)
                y_sel = np.append(y_sel, y_high)
            integral = np.trapz(y_sel, x_sel)
        integrals[label] = integral
        values.append(integral)

    fig = go.Figure(go.Bar(x=list(integrals.keys()), y=values))
    fig.update_layout(
        title=f'Enthalpy Integral over {T_low_F}–{T_high_F}°F',
        xaxis=dict(title='PCM (from filename or autogenerated)'),
        yaxis=dict(title='∫ Enthalpy dT (J/kg·°C)'),
        height=500
    )
    return fig, integrals

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
    
    _uef_time = time.perf_counter()
    uef_totals = calculate_uef(dfs)
    uef_global = [x['uef_all'] for x in uef_totals]
    uef_last_day = [x['uef_last_day'] for x in uef_totals]
    print(f"UEF calculation time: {time.perf_counter() - _uef_time:.2f} seconds")

    _pool_time = time.perf_counter()
    all_plots = parallel_create_temperature_plots(dfs, uef_values=uef_last_day, patterns=['T_WH', 'T_PCM'])
    print(f"Temp chart processing pool time: {time.perf_counter() - _pool_time:.2f} seconds")
    
    # # # # Display all plots
    _plot_time = time.perf_counter()
    parallel_display_plots(all_plots, stagger_delay=0.1)  # 0.1 second delay between plots
    print(f"Temp chart display pool time: {time.perf_counter() - _plot_time:.2f} seconds")
    
    
    
    # _plot_time = time.perf_counter()
    # film_temp_charts, film_temp_metadata = create_deltaT_over_film_coeff_plots(dfs)
    # print(f"Film coeff plots pool time: {time.perf_counter() - _plot_time:.2f} seconds")
    
    # _plot_time = time.perf_counter()
    # parallel_display_plots(film_temp_charts, stagger_delay=0.1)  # 0.1 second delay between plots
    # print(f"Film coeff plots display pool time: {time.perf_counter() - _plot_time:.2f} seconds")
    
    
    
    # _plot_time = time.perf_counter()
    # film_htc_charts, film_htc_metadata = create_film_htc_plots(dfs)
    # print(f"Film HTC plots pool time: {time.perf_counter() - _plot_time:.2f} seconds")
    
    # _plot_time = time.perf_counter()
    # parallel_display_plots(film_htc_charts, stagger_delay=0.1)  # 0.1 second delay between plots
    # print(f"Film HTC plots display pool time: {time.perf_counter() - _plot_time:.2f} seconds")
    
    
    
    
    # _plot_time = time.perf_counter()
    # film_htc_charts, film_htc_metadata = create_deltaT_vs_film_coeff_scatter(dfs)
    # print(f"Film HTC plots pool time: {time.perf_counter() - _plot_time:.2f} seconds")
    
    # _plot_time = time.perf_counter()
    # parallel_display_plots(film_htc_charts, stagger_delay=0.1)  # 0.1 second delay between plots
    # print(f"Film HTC plots display pool time: {time.perf_counter() - _plot_time:.2f} seconds")
    
    
    
    # _plot_time = time.perf_counter()
    # dt_charts, dt_metadata = create_deltaT_over_time_plots(dfs)
    # print(f"ΔT over time plots pool time: {time.perf_counter() - _plot_time:.2f} seconds")
    
    # _plot_time = time.perf_counter()
    # parallel_display_plots(dt_charts, stagger_delay=0.1)  # 0.1 second delay between plots
    # print(f"ΔT over time plots display pool time: {time.perf_counter() - _plot_time:.2f} seconds")
    
    
    
    
    # _plot_time = time.perf_counter()
    # all_plots = parallel_create_energy_output_plots(dfs, uef_values=uef_last_day, patterns=['T_WH', 'T_PCM'])
    # print(f"Energy output processing pool time: {time.perf_counter() - _plot_time:.2f} seconds")
    
    # _plot_time = time.perf_counter()
    # parallel_display_plots(all_plots, stagger_delay=0.1)  # 0.1 second delay between plots
    # print(f"Energy output display pool time: {time.perf_counter() - _plot_time:.2f} seconds")



    # Create water flow and temperature plots
    # _pool_time = time.perf_counter()
    # figures, figure_metadata = create_water_flow_temperature_plots(dfs, uef_values=uef_totals, outlet_gpm=3)
    # print(f"Water flow and temperature processing pool time: {time.perf_counter() - _pool_time:.2f} seconds")
    # for fig in figures:
    #     fig.show()

    # # Draw data summary
    _hot_water_delivered_pool_time = time.perf_counter()
    output = calculate_hot_water_delivered(dfs, first_hour_test=True)
    print(f"Hot water delivered pool time: {time.perf_counter() - _hot_water_delivered_pool_time:.2f} seconds")
    
    # _hot_water_plot_time = time.perf_counter()
    # plot_draw_event_summary(output)
    # print(f"Hot water plot time: {time.perf_counter() - _hot_water_plot_time:.2f} seconds")
    
    # csv_path = export_draw_outputs_csv(output, "../OCHRE_results/results_csv/results_no_FHR_ADJUSTMENT.csv")
    csv_path = export_draw_outputs_csv(output, "../OCHRE_results/results_csv/results_FHR_ADJUSTMENT_PCM_PARMETRIC_LOW_CONDUCTIVITY.csv")
    
    # plot_draw_events(output)
    # plot_totals(output)
    
    # draw 2d matrix plot
    # plot_comparison(dfs, output)
    
    pcms = []
    # pcms = [np.loadtxt(os.path.join(os.path.dirname(__file__), "..", "ochre", "defaults", "pcm_configs", f"100%_ct53-resin_h-T_data_88frac_{i}F.csv"), delimiter=",", skiprows=1) for i in range(110, 142 + 1, 1)]
    # pcms = [np.loadtxt(os.path.join(os.path.dirname(__file__), "..", "ochre", "defaults", "pcm_configs", "cp_h-T_data_shifted_120F.csv"), delimiter=",", skiprows=1)]
    # pcms.append(np.loadtxt(os.path.join(os.path.dirname(__file__), "..", "ochre", "defaults", "pcm_configs", "90-cp_h-T_data_shifted_120F.csv"), delimiter=",", skiprows=1))
    # pcms.append(np.loadtxt(os.path.join(os.path.dirname(__file__), "..", "ochre", "defaults", "pcm_configs", "60-40_PCM55-TPU_cp-h-T_data_shifted_120F.csv"), delimiter=",", skiprows=1))
    
    # pcms_names = [f'PCM {i}F' for i in range(110, 142 + 1, 1)]
    # # pcms_names = ['90% Graphite infiltrated PCM'] 
    
    # _ = plot_pcm_reference_and_deciles_plotly_F(pcms, pcms_names)
    
    # fig, integral = pcm_enthalpy_integral_bar(pcms, pcms_names)
    # fig.show()
    
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
