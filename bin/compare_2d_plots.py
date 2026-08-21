import argparse
import os
import time
from pickle import TRUE
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import griddata
from pathlib import Path
import re
import numpy as np

try:
    # Package/module execution: ``python -m bin...``
    from .calculate_hot_water_delivered import calculate_hot_water_delivered
except ImportError:
    # Direct script execution: ``python bin/compare_2d_plots.py``
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

def plot_draw_event_summary(draw_outputs):
    """
    Plot a grouped bar chart showing the overall totals for each file.
    Total water delivered (gallons) and total heat delivered (kWh) are compared side by side,
    with a horizontal line showing the max possible hot water volume as a reference.
    """
    files = list(draw_outputs.keys())
    total_volumes = [draw_outputs[file]['total_water_volume_gal'] for file in files]
    total_energy = [draw_outputs[file]['total_energy_used_kwh'] for file in files]
    # Assume max_possible_hot_water is a single constant value (same for all files)
    max_volume = list(draw_outputs.values())[0]['max_possible_hot_water']

    # Create the grouped bar chart
    fig = go.Figure(data=[
        go.Bar(
            name='Water Volume (gal)', 
            x=files, 
            y=total_volumes,
            text=[f"{vol:.2f}" for vol in total_volumes],
            textposition='auto'
        ),
        go.Bar(
            name='Energy Used (kWh)', 
            x=files, 
            y=total_energy,
            text=[f"{energy:.3f}" for energy in total_energy],
            textposition='auto'
        )
    ])

    # Add a horizontal line at the max_volume value
    fig.add_shape(
        type="line",
        x0=-0.5,  # starting slightly before the first bar
        x1=len(files)-0.5,  # ending slightly after the last bar
        y0=max_volume,
        y1=max_volume,
        line=dict(color="Red", dash="dash")
    )

    # Optionally add an annotation for the max_volume line
    fig.add_annotation(
        x=len(files)-1,
        y=max_volume,
        xref="x",
        yref="y",
        text=f"Max Volume ({max_volume:.2f} gal)",
        showarrow=True,
        arrowhead=7,
        ax=0,
        ay=-40
    )

    # Update layout with titles and grouped bar mode
    fig.update_layout(
        barmode='group',
        title='Total Hot Water Delivered Summary by File',
        xaxis_title='File',
        yaxis_title='Value'
    )
    fig.show()

def plot_draw_events(draw_outputs):
    """
    For each file, create a separate grouped bar chart for the individual draw events.
    Each event is compared by its water volume (gal) and heat delivered (kWh).
    """
    # Determine the maximum number of draw events among all files.
    max_events = max(len(metrics['draw_events']) for metrics in draw_outputs.values())
    event_numbers = [f"Event {i+1}" for i in range(max_events)]
    
    # Build data dictionaries for water volume and heat delivered per file
    water_data = {}  # key: file, value: list of water volumes per event (or None if missing)
    heat_data = {}   # key: file, value: list of heat delivered per event (or None if missing)
    
    for file, metrics in draw_outputs.items():
        events = metrics['draw_events']
        water_values = []
        heat_values = []
        for i in range(max_events):
            if i < len(events):
                water_values.append(round(events[i]['water_volume_gal'], 2))
                heat_values.append(round(events[i]['heat_delivered_kWh'], 3))
            else:
                water_values.append(None)
                heat_values.append(None)
        water_data[file] = water_values
        heat_data[file] = heat_values

    # Create subplots: 1 row, 2 columns for the two metrics
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=("Water Volume (gal)", "Heat Delivered (kWh)"),
                        shared_xaxes=True)
    
    # For each file, add a bar trace for water volume in subplot 1
    for file, values in water_data.items():
        fig.add_trace(
            go.Bar(
                name=file,
                x=event_numbers,
                y=values,
                text=[f"{v:.2f}" if v is not None else "" for v in values],
                textposition='auto'
            ),
            row=1, col=1
        )
        
    # For each file, add a bar trace for heat delivered in subplot 2
    for file, values in heat_data.items():
        fig.add_trace(
            go.Bar(
                name=file,
                x=event_numbers,
                y=values,
                text=[f"{v:.3f}" if v is not None else "" for v in values],
                textposition='auto'
            ),
            row=2, col=1
        )
    
    # Update the layout for grouped bars and overall titles
    fig.update_layout(
        barmode='group',
        title_text="Draw Events Grouped by Event Number Across Files",
        xaxis_title="Draw Event"
    )
    
    fig.show()


def create_interpolated_plot(
    df,
    z_column,
    x_mesh,
    y_mesh,
    x_grid,
    y_grid,
    title,
    z_label,
    baseline_value=None,
    baseline_description=None,
):
    import numpy as np
    import plotly.graph_objects as go
    from scipy.interpolate import griddata

    if df.empty or df[z_column].isna().all():
        print(f"No valid data for {z_column}")
        return None

    little_font_size = 16
    medium_font_size = 18

    # Points for interpolation
    points = df[['avg_x_value', 'avg_y_value']].values
    values = df[z_column].values

    # Perform interpolation
    grid_z = griddata(points, values, (x_mesh, y_mesh), method='linear')

    # Get min and max values for color scale normalization
    z_min = float(np.nanmin(values))
    z_max = float(np.nanmax(values))

    # Always use the full data range for global rounding
    z_min_rounded = float(np.floor(z_min))
    z_max_rounded = float(np.ceil(z_max))

    # Create figure
    fig = go.Figure()

    # Define whether higher values are better or worse based on metric
    higher_is_better = z_column in ['total_heat_delivered_kWh', 'total_gal_hot_water_delivered', 'total_energy_used', 'first_draw_hot_water_delivered', 'first_draw_hot_water_delivered_liters', 'total_water_FHR_volume_L']

    # Color scales
    vibrant_colorscale = [
        [0.0, 'rgb(68, 1, 84)'],
        [0.1, 'rgb(72, 40, 120)'],
        [0.2, 'rgb(62, 74, 137)'],
        [0.3, 'rgb(49, 104, 142)'],
        [0.4, 'rgb(38, 130, 142)'],
        [0.5, 'rgb(31, 158, 137)'],
        [0.6, 'rgb(53, 183, 121)'],
        [0.7, 'rgb(109, 205, 89)'],
        [0.8, 'rgb(180, 222, 44)'],
        [0.9, 'rgb(223, 205, 35)'],
        [1.0, 'rgb(253, 231, 37)']
    ]
    desaturated_colorscale = [
        [0.0, 'rgb(120, 120, 120)'],
        [0.2, 'rgb(140, 140, 140)'],
        [0.4, 'rgb(160, 160, 160)'],
        [0.6, 'rgb(180, 180, 180)'],
        [0.8, 'rgb(200, 200, 200)'],
        [1.0, 'rgb(220, 220, 220)']
    ]

    # Helper to build ticks
    def make_ticks(vmin, vmax, n=10):
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            return [vmin]
        ticks = np.linspace(vmin, vmax, n)
        return np.round(ticks, 1).tolist()

    if baseline_value is not None:
        # Determine where grid has better-than-baseline values
        if higher_is_better:
            mask_better = grid_z > baseline_value
            mask_worse = grid_z <= baseline_value
        else:
            mask_better = grid_z < baseline_value
            mask_worse = grid_z >= baseline_value

        # Split grids
        grid_z_better = np.where(mask_better, grid_z, np.nan)
        grid_z_worse  = np.where(mask_worse,  grid_z, np.nan)

        has_better_values = np.any(~np.isnan(grid_z_better))
        has_worse_values  = np.any(~np.isnan(grid_z_worse))

        # CLAMP RULE:
        # If there are values above/better than the baseline, clamp the minimum colored value to the baseline.
        # That means the vibrant (better) trace color range starts at the baseline (or ends at baseline for lower-is-better).
        if has_better_values:
            if higher_is_better:
                vibrant_zmin = float(baseline_value)
                vibrant_zmax = z_max_rounded
                desat_zmin   = z_min_rounded
                desat_zmax   = float(baseline_value)  # cap desaturated at baseline
            else:
                vibrant_zmin = z_min_rounded
                vibrant_zmax = float(baseline_value)
                desat_zmin   = float(baseline_value)  # cap desaturated at baseline
                desat_zmax   = z_max_rounded
        else:
            # No better values; fall back to full range for desaturated-only view
            vibrant_zmin = z_min_rounded
            vibrant_zmax = z_max_rounded
            desat_zmin   = z_min_rounded
            desat_zmax   = z_max_rounded

        # Colorbar ticks:
        if has_better_values:
            # Show only vibrant colorbar; ticks anchored to the clamped vibrant range
            tick_values_vibrant = make_ticks(vibrant_zmin, vibrant_zmax, 10)
            tick_values_worse   = make_ticks(desat_zmin, desat_zmax, 10)
        else:
            # No better region; use full range (desaturated only)
            tick_values_vibrant = make_ticks(z_min_rounded, z_max_rounded, 10)
            tick_values_worse   = make_ticks(z_min_rounded, z_max_rounded, 10)

        # Worse-than-baseline (desaturated)
        contour_worse = go.Contour(
            z=grid_z_worse,
            x=x_grid,
            y=y_grid,
            colorscale=desaturated_colorscale,
            colorbar=dict(
                title=z_label,
                ticks="outside",
                tickfont=dict(size=little_font_size),
                len=0.75,
                tickvals=tick_values_worse,
                ticktext=[f"{val:.1f}" for val in tick_values_worse]
            ),
            ncontours=20,
            contours=dict(showlabels=True, labelfont=dict(size=little_font_size, color='white')),
            line=dict(width=0.5, smoothing=0.85),
            zmin=desat_zmin,
            zmax=desat_zmax,
            zauto=False,
            showscale=(not has_better_values)  # only show this colorbar if no better region exists
        )
        fig.add_trace(contour_worse)

        # Better-than-baseline (vibrant)
        contour_better = go.Contour(
            z=grid_z_better,
            x=x_grid,
            y=y_grid,
            colorscale=vibrant_colorscale,
            colorbar=dict(
                title=z_label,
                ticks="outside",
                tickfont=dict(size=little_font_size),
                len=0.75,
                tickvals=tick_values_vibrant,
                ticktext=[f"{val:.1f}" for val in tick_values_vibrant],
                tickmode='array'
            ),
            ncontours=20,
            contours=dict(showlabels=True, labelfont=dict(size=little_font_size, color='white')),
            line=dict(width=0.5, smoothing=0.85),
            zmin=vibrant_zmin,
            zmax=vibrant_zmax,
            zauto=False
        )
        fig.add_trace(contour_better)

    else:
        # No baseline: single vibrant contour over full range
        tick_values = make_ticks(z_min_rounded, z_max_rounded, 10)
        contour = go.Contour(
            z=grid_z,
            x=x_grid,
            y=y_grid,
            colorscale=vibrant_colorscale,
            colorbar=dict(
                title=z_label,
                ticks="outside",
                tickfont=dict(size=little_font_size),
                len=0.75,
                tickvals=tick_values,
                ticktext=[f"{val:.1f}" for val in tick_values]
            ),
            ncontours=20,
            contours=dict(showlabels=True, labelfont=dict(size=little_font_size, color='white')),
            line=dict(width=0.5, smoothing=0.85),
            zmin=z_min_rounded,
            zmax=z_max_rounded,
            zauto=False
        )
        fig.add_trace(contour)

    # Scatter points
    def clamp01(v):
        return max(0.0, min(1.0, float(v)))

    for _, row in df.iterrows():
        if baseline_value is not None:
            is_better = (higher_is_better and row[z_column] > baseline_value) or (not higher_is_better and row[z_column] < baseline_value)
            if is_better:
                # Normalize within vibrant (clamped) range
                denom = (vibrant_zmax - vibrant_zmin) if (vibrant_zmax - vibrant_zmin) != 0 else 1.0
                norm_val = clamp01((row[z_column] - vibrant_zmin) / denom)
                palette = vibrant_colorscale
            else:
                # Normalize within desaturated (capped) range
                denom = (desat_zmax - desat_zmin) if (desat_zmax - desat_zmin) != 0 else 1.0
                norm_val = clamp01((row[z_column] - desat_zmin) / denom)
                palette = desaturated_colorscale
        else:
            # No baseline: normalize across full range
            denom = (z_max_rounded - z_min_rounded) if (z_max_rounded - z_min_rounded) != 0 else 1.0
            norm_val = clamp01((row[z_column] - z_min_rounded) / denom)
            palette = vibrant_colorscale

        color_idx = int(round(norm_val * (len(palette) - 1)))
        color_idx = max(0, min(len(palette) - 1, color_idx))
        color = palette[color_idx][1]

        hover_text = f"File: {row['file']}<br>{z_label}: {row[z_column]:.2f}"
        if baseline_value is not None:
            diff = row[z_column] - baseline_value
            diff_pct = (row[z_column] / baseline_value - 1) * 100 if baseline_value != 0 else np.nan
            comparison = "better" if ((higher_is_better and diff > 0) or ((not higher_is_better) and diff < 0)) else "worse"
            hover_text += f"<br>Compared to baseline: {diff:.2f} ({diff_pct:.1f}%), {comparison}"

        fig.add_trace(go.Scatter(
            x=[row['avg_x_value']],
            y=[row['avg_y_value']],
            mode='markers',
            marker=dict(size=10, color=color, line=dict(width=1, color='black')),
            text=[hover_text],
            hoverinfo='text',
            showlegend=False
        ))

    # Title / axes
    if baseline_value is not None:
        if baseline_description:
            baseline_text = f"{baseline_description} ({baseline_value:.2f} L)"
        else:
            baseline_text = f"{baseline_value:.2f} {z_label.split('(')[0].strip()}"
        title_with_baseline = f"{title}<br>Baseline: {baseline_text}"
    else:
        title_with_baseline = title
    fig.update_layout(
        title=title_with_baseline,
        title_font=dict(size=18),
        xaxis_title="SA/V Ratio",
        yaxis_title="h_value (W/m²K)",
        xaxis=dict(tickfont=dict(size=20), title_font=dict(size=22)),
        yaxis=dict(tickfont=dict(size=20), title_font=dict(size=22)),
        height=600,
        width=800
    )

    # Baseline status banner
    # if baseline_value is not None:
    #     all_worse = all(((not higher_is_better) and val > baseline_value) or (higher_is_better and val < baseline_value) for val in values)
    #     all_better = all((higher_is_better and val > baseline_value) or ((not higher_is_better) and val < baseline_value) for val in values)

    #     if all_worse:
    #         status_msg, status_color = "⚠️ All values WORSE than baseline", "red"
    #     elif all_better:
    #         status_msg, status_color = "✓ All values BETTER than baseline", "green"
    #     else:
    #         status_msg, status_color = "Grey = worse than baseline; Color = ≥ baseline", "gray"

    #     fig.add_annotation(
    #         text=status_msg,
    #         xref="paper", yref="paper",
    #         x=0.01, y=0.99,
    #         showarrow=False,
    #         align="left",
    #         font=dict(size=8, color=status_color, family="Arial Black"),
    #         bordercolor=status_color,
    #         borderwidth=2,
    #         bgcolor="white",
    #         opacity=0.9
    #     )

    return fig




def plot_2d_comparison(dfs, draw_outputs, setpoint, pcm_temp, tank_type, tank_size):
    """
    Create scatter plots comparing average h_value (W/m^2K) and average sa_ratio 
    for each file with special highlighting for the baseline case.
    
    Parameters:
    - dfs: dict
        Dictionary of dataframes keyed by file name. Each dataframe has columns with names like:
        "Water Tank PCM<number> h (W/m^2K)" and "Water Tank PCM<number> sa_ratio".
    - draw_outputs: dict
        Dictionary containing hot water energy metrics for each file. It is assumed that:
          - draw_outputs[file]['draw_events'] is a list of events, each with key 'heat_delivered_kWh'.
          - draw_outputs[file]['energy_used_kWh'] exists for the energy used metric.
    """
    import re
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from scipy.interpolate import griddata
    
    data = []
    baseline_file = None
    baseline_value = None
    

    for file, df in dfs.items():
        # Check if this is a baseline file (without PCM in column names)
        has_pcm = any('PCM' in col for col in df.columns)
        is_baseline = not has_pcm
        
        # Extract values from columns matching the h_value and sa_ratio patterns
        h_values = []
        sa_ratios = []
        
        # For baseline, look for columns without PCM
        if is_baseline:
            baseline_file = file
            # Extract values from non-PCM columns
            for col in df.columns:
                h_match = re.search(r"Water Tank\s*h\s*\(W/m\^?2K\)", col)
                if h_match:
                    h_values.append(df[col].mean())
                
                sa_match = re.search(r"Water Tank\s*sa_ratio", col)
                if sa_match:
                    sa_ratios.append(df[col].mean())
        else:
            # Normal PCM extraction as before
            for col in df.columns:
                h_match = re.search(r"Water Tank PCM(\d+)\s*h\s*\(W/m\^?2K\)", col)
                if h_match:
                    h_values.append(df[col].mean())
                
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
        total_delivered = draw_outputs[file].get('total_heat_delivered_kWh', None)
        total_used = draw_outputs[file].get('total_energy_used_kwh', None)
        
        data.append({
            'file': file,
            'avg_h_value': avg_h,
            'avg_sa_ratio': avg_sa,
            'total_gal_hot_water_delivered': total_gal_hot_water_delivered,
            'total_heat_delivered_kWh': total_delivered,
            'total_energy_used': total_used,
            'is_baseline': is_baseline
        })

    # Create a DataFrame from the collected data
    df_plot = pd.DataFrame(data)

    # Drop any rows with missing values for plotting
    df_plot_clean = df_plot.dropna(subset=['avg_h_value', 'avg_sa_ratio'])
    
    # Get baseline value and then filter out baseline from plotting data
    if baseline_file:
        baseline_row = df_plot_clean[df_plot_clean['is_baseline'] == True]
        if not baseline_row.empty:
            baseline_value = baseline_row['total_gal_hot_water_delivered'].values[0]
    
    # Filter out baseline case from plotting data
    df_plot_clean = df_plot_clean[df_plot_clean['is_baseline'] == False]
    
    if df_plot_clean.empty:
        print("No non-baseline data available for plotting")
        return None

    # Create interpolation grid
    grid_resolution = 100
    x_min, x_max = df_plot_clean['avg_sa_ratio'].min(), df_plot_clean['avg_sa_ratio'].max()
    y_min, y_max = df_plot_clean['avg_h_value'].min(), df_plot_clean['avg_h_value'].max()

    # Add a small buffer to avoid edge issues
    x_buffer = (x_max - x_min) * 0.05
    y_buffer = (y_max - y_min) * 0.05

    x_grid = np.linspace(x_min - x_buffer, x_max + x_buffer, grid_resolution)
    y_grid = np.linspace(y_min - y_buffer, y_max + y_buffer, grid_resolution)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

    # Function to create interpolated plot with baseline reference
    
    
    # Create plots with baseline references (all three plots as in original)
    # fig_delivered = create_interpolated_plot(
    #     df_plot_clean.dropna(subset=['total_heat_delivered_kWh']),
    #     'total_heat_delivered_kWh',
    #     "h_value vs SA_ratio with Hot Water Delivered Energy pcm_125F",
    #     "Total Hot Water Delivered (kWh)",
    #     baseline_value
    # )

    # fig_used = create_interpolated_plot(
    #     df_plot_clean.dropna(subset=['total_energy_used']),
    #     'total_energy_used',
    #     "h_value vs SA_ratio with Energy Used pcm_125F",
    #     "Total Energy Used (kWh)",
    #     baseline_value
    # )
    
    fig_total_water = create_interpolated_plot(
        df_plot_clean.dropna(subset=['total_gal_hot_water_delivered']),
        'total_gal_hot_water_delivered', x_mesh, y_mesh, x_grid, y_grid,
        f"h vs SA_ratio Design Matrix with Cut off Temp 110°F for {tank_type} Water Heater<br>Setpoint: {setpoint}°F PCM Melt Temp: {pcm_temp}°F <br>Tank Size: {tank_size} gal",
        "FHR (>110°F)<br>Delivered (gal)",
        baseline_value
    )

    # Show the plots
    # if fig_delivered:
    #     fig_delivered.show()
    # if fig_used:
    #     fig_used.show()
    if fig_total_water:
        fig_total_water.show()
    
    return {
        "baseline_value": baseline_value,
        "baseline_file": baseline_file,
        "figures": {
            # "delivered": fig_delivered,
            # "used": fig_used,
            "water": fig_total_water
        }
    }
    
    
def plot_2d_comparison_generic(
    dfs,
    draw_outputs,
    x_column_pattern,
    y_column_pattern,
    setpoint=None,
    pcm_temp=None,
    tank_type=None,
    tank_size=None,
    baseline_file=None,
    case_title=None,
    baseline_title=None,
    show=True,
    save_folder=None,
    image_scale=4,
):
    """Create the first-hour and initial-draw 2-D comparison plots.

    ``dfs`` contains the case files and one explicitly selected baseline file.
    The old folder-based callers can still omit ``baseline_file``; in that
    case the historical no-PCM-column fallback is retained.
    """
    import logging
    import traceback

    logger = logging.getLogger("plot_2d_comparison_generic")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    if not isinstance(dfs, dict) or not dfs:
        logger.error("`dfs` must be a non-empty dict of {filename: DataFrame}.")
        return None
    if not isinstance(draw_outputs, dict) or not draw_outputs:
        logger.warning("`draw_outputs` is empty or not a dict; energy metrics may be None.")

    baseline_key = Path(baseline_file).name if baseline_file else None
    data = []
    detected_baseline_file = None

    for file, df in dfs.items():
        if df is None or not hasattr(df, "columns"):
            logger.error(f"[{file}] df is not a DataFrame-like object.")
            continue

        file_key = Path(file).name
        has_pcm = any("PCM" in str(col) for col in df.columns)
        is_baseline = file_key == baseline_key if baseline_key else not has_pcm
        if is_baseline:
            detected_baseline_file = file_key

        x_values, y_values = [], []
        try:
            if is_baseline:
                for col in df.columns:
                    # Keep baseline extraction restricted to non-PCM tank columns.
                    if "Water Tank" not in col or "PCM" in col:
                        continue
                    if x_column_pattern in col:
                        x_values.append(float(df[col].astype(float).mean()))
                    if y_column_pattern in col:
                        y_values.append(float(df[col].astype(float).mean()))
            else:
                for col in df.columns:
                    if not re.search(r"Water Tank PCM\d+", col):
                        continue
                    if x_column_pattern in col:
                        x_values.append(float(df[col].astype(float).mean()))
                    if y_column_pattern in col:
                        y_values.append(float(df[col].astype(float).mean()))
        except Exception:
            logger.error(f"[{file}] Error while extracting X/Y:\n{traceback.format_exc()}")

        if x_values and y_values:
            avg_x = float(np.nanmean(x_values))
            avg_y = float(np.nanmean(y_values))
        else:
            logger.warning(
                f"[{file}] Missing values: x_values={len(x_values)}, "
                f"y_values={len(y_values)}. Setting averages to 0."
            )
            avg_x, avg_y = 0.0, 0.0

        metrics = draw_outputs.get(file, draw_outputs.get(file_key, {})) or {}
        draw_events = metrics.get("draw_events")
        first_draw_hot_water_delivered = (
            draw_events[0].get("water_volume_L")
            if isinstance(draw_events, list) and draw_events
            else None
        )

        data.append(
            {
                "file": file_key,
                "avg_x_value": avg_x,
                "avg_y_value": avg_y,
                "total_gal_hot_water_delivered": metrics.get("total_water_delivered_volume_gal"),
                "total_liters_hot_water_delivered": metrics.get("total_water_delivered_volume_liters"),
                "total_water_FHR_volume_L": metrics.get("total_water_FHR_volume_L"),
                "total_water_FHR_volume_gal": metrics.get("total_water_FHR_volume_gal"),
                "total_heat_delivered_kWh": metrics.get("total_heat_delivered_kWh"),
                "total_energy_used": metrics.get("total_energy_used_kwh"),
                "is_baseline": is_baseline,
                "first_draw_hot_water_delivered": first_draw_hot_water_delivered,
                "average_pcm_temp": metrics.get("average_pcm_temp"),
            }
        )

    df_plot = pd.DataFrame(data)
    df_plot_clean = df_plot.dropna(subset=["avg_x_value", "avg_y_value"])

    baseline_value_fhr = None
    baseline_value_initial_draw = None
    if detected_baseline_file:
        baseline_rows = df_plot_clean[df_plot_clean["is_baseline"]]
        if not baseline_rows.empty:
            baseline_fhr = baseline_rows["total_water_FHR_volume_L"].iloc[0]
            baseline_initial_draw = baseline_rows["first_draw_hot_water_delivered"].iloc[0]
            baseline_value_fhr = baseline_fhr if pd.notna(baseline_fhr) else None
            baseline_value_initial_draw = (
                baseline_initial_draw if pd.notna(baseline_initial_draw) else None
            )

    case_rows = df_plot_clean[~df_plot_clean["is_baseline"]]
    if case_rows.empty:
        logger.error("No non-baseline data available for plotting.")
        return None

    grid_resolution = 100
    try:
        x_min, x_max = case_rows["avg_x_value"].min(), case_rows["avg_x_value"].max()
        y_min, y_max = case_rows["avg_y_value"].min(), case_rows["avg_y_value"].max()

        x_span = float(x_max - x_min)
        y_span = float(y_max - y_min)
        if x_span == 0:
            logger.warning("x_span is zero; expanding artificially by ±1.")
            x_min, x_max, x_span = x_min - 1.0, x_max + 1.0, 2.0
        if y_span == 0:
            logger.warning("y_span is zero; expanding artificially by ±1.")
            y_min, y_max, y_span = y_min - 1.0, y_max + 1.0, 2.0

        x_buffer = max(x_span * 0.05, 1e-9)
        y_buffer = max(y_span * 0.05, 1e-9)
        x_grid = np.linspace(x_min - x_buffer, x_max + x_buffer, grid_resolution)
        y_grid = np.linspace(y_min - y_buffer, y_max + y_buffer, grid_resolution)
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    except Exception:
        logger.error(f"Error building interpolation grid:\n{traceback.format_exc()}")
        return None

    case_files = case_rows["file"].tolist()
    if case_title is None:
        case_title = format_case_title(case_files[0])

    # Use metadata from the file name when the caller did not provide legacy
    # folder metadata. This removes the old setpoint/PCM/tank filters while
    # keeping the title useful for direct programmatic calls.
    case_metadata = parse_case_filename(case_files[0])
    setpoint = setpoint if setpoint is not None else case_metadata.get("setpoint")
    tank_size = tank_size if tank_size is not None else case_metadata.get("tank_size")

    title_details = ["Cutoff=110°F"]
    if setpoint is not None:
        title_details.append(f"Setpoint={format_number(setpoint)}°F")
    if pcm_temp is not None:
        title_details.append(f"PCM Melt={format_number(pcm_temp)}°F")
    if tank_size is not None:
        title_details.append(f"Tank Size={format_number(tank_size)} gal")
    title_suffix = "; ".join(title_details)

    def safe_plot(df_in, z_col, title, z_label, baseline_value):
        """Call ``create_interpolated_plot`` without allowing one bad metric to stop the run."""
        try:
            if df_in is None or df_in.empty:
                logger.warning(f"Skip plot for {z_col}: input df is empty.")
                return None
            fig = create_interpolated_plot(
                df_in.dropna(subset=[z_col]),
                z_col,
                x_mesh,
                y_mesh,
                x_grid,
                y_grid,
                title,
                z_label,
                baseline_value,
                baseline_title,
            )
            if fig is None:
                logger.warning(f"create_interpolated_plot returned None for {z_col}.")
            return fig
        except Exception:
            logger.error(f"Exception plotting {z_col}:\n{traceback.format_exc()}")
            return None

    title_total = (
        f"{case_title} First-Hour Rating (FHR)<br>{y_column_pattern} vs SA/V Ratio {title_suffix}"
    )
    fig_total_water = safe_plot(
        case_rows.dropna(subset=["total_water_FHR_volume_L"]),
        "total_water_FHR_volume_L",
        title_total,
        "FHR (>110°F) (L)",
        baseline_value_fhr,
    )

    title_first = (
        f"{case_title} Initial Draw Rating<br>{y_column_pattern} vs SA/V Ratio {title_suffix}"
    )
    fig_firstdraw_water = safe_plot(
        case_rows.dropna(subset=["first_draw_hot_water_delivered"]),
        "first_draw_hot_water_delivered",
        title_first,
        "Hot Water (>110°F)<br>Delivered (L)",
        baseline_value_initial_draw,
    )

    saved_files = {}
    if save_folder is not None:
        saved_files = save_comparison_figures(
            {"water": fig_total_water, "first_draw": fig_firstdraw_water},
            case_title,
            save_folder,
            scale=image_scale,
            case_setpoint=setpoint,
            baseline_filename=baseline_file,
        )

    if show:
        try:
            if fig_total_water:
                fig_total_water.show()
            if fig_firstdraw_water:
                fig_firstdraw_water.show()
        except Exception:
            logger.error(f"Error showing figures:\n{traceback.format_exc()}")

    return {
        "baseline_value": baseline_value_fhr,
        "baseline_file": detected_baseline_file,
        "figures": {"water": fig_total_water, "first_draw": fig_firstdraw_water},
        "saved_files": saved_files,
        "debug": {
            "rows_total": int(df_plot.shape[0]),
            "rows_clean": int(case_rows.shape[0]),
            "x_range": (float(x_min), float(x_max)),
            "y_range": (float(y_min), float(y_max)),
        },
    }

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

def create_temperature_plots(dfs, uef_values, patterns=['T_WH', 'T_PCM']):
    """Create separate temperature plots for each pattern group in each file,
    with upper and lower heating element overlays on the water temperature curves."""
    all_figs = []
    figure_metadata = []  # List to store metadata separately
    water_temp_cutoff = 43.3333  # 110 F 15 deg delta from 125 F for UEF test

    for i, (file, df) in enumerate(dfs.items()):
        # Get UEF value for this file
        uef = uef_values[i]

        # Find matching columns for this file
        column_groups = find_matching_columns(df, patterns)

        # Extract additional parameters for title
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

        water_volume_col = "Water Volume (L)"
        if water_volume_col not in df.columns:
            water_volume_gal = 45.0
        else:
            water_volume_gal = df[water_volume_col].iloc[-1] * L_TO_GAL_RATIO

        # Create a plot for each pattern that has matching columns
        for pattern, columns in column_groups.items():
            if not columns:  # Skip if no matching columns found
                continue

            # Create a new figure
            fig = go.Figure()

            # Determine the overall temperature range across all columns for this pattern
            temp_min = float('inf')
            temp_max = float('-inf')
            for col in columns:
                temp_min = min(temp_min, df[col].min())
                temp_max = max(temp_max, df[col].max())
            temp_range = temp_max - temp_min
            temp_padding = temp_range * 0.1

            # Add temperature traces for each matching column
            for col in columns:
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

                # Add blue overlay for upper element on regions
                for j, (start_time, end_time) in enumerate(upper_regions):
                    fig.add_trace(
                        go.Scatter(
                            x=[start_time, end_time, end_time, start_time, start_time],
                            y=[temp_min - temp_padding, temp_min - temp_padding,
                               temp_max + temp_padding, temp_max + temp_padding,
                               temp_min - temp_padding],
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
                # Add red overlay for lower element on regions
                for j, (start_time, end_time) in enumerate(lower_regions):
                    fig.add_trace(
                        go.Scatter(
                            x=[start_time, end_time, end_time, start_time, start_time],
                            y=[temp_min - temp_padding, temp_min - temp_padding,
                               temp_max + temp_padding, temp_max + temp_padding,
                               temp_min - temp_padding],
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


            # Add hot water cutoff line
            fig.add_trace(
                go.Scatter(
                    x=df['Time'], y=[water_temp_cutoff for _ in df['Time']],  # invisible point 
                    mode='lines',
                    line=dict(color='red', width=1),
                    name='Hot Water Cutoff Temp (110°F / 43.33°C)',
                    showlegend=True
                )
            )
            # Update the layout with UEF and other file-specific info in the title
            fig.update_layout(
                title=f'{pattern} Temperatures - {file}<br>'
                      f'UEF: {uef:.3f} | PCM h: {pcm_h} W/m^2K | PCM SA Ratio: {pcm_sa} | PCM Mass: {pcm_mass:.3f} kg | '
                      f'Water Volume: {water_volume_gal:.1f} gal',
                xaxis_title='Time',
                yaxis_title='Temperature',
                height=600,
                showlegend=True
            )
            fig.update_yaxes(range=[temp_min - temp_padding, temp_max + temp_padding])

            # Store metadata for later reference
            figure_metadata.append({
                'pattern': pattern,
                'file': file,
                'type': 'temperature_pattern'
            })
            


            all_figs.append(fig)

    return all_figs, figure_metadata

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
                      f'UEF: {uef:.3f} | PCM h: {pcm_h} W/m^2K | PCM SA Ratio: {pcm_sa} | PCM Mass: {pcm_mass:.3f} kg | '
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
        Q_cons_total = Q_cons - PCM_net_heat_loss          # make sure in W*min                            
        UEF = Q_load / Q_cons_total
    else:
        water_volume = df[water_volume_col].iloc[-1]
        water_net_energy = calculate_net_water_energy(water_volume, df['Hot Water Average Temperature (C)'].iloc[-1], water_net_temp_delta) / 60 # make sure in W*min
        Q_cons_total = Q_cons - PCM_net_heat_loss - water_net_energy          # make sure in W*min                            
        UEF = Q_load / Q_cons_total
    
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
    
    # Sum the PCM enthalpy columns row-wise.

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
                    name='Hot Water Cutoff Temp (110°F / 43.33°C)',
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

def plot_pcm_enthalpies(df):
    """
    Plot cp and Enthalpy vs Temperature on the same plot with dual y-axes.
    
    Args:
        df (numpy.ndarray): Array containing columns:
                            'Temp (C)', 'cp (J/g-C)', and 'Enthalpy (J/kg)'
    
    Returns:
        fig (plotly.graph_objects.Figure): Plotly figure object.
    """
    
    fig = go.Figure()

    # Plot cp vs Temperature on the primary y-axis
    fig.add_trace(
        go.Scatter(
            x=df[:,0],  # Temperature
            y=df[:,1],  # Specific Heat Capacity
            name='cp (J/g-C)',
            mode='lines+markers',
            yaxis='y1'  # Attach to primary y-axis
        )
    )

    # Plot Enthalpy vs Temperature on the secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=df[:,0],  # Temperature
            y=df[:,2],  # Enthalpy
            name='Enthalpy (J/kg)',
            mode='lines+markers',
            yaxis='y2'  # Attach to secondary y-axis
        )
    )

    # Update layout with dual y-axis
    fig.update_layout(
        title='PCM cp and Enthalpy vs Temperature',
        xaxis=dict(title='Temperature (C)'),
        yaxis=dict(
            title='cp (J/g-C)', 
            showgrid=False
        ),
        yaxis2=dict(
            title='Enthalpy (J/kg)',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        legend=dict(x=0.05, y=0.95),
        height=600
    )
    
    return fig


def format_number(value):
    """Format a numeric filename field without an unnecessary trailing ``.0``."""
    return f"{float(value):g}"


def parse_case_filename(filename):
    """Extract the title metadata encoded in a case CSV filename."""
    stem = Path(filename).stem
    case_match = re.search(r"^case(?P<case>\d+)", stem, re.IGNORECASE)
    hx_match = re.search(r"^case\d+_(?P<hx>\d+(?:\.\d+)?)_", stem, re.IGNORECASE)
    loading_match = re.search(r"(?P<loading>\d+(?:\.\d+)?)%", stem)
    if loading_match is None:
        # The 60-40 PCM55-TPU file encodes its PCM fraction as ``60-40``
        # instead of using a percent sign.
        loading_match = re.search(
            r"(?P<loading>\d+(?:\.\d+)?)-40[_-]PCM",
            stem,
            re.IGNORECASE,
        )
    setpoint_match = re.search(r"setpoint-(?P<setpoint>\d+(?:\.\d+)?)F", stem, re.IGNORECASE)
    tank_match = re.search(r"(?P<tank>\d+(?:\.\d+)?)gal", stem, re.IGNORECASE)

    return {
        "case": int(case_match.group("case")) if case_match else None,
        "hx": float(hx_match.group("hx")) * 100 if hx_match else None,
        "pcm_loading": float(loading_match.group("loading")) if loading_match else None,
        "setpoint": float(setpoint_match.group("setpoint")) if setpoint_match else None,
        "tank_size": float(tank_match.group("tank")) if tank_match else None,
    }


def format_case_title(filename):
    """Return the human-readable case title used by both comparison plots."""
    metadata = parse_case_filename(filename)
    if metadata["case"] is None:
        return Path(filename).stem

    title = f"Case {metadata['case']}"
    if metadata["hx"] is not None:
        title += f" - {format_number(metadata['hx'])}% HX"
    if metadata["pcm_loading"] is not None:
        title += f" {format_number(metadata['pcm_loading'])}% PCM Loading"
    return title


def format_baseline_title(filename):
    """Return baseline metadata to append to each generated plot title."""
    stem = Path(filename).stem
    parts = []
    if re.search(r"no[_-]?pcm", stem, re.IGNORECASE):
        parts.append("No PCM ")

    setpoint_match = re.search(r"setpoint-(\d+(?:\.\d+)?)F", stem, re.IGNORECASE)
    tank_match = re.search(r"(\d+(?:\.\d+)?)gal", stem, re.IGNORECASE)
    if setpoint_match:
        parts.append(f"Setpoint: {format_number(setpoint_match.group(1))}°F ")
    if tank_match:
        parts.append(f"Tank Size: {format_number(tank_match.group(1))} gal")
    return "| ".join(parts) or Path(filename).name


def format_baseline_filename(filename):
    """Return concise baseline tank/setpoint metadata for PNG filenames."""
    stem = Path(filename).stem
    setpoint_match = re.search(r"setpoint-(\d+(?:\.\d+)?)F", stem, re.IGNORECASE)
    tank_match = re.search(r"(\d+(?:\.\d+)?)gal", stem, re.IGNORECASE)
    parts = []
    if re.search(r"no[_-]?pcm", stem, re.IGNORECASE):
        parts.append("No PCM")
    if tank_match:
        parts.append(f"Tank {format_number(tank_match.group(1))}gal")
    if setpoint_match:
        parts.append(f"Setpoint {format_number(setpoint_match.group(1))}F")
    return "Baseline " + " ".join(parts) if parts else "Baseline"


def _safe_plot_filename(value):
    """Make a title safe to use as a Windows or POSIX filename."""
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", str(value))
    return re.sub(r"\s+", " ", value).strip().rstrip(".") or "comparison"


def save_comparison_figures(
    figures,
    case_title,
    output_folder,
    scale=4,
    case_setpoint=None,
    baseline_filename=None,
):
    """Save comparison figures as high-resolution PNG files.

    Plotly uses the figure's layout dimensions multiplied by ``scale`` for
    the exported image. With the current 800-pixel layout, the default scale
    of 4 produces a 3200-pixel-wide PNG.
    """
    if scale <= 0:
        raise ValueError("PNG image scale must be greater than zero")

    try:
        from importlib.metadata import version
        from packaging.version import Version

        plotly_version = Version(version("plotly"))
        kaleido_version = Version(version("kaleido"))
    except Exception as exc:
        raise RuntimeError(
            "PNG export requires Plotly and Kaleido. Install them with "
            "`python -m pip install --upgrade 'plotly>=6.1.1' 'kaleido>=1.0.0'`."
        ) from exc

    if plotly_version < Version("6.1.1") or kaleido_version < Version("1.0.0"):
        raise RuntimeError(
            "PNG export requires Plotly >= 6.1.1 and Kaleido >= 1.0.0 "
            f"(found Plotly {plotly_version}, Kaleido {kaleido_version}). "
            "Upgrade with `python -m pip install --upgrade 'plotly>=6.1.1' 'kaleido>=1.0.0'`."
        )

    output_folder = Path(output_folder).expanduser()
    output_folder.mkdir(parents=True, exist_ok=True)
    filename_parts = [case_title]
    if case_setpoint is not None:
        filename_parts.append(f"Graph Setpoint {format_number(case_setpoint)}F")
    if baseline_filename is not None:
        filename_parts.append(format_baseline_filename(baseline_filename))
    safe_case_title = _safe_plot_filename(" - ".join(filename_parts))
    labels = {"water": "FHR", "first_draw": "Initial Draw"}
    saved_files = {}

    for key, label in labels.items():
        figure = figures.get(key)
        if figure is None:
            continue

        output_path = output_folder / f"{safe_case_title} - {label}.png"
        print(f"Saving PNG: {output_path}")
        export_start = time.perf_counter()
        try:
            figure.write_image(str(output_path), format="png", scale=scale)
        except Exception as exc:
            raise RuntimeError(
                f"PNG export failed for {output_path}: {exc}. "
                "Install the Kaleido package with `pip install kaleido` "
                "and ensure it is available to this Python environment."
            ) from exc
        saved_files[key] = str(output_path)
        print(f"Saved PNG: {output_path} ({time.perf_counter() - export_start:.2f}s)")

    return saved_files


def load_comparison_data(case_folder, baseline_file):
    """Load all CSV case files and the explicitly selected baseline CSV."""
    case_folder = Path(case_folder).expanduser().resolve()
    if not case_folder.is_dir():
        raise NotADirectoryError(f"Case folder does not exist: {case_folder}")

    baseline_path = Path(baseline_file).expanduser()
    if not baseline_path.is_file():
        baseline_path = case_folder / baseline_path
    baseline_path = baseline_path.resolve()
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Baseline CSV does not exist: {baseline_file}")

    case_paths = sorted(
        path
        for path in case_folder.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".csv"
        and path.resolve() != baseline_path
    )
    if not case_paths:
        raise ValueError(f"No case CSV files found in {case_folder}")

    paths = case_paths + [baseline_path]
    return {path.name: pd.read_csv(path) for path in paths}


def process_single_folder(
    case_folder,
    baseline_file,
    *,
    show=TRUE,
    plot_output_folder=None,
    image_scale=4,
):
    """Run one automated comparison for a case folder and a baseline CSV."""
    total_start = time.perf_counter()

    load_start = time.perf_counter()
    dfs = load_comparison_data(case_folder, baseline_file)
    load_seconds = time.perf_counter() - load_start

    baseline_name = Path(baseline_file).name
    case_files = [name for name in dfs if name != baseline_name]
    first_case = case_files[0]
    case_metadata = parse_case_filename(first_case)

    calculation_start = time.perf_counter()
    outputs = calculate_hot_water_delivered(dfs, first_hour_test=True)
    calculation_seconds = time.perf_counter() - calculation_start

    plot_start = time.perf_counter()
    plot_result = plot_2d_comparison_generic(
        dfs,
        outputs,
        "sa_ratio",
        "h (W/m^2K)",
        setpoint=case_metadata["setpoint"],
        tank_size=case_metadata["tank_size"],
        baseline_file=baseline_name,
        case_title=format_case_title(first_case),
        baseline_title=format_baseline_title(baseline_name),
        show=show,
        save_folder=(
            Path(plot_output_folder).expanduser()
            if plot_output_folder is not None
            else Path(case_folder).expanduser().resolve() / "plots"
        ),
        image_scale=image_scale,
    )
    plot_seconds = time.perf_counter() - plot_start
    total_seconds = time.perf_counter() - total_start

    if plot_result is None:
        print("Comparison finished, but no plot result was returned; no PNGs were saved.")
    elif not plot_result.get("saved_files"):
        print("Comparison finished, but no figures contained exportable data; no PNGs were saved.")

    print(
        "Timing: "
        f"loaded {len(dfs)} CSVs in {load_seconds:.2f}s; "
        f"calculated metrics in {calculation_seconds:.2f}s; "
        f"generated/exported plots in {plot_seconds:.2f}s; "
        f"total {total_seconds:.2f}s"
    )
    print(f"Finished comparison for {case_folder} against {baseline_name}")
    return {
        "dfs": dfs,
        "outputs": outputs,
        "plot": plot_result,
        "timing": {
            "load_seconds": load_seconds,
            "calculation_seconds": calculation_seconds,
            "plot_seconds": plot_seconds,
            "total_seconds": total_seconds,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Create 2-D comparison plots for a folder of case CSV files."
    )
    parser.add_argument("case_folder", type=Path, help="Folder containing the case CSV files")
    parser.add_argument("baseline_file", type=Path, help="Baseline CSV path or filename")
    parser.add_argument("--no-show", action="store_true", help="Build figures without opening Plotly windows")
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="Output folder for PNGs (default: <case_folder>/plots)",
    )
    parser.add_argument(
        "--image-scale",
        type=int,
        default=4,
        help="PNG resolution multiplier (default: 4)",
    )
    args = parser.parse_args(argv)

    if args.image_scale < 1:
        parser.error("--image-scale must be at least 1")

    process_single_folder(
        args.case_folder,
        args.baseline_file,
        show=not args.no_show,
        plot_output_folder=args.plot_output,
        image_scale=args.image_scale,
    )


if __name__ == "__main__":
    main()
