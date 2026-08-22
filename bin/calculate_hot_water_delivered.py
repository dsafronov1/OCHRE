import os
import re
import numpy as np
import pandas as pd
import multiprocessing as mp
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# from bin.run_pcm_wh_janelle import GAL_TO_L

GAL_TO_L = 3.78541 # gallons to liters conversion
L_TO_GAL_RATIO = 0.264172

# =========================
# Core per-file computation
# =========================


def _process_core(file_key, df, first_hour_test, water_temp_cutoff=43.333, L_TO_GAL_RATIO=0.264172):
    water_draw_col       = "Total Water Output Delivered (L/min)"
    water_output_W_col   = "Hot Water Delivered (W)"
    water_draw_tank_col  = "Hot Water Delivered (L/min)"
    inlet_temp_col       = "Hot Water Mains Temperature (C)"
    water_heater_outlet_temp    = "Hot Water Outlet Temperature (C)"
    water_outlet_temp           = "Total Water Output Delivered Temperature (C)"
    energy_used_col      = "Water Heating Delivered (W)"
    try:
        df_copy = df.copy()

        # --- prepare ---
        max_water_volume_L       = 0
        total_water_volume_L     = 0.0
        total_heat_delivered_J   = 0.0
        is_draw_active           = False
        draw_events              = []
        current_event            = None
        is_pcm                   = False

        # time deltas (s)
        df_copy.index = pd.to_datetime(df_copy['Time'], errors='coerce', format='mixed')
        df_copy['time_delta'] = (
            pd.to_datetime(df_copy.index).to_series().diff().dt.total_seconds().fillna(0.0)
        )
        
        # compare begining and end timestamp to find length of test to determine maximum possible water volume
        beginning = df_copy['Time'].iloc[0]
        end = df_copy['Time'].iloc[-1]
        test_duration = parse_timestamp_to_string(end) - parse_timestamp_to_string(beginning)
        test_duration_mins = test_duration.total_seconds() / 60
        water_draw = 3 #gpm from FHR
        max_water_volume_L = test_duration_mins * water_draw * GAL_TO_L

        pcm_columns = [col for col in df_copy.columns if col.startswith('T_PCM')]
        if len(pcm_columns) > 0:
            is_pcm = True
            PCM_COLUMN_ALIASES = {
                "total_pcm_enthalpy": [
                    "Total PCM Enthalpy (J)",
                    "Total Water Heater PCM Enthalpy (J)",
                ],
            }

            def get_first_existing_column(df, aliases):
                for col in aliases:
                    if col in df.columns:
                        return col
                raise KeyError(f"None of the columns found: {aliases}")

        # Usage
            pcm_enthalpy_col = get_first_existing_column(df_copy, PCM_COLUMN_ALIASES["total_pcm_enthalpy"])
            pcm_series = df_copy[pcm_enthalpy_col]
            starting_pcm_enthalpy = pcm_series.iloc[0]
            df_copy['average_pcm_temp'] = df_copy[pcm_columns].mean(axis=1)
            df_copy['is_cutoff_temp'] = df_copy[pcm_columns].lt(water_temp_cutoff).all(axis=1)
            try:
                cutoff_index      = df_copy[df_copy['is_cutoff_temp']].index[0]
                baseline_enthalpy = df_copy[pcm_series][cutoff_index]
            except Exception:
                # Fallback: interpolate enthalpy at cutoff temp from LUT
                m = re.search(r'setpoint-[^_]+_(.*?)_\d+gal', file_key)
                if not m:
                    raise ValueError(f"Cannot extract PCM config from filename: {file_key}")
                base_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
                LUT = os.path.join(base_dir, "../ochre/defaults/pcm_configs", m.group(1) + ".csv")
                enthalpy_lut = np.loadtxt(LUT, delimiter=",", skiprows=1)
                temps = enthalpy_lut[:,0]; enths = enthalpy_lut[:,2]
                h_cutoff = np.interp(water_temp_cutoff, temps, enths)
                baseline_enthalpy = h_cutoff * df_copy['PCM Mass (kg)'].iloc[-1] * 1000  # → J

        # --- iterate rows ---
        for timestamp, row in df_copy.iterrows():
            flow_L_per_min = row[water_draw_col]
            tank_L_per_min = row[water_draw_tank_col]
            intlet_temp_C  = row[inlet_temp_col]
            temp_C         = row[water_outlet_temp]

            heat_W         = row[water_output_W_col]
            dt             = float(row['time_delta'])
            if dt < 0 or not np.isfinite(dt):
                dt = 0.0
            if is_pcm:
                avg_end_pcm_temp = row['average_pcm_temp']
                enthalpy         = row[pcm_enthalpy_col]

            if flow_L_per_min > 0:
                if not is_draw_active:
                    is_draw_active = True
                    current_event = {
                        'start_time': timestamp,
                        'end_time': None,
                        'water_volume_L': 0.0,
                        'heat_delivered_J': 0.0,
                        'max_temp': temp_C,
                        'min_temp': temp_C,
                        'max_flow_rate': flow_L_per_min,
                        'temp_readings': [],
                    }
                # update envelope
                current_event['end_time']      = timestamp
                current_event['max_flow_rate'] = max(current_event['max_flow_rate'], flow_L_per_min)
                current_event['max_temp']      = max(current_event['max_temp'], temp_C)
                current_event['min_temp']      = min(current_event['min_temp'], temp_C)
                current_event['temp_readings'].append(temp_C)

                # only count when outlet >= cutoff
                if temp_C >= water_temp_cutoff and dt > 0:
                    water_L = flow_L_per_min * dt / 60.0                # (L/min)*(s)/60 → L
                    energy_J = heat_W * dt                               # W*s → J
                    total_water_volume_L     += water_L
                    total_heat_delivered_J   += energy_J
                    current_event['water_volume_L']  += water_L
                    current_event['heat_delivered_J'] += energy_J

            elif is_draw_active:
                # draw ended
                if current_event and current_event['water_volume_L'] > 0:
                    current_event['water_volume_gal']   = current_event['water_volume_L'] * L_TO_GAL_RATIO
                    current_event['heat_delivered_kWh'] = current_event['heat_delivered_J'] * 2.77778e-7
                    if current_event['temp_readings']:
                        current_event['avg_temp'] = sum(current_event['temp_readings']) / len(current_event['temp_readings'])
                    if is_pcm:
                        current_event['pcm_enthalpy'] = enthalpy
                        current_event['avg_pcm_temp'] = avg_end_pcm_temp
                        denom = (baseline_enthalpy if baseline_enthalpy != 0 else 1.0)
                        current_event['pcm_soc'] = (enthalpy - baseline_enthalpy) / denom
                    draw_events.append(current_event)
                is_draw_active = False
                current_event  = None

        # close last event if still open
        if is_draw_active and current_event and current_event['water_volume_L'] > 0:
            current_event['water_volume_gal']   = current_event['water_volume_L'] * L_TO_GAL_RATIO
            current_event['heat_delivered_kWh'] = current_event['heat_delivered_J'] * 2.77778e-7
            if current_event['temp_readings']:
                current_event['avg_temp'] = sum(current_event['temp_readings']) / len(current_event['temp_readings'])
            if is_pcm:
                current_event['pcm_enthalpy'] = enthalpy
                current_event['avg_pcm_temp'] = avg_end_pcm_temp
                denom = (baseline_enthalpy if baseline_enthalpy != 0 else 1.0)
                current_event['pcm_soc'] = (enthalpy - baseline_enthalpy) / denom
            draw_events.append(current_event)

        # aggregate totals
        total_water_volume_gal   = total_water_volume_L * 0.264172
        total_water_drawn_volume_L = total_water_volume_L
        total_water_drawn_volume_gal = total_water_volume_gal
        total_heat_delivered_kWh = total_heat_delivered_J * 2.77778e-7
        # integrate energy with dt (assumes W)
        total_energy_used_kwh = float(((df_copy['Water Heating Delivered (W)'] * df_copy['time_delta']).sum()) / 3600.0 / 1000.0)

        # first-hour adjustment
        if first_hour_test and len(draw_events) >= 2:
            # determine if test is a FHR test by length
            if test_duration_mins <=120:
                final = draw_events[-1]
                prev  = draw_events[-2]
                dur = (final['end_time'] - final['start_time']).total_seconds()
                if dur >= 30 and final['max_temp'] >= water_temp_cutoff:
                    numerator = (final.get('avg_temp', final['min_temp']) - prev['min_temp'])
                    denom = (prev.get('avg_temp', prev['min_temp']) - prev['min_temp']) or 1.0
                    adj   = numerator/denom
                    adjusted_gal           = (total_water_volume_gal - final['water_volume_gal']) + (final['water_volume_gal'] * adj)
                    total_water_volume_gal = adjusted_gal
                    total_water_volume_L   = total_water_volume_gal / 0.264172
            else:
                first_hour_test = False
        elif not first_hour_test:
            total_water_volume_gal = 0.0
            total_water_volume_L = 0.0

        # PCM tail metrics
        try:
            if is_pcm and len(pcm_columns) > 0:
                pcm_temps = [df[col].iloc[-1] for col in pcm_columns]
                average_pcm_end_temp = float(np.mean(pcm_temps))
                pcm_soc = (df_copy[pcm_enthalpy_col].iloc[-1] - baseline_enthalpy) / ((starting_pcm_enthalpy - baseline_enthalpy) or 1.0)
            else:
                average_pcm_end_temp = 14.44
                pcm_soc = -100
        except Exception as e:
            print(f"Error calculating average PCM temperature for file [{file_key}]: {e}")
            average_pcm_end_temp = 14.44
            pcm_soc = -100

        return file_key, {
            "average_pcm_end_temp": average_pcm_end_temp,
            "pcm_soc": pcm_soc,
            'total_water_delivered_volume_L': total_water_drawn_volume_L,
            'total_water_delivered_volume_gal': total_water_drawn_volume_gal,
            'total_water_FHR_volume_L': total_water_volume_L,
            'total_water_FHR_volume_gal': total_water_volume_gal,
            'total_energy_used_kwh': total_energy_used_kwh,
            'total_heat_delivered_J': total_heat_delivered_J,
            'total_heat_delivered_kWh': total_heat_delivered_kWh,
            'max_possible_hot_water_gal': max_water_volume_L * L_TO_GAL_RATIO,
            'draw_events': draw_events,
            'is_FHR': first_hour_test,
            'num_draw_events': len(draw_events)
        }
    except Exception as e:
        return file_key, {"error": f"Error processing file [{file_key}]: {e}"}

# Wrapper for DataFrames-in-memory (simpler; pays pickling cost)
def _process_file_df(args):
    file_key, df, first_hour_test = args
    return _process_core(file_key, df, first_hour_test)

# Wrapper for file paths (best performance; loads inside worker)
def _process_file_path(args):
    file_key, path, first_hour_test, read_csv_kwargs = args
    df = pd.read_csv(path, **(read_csv_kwargs or {}))
    return _process_core(file_key, df, first_hour_test)


def calculate_uef(df, name):
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

    def parse_tank_volume_from_name(filename):
        """
        Parses a filename to extract tank volume in gallons.
        Returns a tuple: (volume_in_gal, is_default)
        """
        is_default = "zDefault" in filename
        match = re.search(r'_([0-9]+(?:\.[0-9]+)?)gal_', filename)
        volume = float(match.group(1)) if match else None
        return volume, is_default

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

    # If index is already named 'Time', use it directly.
    if df_all.index.name == "Time":
        df_all = df_all.reset_index()
    else:
        df_all["Time"] = df_all.index

    df_all["Time"] = _parse_time(df_all["Time"])
    df_all = df_all.dropna(subset=["Time"])
    df_all = df_all.sort_values("Time")

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

def calculate_net_water_temp(df):
    water_temp_column = ["Hot Water Average Temperature (C)"]
    valid_columns = [col for col in water_temp_column if col in df.columns]

    if not valid_columns:
        return 0

    # Compute the average of the first and last row for all available columns
    average_start_temp = df[valid_columns].iloc[0].mean()
    average_end_temp = df[valid_columns].iloc[-1].mean()

    return average_end_temp - average_start_temp

def calculate_net_PCM_heat(df):
    # Dynamically find all PCM enthalpy columns.
    pcm_column = "Total PCM Heat Injected (W)"

    if pcm_column not in df.columns:
        return 0

    # Sum the PCM enthalpy columns row-wise.

    net_PCM_to_water_heat_Transfer = df[pcm_column].sum()

    return net_PCM_to_water_heat_Transfer

def calculate_net_PCM_enthalpy(df):
    # Dynamically find all PCM enthalpy columns.
    pcm_column = "Total PCM Enthalpy (J)"

    if pcm_column not in df.columns:
        return 0

    # Sum the PCM enthalpy columns row-wise.

    net_PCM_enthalpy = df[pcm_column].iloc[-1] - df[pcm_column].iloc[0]

    return net_PCM_enthalpy

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
    # Predefined lookup arrays for water properties at 1 atm.
    _TEMPS = np.array([0, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
    _DENSITIES = np.array(
        [
            999.8,
            1000.0,
            999.7,
            998.2,
            995.7,
            992.2,
            988.1,
            983.2,
            977.8,
            971.8,
            965.3,
            958.4,
        ],
        dtype=float,
    )
    _THERMALEXPANSIONS = np.array(
        [
            -1.0e-4,
            0.0,
            1.1e-4,
            2.1e-4,
            2.6e-4,
            3.1e-4,
            3.6e-4,
            4.1e-4,
            4.7e-4,
            5.3e-4,
            5.9e-4,
            6.6e-4,
        ],
        dtype=float,
    )
    
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

def calculate_net_water_energy(volume, temperature, temperature_difference):
    water_properties_start = lookup_water_properties(temperature - temperature_difference)
    water_properties_end = lookup_water_properties(temperature)
    density_start = water_properties_start["density"]
    density_end = water_properties_end["density"]
    density_average = (density_start + density_end) / 2
    # volume is in Liters
    water_weight_average = density_average * volume / 1e3  # in kg

    return water_weight_average * temperature_difference * 4184  # J

def parse_timestamp_to_string(date_str):
    try:
        # Try with fractional seconds first
        return datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        # Fallback to whole seconds
        return datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    

# ======================================================
# Public APIs: choose one depending on how your data is
# ======================================================

def calculate_hot_water_delivered(dfs, first_hour_test=False, workers=None):
    """
    Multiprocessing version that accepts a dict {file_key: DataFrame}.
    Uses all CPU cores by default. On Windows, call from inside
    if __name__ == '__main__': to avoid spawn recursion.
    """
    if not dfs:
        return {}
    max_workers = workers or (os.cpu_count() or 1)
    # Explicit spawn context for Windows & safety
    ctx = mp.get_context("spawn")
    output = {}
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futures = {ex.submit(_process_file_df, (k, v, first_hour_test)): k for k, v in dfs.items()}
        for fut in as_completed(futures):
            k, res = fut.result()
            output[k] = res
    return output

def calculate_hot_water_delivered_from_paths(file_map, first_hour_test=False, workers=None, read_csv_kwargs=None):
    """
    Faster, low-overhead alternative: pass {file_key: /path/to.csv}.
    Each worker loads its own CSV → far less pickling/memory churn.
    """
    if not file_map:
        return {}
    max_workers = workers or (os.cpu_count() or 1)
    ctx = mp.get_context("spawn")
    output = {}
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futures = {
            ex.submit(_process_file_path, (k, path, first_hour_test, read_csv_kwargs)): k
            for k, path in file_map.items()
        }
        for fut in as_completed(futures):
            k, res = fut.result()
            output[k] = res
    return output
