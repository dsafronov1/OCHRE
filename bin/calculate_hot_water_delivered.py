import os
import re
import numpy as np
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# =========================
# Core per-file computation
# =========================
def _process_core(file_key, df, first_hour_test, water_temp_cutoff=43.333, L_TO_GAL_RATIO=0.264172):
    water_draw_col       = "Total Water Output (L/min)"
    water_output_W_col   = "Hot Water Delivered (W)"
    water_outlet_temp    = "Hot Water Outlet Temperature (C)"
    energy_used_col      = "Water Heating Delivered (W)"
    try:
        df_copy = df.copy()

        # --- prepare ---
        max_water_volume_L       = 180  # fixed cap like your original
        total_water_volume_L     = 0.0
        total_heat_delivered_J   = 0.0
        is_draw_active           = False
        draw_events              = []
        current_event            = None
        is_pcm                   = False

        # time deltas (s)
        df_copy.index = pd.to_datetime(df_copy['Time'])
        df_copy['time_delta'] = (
            pd.to_datetime(df_copy.index).to_series().diff().dt.total_seconds().fillna(0.0)
        )

        pcm_columns = [col for col in df_copy.columns if col.startswith('T_PCM')]
        if len(pcm_columns) > 0:
            is_pcm = True
            pcm_enthalpy_column   = 'Total PCM Enthalpy (J)'
            starting_pcm_enthalpy = df_copy[pcm_enthalpy_column].iloc[0]
            df_copy['average_pcm_temp'] = df_copy[pcm_columns].mean(axis=1)
            df_copy['is_cutoff_temp']   = df_copy['average_pcm_temp'] < water_temp_cutoff
            try:
                cutoff_index      = df_copy[df_copy['is_cutoff_temp']].index[0]
                baseline_enthalpy = df_copy[pcm_enthalpy_column][cutoff_index]
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
            temp_C         = row[water_outlet_temp]
            heat_W         = row[water_output_W_col]
            dt             = float(row['time_delta'])
            if dt < 0 or not np.isfinite(dt):
                dt = 0.0
            if is_pcm:
                avg_end_pcm_temp = row['average_pcm_temp']
                enthalpy         = row['Total PCM Enthalpy (J)']

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
                        'temp_readings': [] if first_hour_test else None,
                    }
                # update envelope
                current_event['end_time']      = timestamp
                current_event['max_flow_rate'] = max(current_event['max_flow_rate'], flow_L_per_min)
                current_event['max_temp']      = max(current_event['max_temp'], temp_C)
                current_event['min_temp']      = min(current_event['min_temp'], temp_C)
                if first_hour_test:
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
                    if first_hour_test and current_event['temp_readings']:
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
            if first_hour_test and current_event['temp_readings']:
                current_event['avg_temp'] = sum(current_event['temp_readings']) / len(current_event['temp_readings'])
            if is_pcm:
                current_event['pcm_enthalpy'] = enthalpy
                current_event['avg_pcm_temp'] = avg_end_pcm_temp
                denom = (baseline_enthalpy if baseline_enthalpy != 0 else 1.0)
                current_event['pcm_soc'] = (enthalpy - baseline_enthalpy) / denom
            draw_events.append(current_event)

        # aggregate totals
        total_water_volume_gal   = total_water_volume_L * 0.264172
        total_heat_delivered_kWh = total_heat_delivered_J * 2.77778e-7
        # integrate energy with dt (assumes W)
        total_energy_used_kwh = float(((df_copy['Water Heating Delivered (W)'] * df_copy['time_delta']).sum()) / 3600.0 / 1000.0)

        # first-hour adjustment
        if first_hour_test and len(draw_events) >= 2:
            final = draw_events[-1]
            prev  = draw_events[-2]
            dur = (final['end_time'] - final['start_time']).total_seconds()
            if dur >= 30 and final['max_temp'] >= water_temp_cutoff:
                denom = (prev.get('avg_temp', prev['min_temp']) - prev['min_temp']) or 1.0
                adj   = ((final.get('avg_temp', final['min_temp']) - prev['min_temp']) / denom)
                adjusted_gal           = (total_water_volume_gal - final['water_volume_gal']) + (final['water_volume_gal'] * adj)
                total_water_volume_gal = adjusted_gal
                total_water_volume_L   = total_water_volume_gal / 0.264172

        # PCM tail metrics
        try:
            if is_pcm and len(pcm_columns) > 0:
                pcm_temps = [df[col].iloc[-1] for col in pcm_columns]
                average_pcm_end_temp = float(np.mean(pcm_temps))
                pcm_soc = (df_copy['Total PCM Enthalpy (J)'].iloc[-1] - baseline_enthalpy) / ((starting_pcm_enthalpy - baseline_enthalpy) or 1.0)
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
            'total_water_volume_L': total_water_volume_L,
            'total_water_volume_gal': total_water_volume_gal,
            'total_energy_used_kwh': total_energy_used_kwh,
            'total_heat_delivered_J': total_heat_delivered_J,
            'total_heat_delivered_kWh': total_heat_delivered_kWh,
            'max_possible_hot_water': max_water_volume_L * 0.264172,
            'draw_events': draw_events,
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
