import numpy as np
import pandas as pd
import re

def calculate_hot_water_delivered(dfs, first_hour_test=False):
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
                        energy_J = heat_W * dt
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