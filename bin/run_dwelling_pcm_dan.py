import os
import datetime as dt
import pandas as pd

from ochre import Dwelling, Analysis, CreateFigures
from ochre.Models import TankWithMultiPCM
from ochre.Models.WaterPCM import TankWithMultiPCMExternal
from ochre.utils import default_input_path

# Test script to run single Dwelling

pd.set_option('display.precision', 3)      # precision in print statements
pd.set_option('expand_frame_repr', False)  # Keeps results on 1 line
pd.set_option('display.max_rows', 30)      # Shows up to 30 rows of data
# pd.set_option('max_columns', None)       # Prints all columns

start_node = 4
end_node = 9

# volume fraction of each node
vol_fracs = [0.74]

# Can specify volume fractions for each node independantly
# pcm_vol_fractions = [{4: 0.5, 5: 0.8, 6: 0.6}
pcm_vol_fractions = []
for vol_fract in vol_fracs:
    pcm_vol_fractions.append(
        {node: vol_fract for node in range(start_node, end_node + 1)}
    )
    
DEFAULT_PCM_PROPERTIES = {
"h": 600,  # W/m^2K
"sa_ratio": 16, # m^2/m^3 of total pcm heat exchanger volume
"h_conv": 1200,  # W/m^2-K, accounts for surface area (ha)
"solid": {
    "pcm_density": 0.904,  # g/cm^3
    "pcm_cp": 0.6,  # J/g-C # adjusted by real measurements average from 0-45c
    "pcm_conductivity":0.2,  # W/m-C
},
"enthalpy_lut_file": "cp_h-T_data_shifted_120F.csv",
}

DEFAULT_PCM_PROPERTIES_EXTERNAL = {
    "solid": {
        "pcm_density": 0.991,  # g/cm**3 10% graphite 90% pcm
        "pcm_cp": 0.6,  # J/g-C # adjusted by real measurements average from 0-45c
        # "pcm_conductivity": 0.28,  # W/m-C, not used Bulk PCM conductivity
        "pcm_conductivity": 10,  # W/m-C, not used graphite infiltrated PCM conductivity
        # "pcm_c": 1717.6,  # J/m**3-C, not used
    },
    "enthalpy_lut": "cp_h-T_data_shifted_120F.csv",
}


dwelling_args = {
    # 'name': 'OCHRE_Test_House'  # simulation name

    # Timing parameters
    'start_time': dt.datetime(2018, 1, 1, 0, 0),  # year, month, day, hour, minute
    'time_res': dt.timedelta(minutes=1),         # time resolution of the simulation
    'duration': dt.timedelta(days=2),             # duration of the simulation
    'initialization_time': dt.timedelta(days=1),  # used to create realistic starting temperature
    'time_zone': None,                            # option to specify daylight savings, in development

    # Input parameters - Sample building (uses HPXML file and time series schedule file)
    'hpxml_file': os.path.join(default_input_path, 'Input Files', 'sample_resstock_properties.xml'),
    'schedule_input_file': os.path.join(default_input_path, 'Input Files', 'sample_resstock_schedule.csv'),

    # Input parameters - weather (note weather_path can be used when Weather Station is specified in HPXML file)
    # 'weather_path': weather_path,
    'weather_file': os.path.join(default_input_path, 'Weather', 'USA_CO_Denver.Intl.AP.725650_TMY3.epw'),

    # Output parameters
    'verbosity': 3,                         # verbosity of time series files (0-9)
    # 'metrics_verbosity': 6,               # verbosity of metrics file (0-9), default=6
    'save_results': True,                # saves results to files. Defaults to True if verbosity > 0
    'output_path': os.getcwd(),           # defaults to hpxml_file path
    # 'save_args_to_json': True,            # includes data from this dictionary in the json file
    # 'output_to_parquet': True,            # saves time series files as parquet files (False saves as csv files)
    # 'save_schedule_columns': [],          # list of time series inputs to save to schedule file
    # 'export_res': dt.timedelta(days=61),  # time resolution for saving files, to reduce memory requirements

    # Envelope parameters
    # 'Envelope': {
    #     'save_results': True,  # Saves detailed envelope inputs and states
    #     'linearize_infiltration': True,
    #     'external_radiation_method': 'linear',
    #     'internal_radiation_method': 'linear',
    #     'reduced_states': 7,
    #     'save_matrices': True,
    #     'zones': {'Indoor': {
    #         'enable_humidity': False,
    #     }},
    # },

    # Occupancy parameters
    # 'Occupancy': {
    #     'Number of Occupants (-)': 3,
    # },

    # Equipment parameters
    'Equipment': {
        # HVAC equipment
        # Note: dictionary key can be end use (e.g., HVAC Heating) or specific equipment name (e.g., Gas Furnace)
        # 'HVAC Heating': {
        #     # 'use_ideal_capacity': True,
        #     # 'show_eir_shr': True,
        # },
        # 'Air Conditioner': {
        #     'speed_type': 'Double',
        # },
        # 'Gas Furnace': {
        #     'heating capacity (W)': 6000,
        #     # 'supplemental heating capacity (W)': 6000,
        # },

        # Water heating equipment
        # Note: dictionary key can be end use (Water Heating) or specific equipment name (e.g., Gas Water Heater)
        # 'Water Heating': {
        #     'water_nodes': 12,
        #     'rc_params': {'R_WH1_AMB': 1,
        #                 'C_WH1': 1e6},
        #     'Water Tank': {
        #         'save_results': True,
        #     },
        # },
        # 'Heat Pump Water Heater': {
        #     'HPWH COP (-)': 4.5,
        #     'Tank Volume (L)': 40 * 3.78541,
        #     'hp_only_mode': True,
        #     "model_class": TankWithMultiPCM,
        #     "Setpoint Temperature (C)": 60,
        #     "Water Tank": {
        #         "pcm_node_vol_fractions": pcm_vol_fractions[0],
        #         "pcm_properties": DEFAULT_PCM_PROPERTIES,
        #     },
        # },
        'Heat Pump Water Heater': {
            'HPWH COP (-)': 4.5,
            'Tank Volume (L)': 40 * 3.78541,
            'hp_only_mode': True,
            "model_class": TankWithMultiPCMExternal,
            "Setpoint Temperature (C)": 60,
            "Water Tank": {
                "pcm_properties": DEFAULT_PCM_PROPERTIES,
                "pcm_thickness_in": 1,
                "pcm_segment_thickness_in": 1,  # in PCM thickness
                "insulation_thickness_in": 2,  # this is pcm + insulation thickness aka the max thickness outside of the tank
                "insulation_k_value": 0.0484,  # W/m·K
                "insulation_cp_value": 1000.0,  # J/kg·K
                "insulation_density": 40.0,  # kg/m³
                "enamel_thickness_in": 0.008,  # ≈0.2 mm glass-enamel
                "enamel_k_value": 1.0,  # W/m·K (vitreous enamel)
                "enamel_cp_value": 840.0,  # J/kg·K (glass)
                "enamel_density": 2500.0,  # kg/m³ (glass)
                "steel_wall_thickness_in": 0.1,  # ≈2.75 mm total wall
                "steel_k_value": 55.0,  # W/m·K mildsteel
                "steel_cp_value": 490.0,  # J/kg·K mild steel
                "steel_density": 7850.0,  # kg/m³
                "water_side_film_h": 50,  # W/m²·K This value various with flow rate is set static for now
            },
        },
        # 'Electric Resistance Water Heater': {
        #     'use_ideal_capacity': True,
        # },

        # Other equipment
        # 'EV': {
        #     'vehicle_type': 'PHEV',
        #     'charging_level': 'Level 1',
        #     'mileage': 50,
        # },
        # 'PV': {
        #     'capacity': 5,
        #     'tilt': 20,
        #     'azimuth': 180,
        # },
        # 'Battery': {
        #     'capacity_kwh': 6,
        #     'capacity': 3,
        #     'soc_init': 0.5,
        #     'zone': 'Indoor',
        #     # 'control_type': 'Schedule',
        #     'verbosity': 6,
        # },
    },

    # 'modify_hpxml_dict': {},  # Directly modifies values from HPXML input file
    # 'schedule': {},  # Directly modifies columns from OCHRE schedule file (dict or pandas.DataFrame)
}

if __name__ == '__main__':
    # Initialization
    dwelling = Dwelling(**dwelling_args)

    # Simulation
    df, metrics, hourly = dwelling.simulate()

    # Load results from previous run
    # output_path = dwelling_args.get('output_path', os.path.dirname(dwelling_args['hpxml_file']))
    # df, metrics, hourly = Analysis.load_ochre(output_path, simulation_name)

    # Plot results
    data = {'': df}
    CreateFigures.plot_all_powers(data)
    CreateFigures.plot_power_stack(df)
    # CreateFigures.plot_envelope(data)
    # CreateFigures.plot_hvac(data)
    CreateFigures.plt.show()
