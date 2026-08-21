import datetime as dt
import argparse
import json
import numpy as np
import pandas as pd
import os
import shutil
import tempfile
from pathlib import Path

from ochre import (
    HeatPumpWaterHeater,
    CreateFigures,
    ElectricResistanceWaterHeater,
)
from ochre.Models import TankWithMultiPCM
from ochre.utils import convert
from bin.calculate_hot_water_delivered import calculate_hot_water_delivered_from_paths
from bin.create_output_csv import export_draw_outputs_csv
import time
from bin.run_dwelling import dwelling_args
import multiprocessing
import copy
import re


RUNNER_ROOT = Path(__file__).resolve().parent.parent

# ANSI Terminal color codes
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"


def _format_progress_duration(seconds):
    """Format elapsed or estimated time as HH:MM:SS."""
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _progress_bar(completed, total, width=24):
    """Return a compact ASCII progress bar."""
    if total <= 0:
        ratio = 1.0
    else:
        ratio = min(1.0, max(0.0, completed / total))
    filled = int(ratio * width)
    return f"[{'#' * filled}{'-' * (width - filled)}]"


class ProgressBars:
    """Track current-scenario and all-scenarios simulation progress."""

    def __init__(self, total_cases):
        self.total_cases = max(0, int(total_cases))
        self.global_completed = 0
        self.global_started = time.perf_counter()
        self.case_name = None
        self.case_total = 0
        self.case_completed = 0
        self.case_started = None
        self.last_case_report = 0
        self.last_global_report = 0

    @staticmethod
    def _line(label, completed, total, started):
        elapsed = time.perf_counter() - started
        if completed > 0:
            eta_seconds = elapsed / completed * max(0, total - completed)
            eta = _format_progress_duration(eta_seconds)
        else:
            eta = "--:--:--"
        percent = 100.0 if total <= 0 else min(100.0, completed / total * 100)
        return (
            f"{label:<18} {_progress_bar(completed, total)} "
            f"{completed:>5}/{total:<5} ({percent:5.1f}%) "
            f"elapsed {_format_progress_duration(elapsed)} ETA {eta}"
        )

    def start_case(self, case_name, total_cases):
        self.case_name = str(case_name)
        self.case_total = max(0, int(total_cases))
        self.case_completed = 0
        self.case_started = time.perf_counter()
        self.last_case_report = 0
        self.render(force=True)

    def advance(self, count=1, force=False):
        if self.case_name is None:
            return
        self.case_completed = min(self.case_total, self.case_completed + count)
        self.global_completed = min(
            self.total_cases, self.global_completed + count
        )
        self.render(force=force)

    def render(self, force=False):
        if self.case_name is None or self.case_started is None:
            return
        case_step = max(1, self.case_total // 20)
        global_step = max(1, self.total_cases // 100)
        should_report = (
            force
            or self.case_completed == self.case_total
            or self.global_completed == self.total_cases
            or self.case_completed - self.last_case_report >= case_step
            or self.global_completed - self.last_global_report >= global_step
        )
        if not should_report:
            return

        print(
            self._line(
                f"Case {self.case_name}",
                self.case_completed,
                self.case_total,
                self.case_started,
            )
        )
        print(
            self._line(
                "Global total",
                self.global_completed,
                self.total_cases,
                self.global_started,
            )
        )
        self.last_case_report = self.case_completed
        self.last_global_report = self.global_completed



GAL_TO_L = 3.78541
L_TO_GAL_RATIO = 0.264172


start_node = 4
end_node = 9

DEFAULT_PCM_PROPERTIES = {
    "h": 600,  # W/m^2K
    "sa_ratio": 15, # m^2/m^3 of total pcm heat exchanger volume
    "h_conv": 100,  # W/m^2-K, accounts for surface area (ha)
    "solid": {
        "pcm_density": 0.908,  # g/cm^3
        "pcm_cp": 1.95,  # J/g-C # adjusted by real measurements average from 0-45c
        "pcm_conductivity":0.2,  # W/m-C
    },
    "enthalpy_lut_file": "cp_h-T_data_shifted_120F.csv",
}

num_points = 10

sa_ratios = [4]
h_values = [2000]
# 700 — 2200 in^2 for the MEPCM 66% vol fraction fill
# sa_ratios = [0.673573, 0.817910, 0.962247, 1.106584, 1.250921, 1.395258, 1.539595, 1.683932, 1.828269, 1.972606, 2.116943]
# 200-2000 in^2 for the backfilled 74% vol fraction fill
# sa_ratios = [
#     0.171644, 0.326124, 0.480603, 0.635083, 0.789563,
#     0.944042, 1.098522, 1.253001, 1.407481, 1.561961,
#     1.716440
# ]

# 200-2000 in^2 for the backfilled 26% vol fraction fill
# sa_ratios = [
#     0.488525, 0.928198, 1.367871, 1.807544, 2.247216,
#     2.686889, 3.126562, 3.566235, 4.005908, 4.445580,
#     4.885253
# ]

# case 1 
# sa_ratios = [15, 87, 159, 231, 303, 375]

# case 2
# sa_ratios = np.linspace(1, 16, num_points)

# case 3
# sa_ratios = np.linspace(1, 50, num_points)

# h_values = np.linspace(np.log10(50), np.log10(5000), num_points)
# h_values = np.linspace(50, 5000, num_points)
# h_values = [500]
# h_values = np.linspace(50, 5000, 20)
# h_values = [25, 525, 1025, 1525, 2025, 2525]

# pcm_file_names = [f"cp_h-T_data_shifted_{i}F.csv" for i in range(110, 142, 2)]


simulation_duration_days = 220

# pcm_file_names = ['cp_h-T_data_shifted_120F.csv']
# pcm_file_names = ['60-40_PCM55-TPU_cp-h-T.csv']

# pcm_file_names = ['ct53_h-T_data_66frac.csv']
# pcm_file_names = ['ct53_h-T_data_64frac.csv']
# pcm_file_names = ['ct53_h-T_data_58frac.csv']W
# pcm_file_names = ['ct53_h-T_data_57frac.csv']

# pcm_file_names = ['ct53-resin_h-T_data_88frac.csv']
pcm_file_names = ['CT53_90%_cp_h_T_data.csv']
# pcm_file_names = ['ct53-resin_h-T_data_85frac.csv']
# pcm_file_names = ['ct53-resin_h-T_data_83frac.csv']
# pcm_file_names = [f'100%_ct53-resin_h-T_data_88frac_{x}F.csv' for x in range (110, 142 + 1, 1)]
# pcm_file_names = [f'100%_ct53-resin_h-T_data_88frac_{x}F.csv' for x in range (110, 142 + 1, 1)]
# pcm_file_names = ['ct53-resin_h-T_data_81frac.csv']
# pcm_file_names = ['ct53-resin_h-T_data_77frac.csv']
# pcm_file_names = ['ct53-resin_h-T_data_55frac.csv']
# pcm_file_names = ['ct53-resin_h-T_data_45frac.csv']

# setpoint_temps_f = [140]
setpoint_temps_f = [140]
setpoint_temps_c = [
    (setpoint_temp - 32) * (5 / 9) for setpoint_temp in setpoint_temps_f
]

case_run_name = "case01"

# tank_volume_gal = [40,50, 65]
tank_volume_gal = [40,50]


# vol_fract = 0.00000001  # 1.540e-06 kg
# vol_fract = 0.0001  # 1.540e-02 kg
# vol_fract = 0.5  # 7.700e+01 kg
# vol_fracs = [0.66]
# vol_fracs = [0.74]
# vol_fracs = [0.26]
vol_fracs = [0.13]

# pcm_vol_fractions = [{i: vol_fract for i in range(1, n + 1)} for n in range(1, num_nodes + 1)]
# pcm_vol_fractions = [
#                     {7: vol_fract},
#                     ]

# pcm_vol_fractions = [
#                     {12: vol_fract},
#                     ]
pcm_vol_fractions = []
for vol_fract in vol_fracs:
    pcm_vol_fractions.append(
        {node: vol_fract for node in range(start_node, end_node + 1)}
    )



load_profile = "2.00gpm30min_0gpm180min_cycling.csv"
load_profile = "2.00gpm30min_0gpm600min_cycling.csv"
load_profile = "2.00gpm120min_0gpm600min_cycling.csv"
load_profile = "MediumUseL.csv"
load_profile = "2.00gpm_1200minStartIdle_2cycles_30minDraw_240minOff_0minEndIdle.csv"
load_profile = "2.00gpm_1200minStartIdle_ 2cycles_30minDraw_240minOff_0minEndIdle_Single_draw.csv"
load_profiles = ["MediumUseL.csv", "2.00gpm30min_0gpm600min_cycling.csv"]
load_profile = "net_flow_90023_220day.csv"

def convert_dict_to_name(dict):
    # Check if all values are the same
    values = list(dict.values())
    if len(values) > 0 and all(v == values[0] for v in values):
        # Check if keys are sequential integers
        keys = sorted(dict.keys())
        if all(isinstance(k, int) for k in keys) and keys == list(
            range(min(keys), max(keys) + 1)
        ):
            # Return the compact format
            return f"{values[0]}_pcm{min(keys)}-{max(keys)}"

    # Fallback to original format if conditions aren't met
    return "".join([str(key) + "_" + str(value) + "_" for key, value in dict.items()])


def add_pcm_model(default_args, name, pcm_vol_fractions, pcm_properties):
    
    # default_args['model_class'] = TankWithMultiPCMExternal
    default_args['model_class'] = TankWithMultiPCM
    default_args['water_nodes'] = 12
    default_args['Water Tank'] = {
        'pcm_node_vol_fractions': pcm_vol_fractions,
        'pcm_properties': pcm_properties,
    }

    default_args["name"] = name
    return default_args

default_args = {
    "start_time": dt.datetime(2018, 1, 1, 0, 0),  # year, month, day, hour, minute
    "time_res": dt.timedelta(minutes=1),
    "verbosity": 9,
    "save_results": None,  # if True, must specify output_path # None Merges the simulator results into 1 file
    "output_path": '../OCHRE_output/OCHRE_results/results/',
    "name": "ZDefault_ElectricResistanceWaterHeater",
    "Mixed Delivery Temperature (C)": convert(125, "degF", "degC"),
    # "schedule_input_file": load_profile,
}


DEFAULT_SCENARIO = {
    "name": case_run_name,
    "nodes": list(range(start_node, end_node + 1)),
    "sa_ratios": list(sa_ratios),
    "h_values": list(h_values),
    "setpoint": list(setpoint_temps_f),
    "volume": list(tank_volume_gal),
    "pcm_file_name": list(pcm_file_names),
    "volume_fraction": list(vol_fracs),
    "heater_type": "heatpump",
    "load_profile": None,
    "default_parameters": {},
    "pcm_defaults": {},
    "tests": None,
}

DEFAULT_TESTS = [
    # {
    #     "test_type": "FHR",
    #     "duration_hours": 48,
    #     "time_interval_seconds": 0.5,
    #     "load_profile": None,
    # },
    {
            "test_type": "UEF",
            "duration_hours": 48,
            "time_interval_seconds": 30,
            "load_profile": "MediumUseL.csv",
        }
]


DEFAULT_RUNNER_CONFIG = {
    "results_root": "../OCHRE_output/OCHRE_results/scenarios",
    "plot_output_folder": "../OCHRE_output/results",
    "generate_plots": False,
    "skip_showing_plots": True,
    "run_default_no_pcm": True,
    "default_no_pcm_type": "heatpump",
    "processes": None,
    "image_scale": 4,
    "baseline_file": "../OCHRE_output/zDefault_No_PCM_HeatPump_setpoint-140F_40gal_0.csv",
    "export_draw_outputs": False,
    "draw_outputs_csv": "../OCHRE_output/OCHRE_results/results_csv/results.csv",
    "tests": DEFAULT_TESTS,
    "scenarios": [DEFAULT_SCENARIO],
}


def _as_list(value, field_name):
    """Return a scalar or JSON array as a non-empty list."""
    if value is None:
        raise ValueError(f"Scenario field '{field_name}' is required")
    values = value if isinstance(value, (list, tuple)) else [value]
    if not values:
        raise ValueError(f"Scenario field '{field_name}' cannot be empty")
    return list(values)


def _format_number(value):
    return f"{float(value):.0f}"


def _safe_folder_token(value):
    value = str(value).replace("%", "pct")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value)
    return value.strip("-.") or "value"


def _value_summary(values, suffix=""):
    return "-".join(_safe_folder_token(_format_number(value)) for value in values) + suffix


def _merge_dicts(base, overrides):
    """Deep-merge JSON overrides into a copied dictionary."""
    merged = copy.deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def normalize_test_config(raw_test):
    """Normalize one configured water-heater test definition."""
    if isinstance(raw_test, str):
        raw_test = {"test_type": raw_test}
    if not isinstance(raw_test, dict):
        raise ValueError("Each entry in 'tests' must be a string or JSON object")

    test_type = str(raw_test.get("test_type", raw_test.get("name", ""))).upper()
    if test_type not in {"FHR", "UEF"}:
        raise ValueError("Test type must be 'FHR' or 'UEF'")

    defaults = {
        "FHR": {
            "duration_hours": 48,
            "time_interval_seconds": 0.5,
            "load_profile": None,
        },
        "UEF": {
            "duration_hours": 48,
            "time_interval_seconds": 30,
            "load_profile": "MediumUseL.csv",
        },
    }[test_type]
    duration_hours = float(raw_test.get("duration_hours", defaults["duration_hours"]))
    time_interval_seconds = float(
        raw_test.get("time_interval_seconds", defaults["time_interval_seconds"])
    )
    load_profile_for_test = raw_test.get("load_profile", defaults["load_profile"])

    if duration_hours <= 0:
        raise ValueError(f"{test_type} duration_hours must be positive")
    if time_interval_seconds <= 0:
        raise ValueError(f"{test_type} time_interval_seconds must be positive")
    if test_type == "UEF" and not load_profile_for_test:
        raise ValueError("UEF test requires a load_profile")

    return {
        "test_type": test_type,
        "duration_hours": duration_hours,
        "time_interval_seconds": time_interval_seconds,
        "load_profile": load_profile_for_test,
    }


def normalize_test_configs(raw_tests):
    tests = [normalize_test_config(test) for test in _as_list(raw_tests, "tests")]
    test_types = [test["test_type"] for test in tests]
    if len(test_types) != len(set(test_types)):
        raise ValueError("Configured test types must be unique")
    return tests


def normalize_scenario(raw_scenario):
    """Normalize and validate one JSON scenario into runner-ready values."""
    if not isinstance(raw_scenario, dict):
        raise ValueError("Each entry in 'scenarios' must be a JSON object")

    raw = _merge_dicts(DEFAULT_SCENARIO, raw_scenario)
    name = str(raw.get("name", "")).strip()
    if not name:
        raise ValueError("Each scenario must have a non-empty 'name'")

    nodes = [int(node) for node in _as_list(raw["nodes"], "nodes")]
    if any(node < 0 for node in nodes):
        raise ValueError(f"Scenario '{name}' has a negative PCM node")

    sa_values = [float(value) for value in _as_list(raw["sa_ratios"], "sa_ratios")]
    h_values_for_scenario = [float(value) for value in _as_list(raw["h_values"], "h_values")]
    setpoints_f = [float(value) for value in _as_list(raw["setpoint"], "setpoint")]
    volumes_gal = [float(value) for value in _as_list(raw["volume"], "volume")]
    pcm_names = [str(value) for value in _as_list(raw["pcm_file_name"], "pcm_file_name")]
    volume_fractions = [
        float(value) for value in _as_list(raw["volume_fraction"], "volume_fraction")
    ]

    if any(value <= 0 for value in sa_values):
        raise ValueError(f"Scenario '{name}' must use positive sa_ratios")
    if any(value <= 0 for value in h_values_for_scenario):
        raise ValueError(f"Scenario '{name}' must use positive h_values")
    if any(value <= 0 for value in volumes_gal):
        raise ValueError(f"Scenario '{name}' must use positive tank volumes")
    if any(value < 0 or value > 1 for value in volume_fractions):
        raise ValueError(
            f"Scenario '{name}' volume_fraction values must be between 0 and 1"
        )
    if any(not value.strip() for value in pcm_names):
        raise ValueError(f"Scenario '{name}' contains an empty pcm_file_name")

    heater_type = str(raw.get("heater_type", "heatpump")).lower()
    if heater_type not in {"heatpump", "electric"}:
        raise ValueError(
            f"Scenario '{name}' heater_type must be 'heatpump' or 'electric'"
        )

    return {
        "name": name,
        "nodes": nodes,
        "sa_ratios": sa_values,
        "h_values": h_values_for_scenario,
        "setpoints_f": setpoints_f,
        "volumes_gal": volumes_gal,
        "pcm_file_names": pcm_names,
        "volume_fractions": volume_fractions,
        "heater_type": heater_type,
        "load_profile": raw.get("load_profile"),
        "default_parameters": raw.get("default_parameters", {}),
        "pcm_defaults": raw.get("pcm_defaults", {}),
        "baseline_file": raw.get("baseline_file"),
        "run_default_no_pcm": raw.get("run_default_no_pcm"),
        "default_no_pcm_type": raw.get("default_no_pcm_type"),
        "folder_name": raw.get("folder_name"),
        "tests": (
            normalize_test_configs(raw["tests"])
            if raw.get("tests") is not None
            else None
        ),
    }


def load_runner_config(config_path=None):
    """Load the runner JSON, or return the current script defaults."""
    configured_scenarios = None
    if config_path is None:
        config = copy.deepcopy(DEFAULT_RUNNER_CONFIG)
        config_base = RUNNER_ROOT
    else:
        config_path = Path(config_path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as config_file:
            loaded = json.load(config_file)
        if isinstance(loaded, list):
            loaded = {"scenarios": loaded}
        if not isinstance(loaded, dict):
            raise ValueError("Runner config must be a JSON object or scenario array")
        config = _merge_dicts(DEFAULT_RUNNER_CONFIG, loaded)
        configured_scenarios = loaded.get("scenarios", loaded.get("cases"))
        config_base = RUNNER_ROOT

    scenario_entries = (
        configured_scenarios
        if configured_scenarios is not None
        else config.get("scenarios", config.get("cases"))
    )
    if not scenario_entries:
        raise ValueError("Runner config must contain at least one scenario")
    config["tests"] = normalize_test_configs(config.get("tests", DEFAULT_TESTS))
    config["scenarios"] = [normalize_scenario(entry) for entry in scenario_entries]
    scenario_names = [scenario["name"] for scenario in config["scenarios"]]
    if len(scenario_names) != len(set(scenario_names)):
        raise ValueError("Scenario names must be unique")
    config["config_base"] = config_base
    return config


def resolve_runner_path(value, base_dir):
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()

    # All relative runner paths are rooted at the repository directory, not
    # at bin/ or whichever directory the command happened to be launched from.
    return (Path(base_dir) / path).resolve()


def scenario_folder_name(scenario):
    """Create a readable, filesystem-safe summary folder name."""
    pcm_summary = "-".join(_safe_folder_token(Path(name).stem) for name in scenario["pcm_file_names"])
    volume_fraction_summary = "-".join(
        _safe_folder_token(_format_number(value * 100))
        for value in scenario["volume_fractions"]
    )
    return "__".join(
        [
            _safe_folder_token(scenario["folder_name"] or scenario["name"]),
            f"nodes{_value_summary(scenario['nodes'])}",
            f"sa{_value_summary(scenario['sa_ratios'])}",
            f"h{_value_summary(scenario['h_values'])}",
            f"sp{_value_summary(scenario['setpoints_f'], 'F')}",
            f"vol{_value_summary(scenario['volumes_gal'], 'gal')}",
            f"pcm-{pcm_summary}",
            f"vf{volume_fraction_summary}pct",
        ]
    )


SCENARIO_CACHE_VERSION = 2
SCENARIO_CACHE_FILE = ".scenario_cache.json"


def scenario_cache_payload(scenario, run_default_no_pcm, no_pcm_type, tests=None):
    """Return the decoded scenario configuration used to identify its cache."""
    return {
        "cache_version": SCENARIO_CACHE_VERSION,
        "scenario": copy.deepcopy(scenario),
        "run_default_no_pcm": bool(run_default_no_pcm),
        "default_no_pcm_type": str(no_pcm_type).lower(),
        "tests": copy.deepcopy(tests or scenario.get("tests") or DEFAULT_TESTS),
    }


def _cache_identity_matches(scenario_folder, cache_payload):
    """Check the stored JSON identity when this folder has one.

    Existing result folders from before the cache metadata was added are still
    eligible for reuse when their exact expected result files are present.
    """
    cache_path = scenario_folder / SCENARIO_CACHE_FILE
    if not cache_path.is_file():
        return True

    try:
        with cache_path.open("r", encoding="utf-8") as cache_file:
            stored_payload = json.load(cache_file)
    except (OSError, json.JSONDecodeError):
        return False

    return stored_payload == cache_payload


def _write_scenario_cache(scenario_folder, cache_payload):
    """Atomically record a completed scenario's decoded configuration."""
    cache_path = scenario_folder / SCENARIO_CACHE_FILE
    temporary_path = scenario_folder / f"{SCENARIO_CACHE_FILE}.{os.getpid()}.tmp"
    try:
        with temporary_path.open("w", encoding="utf-8") as cache_file:
            json.dump(cache_payload, cache_file, indent=2, sort_keys=True)
            cache_file.write("\n")
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _task_result_is_complete(task):
    """Return whether a task has a non-empty result and OCHRE completion mark."""
    result_path = task["result_path"]
    complete_path = task["complete_path"]
    return (
        result_path.is_file()
        and result_path.stat().st_size > 0
        and complete_path.is_file()
    )


def cached_scenario_tasks(tasks, scenario_folder, cache_payload):
    """Return tasks reusable from this scenario folder.

    A mismatched cache identity invalidates the whole folder. With no metadata
    file, exact task result paths provide backward-compatible cache discovery.
    """
    if not _cache_identity_matches(scenario_folder, cache_payload):
        return []
    return [task for task in tasks if _task_result_is_complete(task)]


def scenario_results_are_complete(tasks):
    """Return whether every expected task result is present and complete."""
    return bool(tasks) and all(_task_result_is_complete(task) for task in tasks)


def import_water_heating_schedule(schedule_file, duration):
    """Load the requested water-use columns for the configured duration."""
    schedule_path = Path(schedule_file).expanduser()
    if not schedule_path.is_absolute():
        input_file_path = RUNNER_ROOT / "ochre" / "defaults" / "Input Files" / schedule_path
        runner_path = RUNNER_ROOT / schedule_path
        schedule_path = input_file_path if input_file_path.is_file() else runner_path
    schedule_path = schedule_path.resolve()
    if not schedule_path.is_file():
        raise FileNotFoundError(f"Water-heating load profile not found: {schedule_path}")

    raw = pd.read_csv(schedule_path)
    water_column_names = {
        "hot_water_fixtures": "Water Fixtures (L/min)",
        "hot_water_clothes_washer": "Clothes Washer (L/min)",
        "hot_water_dishwasher": "Dishwasher (L/min)",
    }
    available_columns = [column for column in water_column_names if column in raw.columns]
    if available_columns:
        hot_water_schedule = raw.loc[:, available_columns].rename(columns=water_column_names)
    elif len(raw.columns) == 1:
        # Support legacy single-column profiles with either a header or raw values.
        hot_water_schedule = raw.copy()
        hot_water_schedule.columns = ["Water Fixtures (L/min)"]
    else:
        raise ValueError(
            f"Load profile {schedule_path} has no supported hot_water_* columns"
        )

    hot_water_schedule = hot_water_schedule.apply(pd.to_numeric, errors="coerce").fillna(0)
    required_minutes = int(np.ceil(duration.total_seconds() / 60))
    if len(hot_water_schedule) < required_minutes:
        raise ValueError(
            f"Load profile {schedule_path} has {len(hot_water_schedule)} minute(s), "
            f"but the configured test requires {required_minutes}"
        )
    hot_water_schedule = hot_water_schedule.iloc[:required_minutes].copy()
    hot_water_schedule.index = pd.date_range(
        default_args["start_time"], periods=len(hot_water_schedule), freq="1min"
    )
    return hot_water_schedule


def create_profile_water_schedule(hot_water_schedule, setpoint_temp):
    """Add UEF water-use columns to the common water-heater schedule."""
    fixtures = hot_water_schedule.get(
        "Water Fixtures (L/min)",
        pd.Series(0.0, index=hot_water_schedule.index),
    )
    schedule = create_water_schedule(
        withdraw_rate_lpm=fixtures,
        setpoint_default=setpoint_temp,
        no_heating_during_draw=False,
        times=hot_water_schedule.index,
    )
    for column in ("Clothes Washer (L/min)", "Dishwasher (L/min)"):
        if column in hot_water_schedule:
            schedule[column] = hot_water_schedule[column]
    return schedule
        
    
def simulate_first_hour_test(wh, enable_first_hour_test=True, disable_heating_during_draw=False, first_hour_duration=60, draw_rate_gpm=3, allow_setpoint_start=False, hot_water_temp_f=110, setpoint_temp_f=140):
    
    test_active = False
    test_completed = False
    test_initialized = False
    test_timer = 0.0
    draw_active = False
    total_gallons_delivered = 0.0
    final_draw_triggered = False  # New flag to track if final draw has been triggered

    # These constants would typically be defined globally or passed in.
    # For the function to run stand-alone, they'd need to be defined here or passed.
    # Assuming GAL_TO_L and convert are available in the scope this function is called from.
    # e.g. GAL_TO_L = 3.78541
    # e.g. from some_utils import convert 

    times = wh.sim_times
    prev_t = None

    setpoint_temp = convert(setpoint_temp_f, 'degF', 'degC')
    hot_water_temp = convert(hot_water_temp_f, 'degF', 'degC')
    
    def pcm_label():
        model = getattr(wh, "model", None)
        pcm_props = getattr(model, "pcm_properties", None)
        if isinstance(pcm_props, dict):
            try:
                return f"sa_ratio: {pcm_props['sa_ratio']:.2f}, ha: {pcm_props['h_conv']:.2f}"
            except Exception:
                return "pcm=unknown"
        return "Default no PCM"

    for t_idx, t in enumerate(times):
        control_signal = {}

        if prev_t is not None:
            delta_min = (t - prev_t).total_seconds() / 60.0
        else:
            delta_min = 0.0
        prev_t = t

        if enable_first_hour_test:
            if t_idx == 1:
                test_initialized = True

            if test_initialized and not test_active and not test_completed:
                if wh.mode == 'Off':
                    test_active = True
                    test_timer = float(first_hour_duration)
                    draw_active = True
                    control_signal = {
                        'Water Fixtures (L/min)': draw_rate_gpm * GAL_TO_L
                    }
                    if disable_heating_during_draw:
                        control_signal['Water Heating Setpoint (C)'] = 5
                    print(f"[{t}] ({pcm_label()})  -> FIRST-HOUR test STARTED, drawing {draw_rate_gpm} gpm")

            elif test_active:
                test_timer -= delta_min

                if wh.mode == 'Off' and not draw_active:
                    draw_active = True
                    # control_signal will be set in accumulation step if draw_active remains true
                    print(f"[{t}] ({pcm_label()}) -> MODE=Off, RESTARTING draw")

                # Check if we've reached the test duration and should trigger final draw
                if test_timer <= 0 and not final_draw_triggered:
                    final_draw_triggered = True
                    # If not already drawing, start the draw
                    if not draw_active:
                        draw_active = True
                        print(f"[{t}] ({pcm_label()}) -> FINAL DRAW: Timer elapsed ({test_timer:.2f}), initiating final draw")
                    else:
                        print(f"[{t}] ({pcm_label()}) -> FINAL DRAW: Timer elapsed ({test_timer:.2f}), draw already active, continuing")
                
                if draw_active and wh.model.outlet_temp < hot_water_temp:
                    draw_active = False
                    control_signal = {} # Explicitly stop draw signal for this step
                    print(f"[{t}] ({pcm_label()}) -> temp fell ({wh.model.outlet_temp:.1f}C vs {hot_water_temp:.1f}C limit), STOPPING draw")

                if (allow_setpoint_start and not draw_active and wh.model.outlet_temp >= setpoint_temp):
                    draw_active = True
                    # control_signal will be set in accumulation step
                    print(f"[{t}] ({pcm_label()}) -> reached setpoint ({wh.model.outlet_temp:.1f}C), STARTING draw")

                if draw_active:
                    control_signal['Water Fixtures (L/min)'] = draw_rate_gpm * GAL_TO_L
                    if disable_heating_during_draw:
                        control_signal['Water Heating Setpoint (C)'] = 5
                    total_gallons_delivered += draw_rate_gpm * delta_min

                # Only complete the test if the timer is up AND 
                # either: 1) final draw has been completed (not active) or 2) outlet temp fell below limit
                if test_timer <= 0 and final_draw_triggered and not draw_active:
                    test_active = False
                    test_completed = True
                    print(f"[{t}] ({pcm_label()}) -> TEST COMPLETE: delivered {total_gallons_delivered:.2f} gallons")
                    # We don't break immediately to ensure the final state is properly updated
                    # Instead, we'll break at the end of this iteration
        
        _ = wh.update(schedule_inputs=control_signal)

        # If test completed inside the 'if enable_first_hour_test' block, break from the outer loop
        if test_completed:
            break 

    return total_gallons_delivered, wh 

def create_water_schedule(
    withdraw_rate_lpm=None,
    withdraw_rate_gpm=3.0,
    setpoint_default=51.667,
    deadband_default=5.56,
    zone_temp_c=19.722222,
    zone_wet_bulb_temp_c=15.0,
    mains_temp_c=14.4444,
    start_idle_time_min=0,
    draw_time_min=30,
    off_time_min=240,
    draw_repeats=2,
    end_idle_time_min=0,
    no_heating_during_draw=False,
    times=pd.date_range(
        dt.datetime(2018, 1, 1, 0, 0),
        dt.datetime(2018, 1, 1, 0, 0) + dt.timedelta(days=2)+ dt.timedelta(minutes=1),
        freq=dt.timedelta(minutes=1),
        inclusive="left",
    )
):
    """
    Create a water schedule DataFrame with adjustable parameters for all variables.
    
    Parameters:
    -----------
    withdraw_rate_gpm : float
        Water withdrawal rate in gal/min during draw periods
    setpoint_default : float
        Default water heating setpoint temperature in Celsius
    deadband_default : float
        Default water heating deadband in Celsius
    zone_temp_c : float
        Zone temperature in Celsius
    zone_wet_bulb_temp_c : float
        Zone wet bulb temperature in Celsius
    mains_temp_c : float
        Mains water temperature in Celsius
    start_idle_time_min : int
        Duration of initial idle period in minutes
    draw_time_min : int
        Duration of each draw period in minutes
    off_time_min : int
        Duration of each off period in minutes
    draw_repeats : int
        Number of draw cycles to repeat
    end_idle_time_min : int
        Duration of final idle period in minutes
    no_heating_during_draw : bool
        If True, the setpoint will drop below mains temperature during water draws
        to prevent heating. If False, setpoint remains constant.
    times : pd.DatetimeIndex
        Datetime index for the schedule (default is 2 days starting at January 1, 2018)
        
    Returns:
    --------
    pd.DataFrame
        Water schedule with all variables
    """
    GAL_TO_L = 3.78541
    
    withdraw_rate = withdraw_rate_gpm * GAL_TO_L
    
    total_minutes = len(times)
    
    # Initialize arrays for each variable
    water_withdraw = []
    setpoint_temp = []
    
    # Define setpoint temperature during draw (slightly below mains temperature)
    draw_setpoint = mains_temp_c - 2.0  # 2°C below mains temperature
        
    if withdraw_rate_lpm is None:
        # Build the pattern for water withdrawal and setpoint temperatures
        # 1. Start idle period
        water_withdraw.extend([0] * start_idle_time_min)
        setpoint_temp.extend([setpoint_default] * start_idle_time_min)
        
        # 2. Draw profile repeated as many times as specified
        for _ in range(draw_repeats):
            # Draw period (on) - set withdrawal rate and possibly lower setpoint
            water_withdraw.extend([withdraw_rate] * draw_time_min)
            
            # Set setpoint based on no_heating_during_draw flag
            if no_heating_during_draw:
                setpoint_temp.extend([draw_setpoint] * draw_time_min)
            else:
                setpoint_temp.extend([setpoint_default] * draw_time_min)
            
            # Off period (idle) - no withdrawal and restore default setpoint
            water_withdraw.extend([0] * off_time_min)
            setpoint_temp.extend([setpoint_default] * off_time_min)
        
        # 3. End idle period
        water_withdraw.extend([0] * end_idle_time_min)
        setpoint_temp.extend([setpoint_default] * end_idle_time_min)
        
        # Ensure the patterns reach the required length
        if len(water_withdraw) >= total_minutes:
            water_withdraw = water_withdraw[:total_minutes]
            setpoint_temp = setpoint_temp[:total_minutes]
        else:
            # If the constructed patterns are shorter than required, pad the end
            remaining = total_minutes - len(water_withdraw)
            water_withdraw.extend([0] * remaining)
            setpoint_temp.extend([setpoint_default] * remaining)
    else:
        water_withdraw = list(withdraw_rate_lpm)
        setpoint_temp = [setpoint_default] * total_minutes
        # If no_heating_during_draw is True, set the setpoint to draw_setpoint 
        # during periods when water is being withdrawn (non-zero withdrawal rate)
        if no_heating_during_draw:
            for i in range(total_minutes):
                if water_withdraw[i] > 0:
                    setpoint_temp[i] = draw_setpoint
        
    # Fill in other variables with their default values
    deadband = [deadband_default] * total_minutes
    zone_temp = [zone_temp_c] * total_minutes
    zone_wet_bulb = [zone_wet_bulb_temp_c] * total_minutes
    mains_temp = [mains_temp_c] * total_minutes
    
    # Create the DataFrame with all variables
    schedule = pd.DataFrame(
        {
            "Water Fixtures (L/min)": water_withdraw,
            "Water Heating Setpoint (C)": setpoint_temp,
            "Water Heating Deadband (C)": deadband,
            "Zone Temperature (C)": zone_temp,
            "Zone Wet Bulb Temperature (C)": zone_wet_bulb,
            "Mains Temperature (C)": mains_temp,
        },
        index=times,
    )
    
    return schedule

def f_to_c(f):
    return (f - 32) * 5 / 9

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

def calculate_net_water_temp(df):
    water_temp_column = ["Hot Water Average Temperature (C)"]
    valid_columns = [col for col in water_temp_column if col in df.columns]

    if not valid_columns:
        return 0

    # Compute the average of the first and last row for all available columns
    average_start_temp = df[valid_columns].iloc[0].mean()
    average_end_temp = df[valid_columns].iloc[-1].mean()

    return average_end_temp - average_start_temp




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


def _simulation_test_settings(configured_args):
    """Remove runner-only test settings before constructing OCHRE equipment."""
    simulation_args = copy.deepcopy(configured_args)
    test_type = str(simulation_args.pop("_test_type", "FHR")).upper()
    duration = dt.timedelta(
        hours=float(simulation_args.pop("_test_duration_hours", 48))
    )
    time_res = dt.timedelta(
        seconds=float(simulation_args.pop("_test_time_interval_seconds", 0.5))
    )
    load_profile_for_test = simulation_args.pop("schedule_input_file", None)
    return simulation_args, test_type, duration, time_res, load_profile_for_test


def _create_test_schedule(test_type, load_profile_for_test, duration, setpoint_temp):
    if test_type == "FHR":
        return create_water_schedule(
            setpoint_default=setpoint_temp,
            withdraw_rate_gpm=0,
            no_heating_during_draw=False,
        )
    if not load_profile_for_test:
        raise ValueError(f"{test_type} test requires a water-heating load profile")
    hot_water_schedule = import_water_heating_schedule(load_profile_for_test, duration)
    return create_profile_water_schedule(hot_water_schedule, setpoint_temp)


def run_water_heater_electric(default_args, setpoint_temp, tank_volume):
    simulation_args, test_type, duration, time_res, profile = _simulation_test_settings(
        default_args
    )
    schedule = _create_test_schedule(test_type, profile, duration, setpoint_temp)
    equipment_args = {
        # Equipment parameters
        # "Setpoint Temperature (C)": 14.4444,
        "Setpoint Temperature (C)": setpoint_temp,
        "Tank Volume (L)": tank_volume * GAL_TO_L * 0.9,
        "Tank Height (m)": 1.22,
        "UA (W/K)": 2.17,
        # "UA (W/K)": 1e-9, 
        # "schedule": schedule,
        "Capacity (W)": 4500,
        "water_nodes": 12,
        **simulation_args,
        "duration": duration,
        "time_res": time_res,
    }

    # Initialize equipment
    wh = ElectricResistanceWaterHeater(schedule=schedule, **equipment_args,)

    if test_type == "FHR":
        hot_water_output_gallons, wh = simulate_first_hour_test(wh, enable_first_hour_test=True, disable_heating_during_draw=False, first_hour_duration=60, draw_rate_gpm=3, allow_setpoint_start=False, hot_water_temp_f=110, setpoint_temp_f=140)
        df = wh.finalize()
    else:
        df = wh.simulate()

    uef = calculate_uef(df, equipment_args['name'])
    
    return uef

    # print(df.head())



def run_water_heater_heatpump(default_args, setpoint_temp, tank_volume):
    simulation_args, test_type, duration, time_res, profile = _simulation_test_settings(
        default_args
    )
    schedule = _create_test_schedule(test_type, profile, duration, setpoint_temp)
    equipment_args = {
        "verbosity": 9,  # required to get setpoint and deadband in results
        "save_results": None,  # if True, must specify output_path None Merges the simulator results into 1 file
        "output_path": '../OCHRE_output/OCHRE_results/results/',
        # "Setpoint Temperature (C)": 14.4444,
        "Setpoint Temperature (C)": setpoint_temp,
        "Tank Volume (L)": tank_volume * GAL_TO_L * 0.9,
        "Tank Height (m)": 1.22,
        "UA (W/K)": 2.17,
        # "UA (W/K)": 1e-9,
        "HPWH COP (-)": 4.5,
        **simulation_args,
        "duration": duration,
        "time_res": time_res,
        "hp_only_mode": True
    }

    deadband_default = schedule['Water Heating Deadband (C)'].iloc[0]
    # Initialize equipment
    hpwh = HeatPumpWaterHeater(schedule=schedule, **equipment_args)

    if test_type == "FHR":
        hot_water_output_gallons, hpwh = simulate_first_hour_test(hpwh, enable_first_hour_test=True, disable_heating_during_draw=False, first_hour_duration=60, draw_rate_gpm=3, allow_setpoint_start=False, hot_water_temp_f=110, setpoint_temp_f=140)
        df = hpwh.finalize()
    else:
        df = hpwh.simulate()
    
    uef = calculate_uef(df, equipment_args['name'])
    
    return uef

    # # print(df.head())
    # cols_to_plot = [
    #     "Hot Water Outlet Temperature (C)",
    #     "Hot Water Average Temperature (C)",
    #     "Water Heating Deadband Upper Limit (C)",
    #     "Water Heating Deadband Lower Limit (C)",
    #     "Water Heating Electric Power (kW)",
    #     "Hot Water Unmet Demand (kW)",
    #     "Hot Water Delivered (L/min)",
    # ]
    # df.loc[:, cols_to_plot].plot()
    # CreateFigures.plt.show()


main_results_folder = "../OCHRE_output/OCHRE_results/results/"
graphing_results_folder = "../OCHRE_output/results/"


def move_results(results_folder, graphings_results_folder, num_files_to_move):
    # List CSV files that do not have '_schedule' in the name.
    files = [
        f
        for f in os.listdir(results_folder)
        if f.endswith(".csv") and "_schedule" not in f
    ]

    # Sort files by modification time.
    files.sort(key=lambda x: os.path.getmtime(os.path.join(results_folder, x)))

    # Select the last num_files_to_move files.
    files_to_move = files[-num_files_to_move:]

    # Clear or create the graphings_results_folder.
    if os.path.exists(graphings_results_folder):
        shutil.rmtree(graphings_results_folder)
    os.mkdir(graphings_results_folder)

    # Move each selected file.
    for file in files_to_move:
        source = os.path.join(results_folder, file)
        destination = os.path.join(graphings_results_folder, file)
        os.rename(source, destination)


def run_water_heater_process(
    default_args, tank_volume, setpoint_temp, filename, submission_time
):
    """
    Function wrapper to execute run_water_heater in a separate process and measure:
    - Queue wait time (time from submission to actual start)
    - Active simulation processing
    - Total time (waiting + processing)
    """
    actual_start_time = time.perf_counter()
    title = default_args.get('name', 'Name_Not_Specified')
    is_heatpump = default_args.get('is_heatpump', False)
    wait_time = actual_start_time - submission_time
    print(
        f"{CYAN}Starting simulation process: {filename}; waited {wait_time:.2f} sec in queue{RESET}"
    )

    # Run the actual simulation
    result = None
    simulation_error = None
    try:
        if is_heatpump:
            result = run_water_heater_heatpump(default_args, setpoint_temp, tank_volume)
        else:
            result = run_water_heater_electric(default_args, setpoint_temp, tank_volume)
    except Exception as e:
        simulation_error = e
        print(
            f"{RED}{title}Error in simulation process run_water_heater_process: {str(e)}{RESET}"
        )
    end_time = time.perf_counter()
    sim_duration = end_time - actual_start_time  # time spent in simulation function
    total_duration = end_time - submission_time  # includes waiting time

    print(
        f"{CYAN}Process {filename}: {title} completed in {sim_duration:.2f} sec (simulation), "
        f"total {total_duration:.2f} sec (including wait){RESET}"
    )

    if simulation_error is not None:
        raise simulation_error

    return title, result, sim_duration, wait_time, total_duration


def effective_run_default_no_pcm(scenario, config):
    """Resolve the scenario override against the runner-level default."""
    configured = scenario["run_default_no_pcm"]
    return config["run_default_no_pcm"] if configured is None else configured


def effective_scenario_tests(scenario, config):
    """Resolve an optional scenario test override against config-level tests."""
    return scenario.get("tests") or config["tests"]


def scenario_task_count(scenario, run_default_no_pcm, tests=None):
    """Count simulation cases generated by one normalized scenario."""
    tests = tests or scenario.get("tests") or DEFAULT_TESTS
    setpoint_volume_pairs = len(scenario["setpoints_f"]) * len(
        scenario["volumes_gal"]
    )
    pcm_grid_count = (
        len(scenario["pcm_file_names"])
        * len(scenario["volume_fractions"])
        * len(scenario["sa_ratios"])
        * len(scenario["h_values"])
    )
    baseline_count = int(bool(run_default_no_pcm))
    return len(tests) * setpoint_volume_pairs * (baseline_count + pcm_grid_count)


def build_scenario_tasks(
    scenario,
    scenario_folder,
    run_default_no_pcm,
    no_pcm_type,
    tests=None,
):
    """Build all baseline and PCM grid-point tasks for one scenario."""
    tests = tests or scenario.get("tests") or DEFAULT_TESTS
    scenario_base_args = copy.deepcopy(default_args)
    scenario_base_args.update(scenario["default_parameters"])
    scenario_base_args["output_path"] = str(scenario_folder)
    # Results are required by the comparison pipeline, regardless of the
    # historical default in this script.
    scenario_base_args["save_results"] = None
    scenario_base_args.pop("schedule_input_file", None)

    pcm_defaults = _merge_dicts(DEFAULT_PCM_PROPERTIES, scenario["pcm_defaults"])
    tasks = []
    case_paths_by_group = {}
    baseline_paths_by_group = {}
    task_index = 0

    def add_task(args, tank_volume, setpoint_temp_c, filename, group, test_type):
        tasks.append(
            {
                "args": args,
                "tank_volume": tank_volume,
                "setpoint_temp_c": setpoint_temp_c,
                "filename": filename,
                "group": group,
                "test_type": test_type,
                "result_path": scenario_folder / f"{filename}.csv",
                "complete_path": scenario_folder / f"{filename}_complete",
                "draw_output_path": scenario_folder / f"{filename}_draw_outputs.csv",
            }
        )

    def apply_test_config(args, test):
        args["_test_type"] = test["test_type"]
        args["_test_duration_hours"] = test["duration_hours"]
        args["_test_time_interval_seconds"] = test["time_interval_seconds"]
        if test["load_profile"] is None:
            args.pop("schedule_input_file", None)
        else:
            args["schedule_input_file"] = test["load_profile"]

    for tank_volume in scenario["volumes_gal"]:
        for setpoint_temp_f in scenario["setpoints_f"]:
            setpoint_temp_c = convert(setpoint_temp_f, "degF", "degC")
            group = (float(setpoint_temp_f), float(tank_volume))

            for test in tests:
                test_type = test["test_type"]
                if run_default_no_pcm:
                    current_default_args = copy.deepcopy(scenario_base_args)
                    apply_test_config(current_default_args, test)
                    current_default_args["is_heatpump"] = no_pcm_type == "heatpump"
                    no_pcm_title_base = (
                        "zDefault_No_PCM_HeatPump"
                        if no_pcm_type == "heatpump"
                        else "zDefault_No_PCM_Electric"
                    )
                    no_pcm_title = (
                        f"{no_pcm_title_base}_setpoint-{setpoint_temp_f:g}F_"
                        f"{tank_volume:g}gal_{task_index}_{test_type}"
                    )
                    current_default_args["name"] = no_pcm_title
                    add_task(
                        current_default_args,
                        tank_volume,
                        setpoint_temp_c,
                        no_pcm_title,
                        group,
                        test_type,
                    )
                    if test_type == "FHR":
                        baseline_paths_by_group[group] = scenario_folder / f"{no_pcm_title}.csv"
                    task_index += 1

                for pcm_file_name in scenario["pcm_file_names"]:
                    for volume_fraction in scenario["volume_fractions"]:
                        pcm_vol_fraction = {
                            node: volume_fraction for node in scenario["nodes"]
                        }
                        for sa_ratio in scenario["sa_ratios"]:
                            for h_value in scenario["h_values"]:
                                current_default_args = copy.deepcopy(scenario_base_args)
                                apply_test_config(current_default_args, test)
                                current_default_args["is_heatpump"] = (
                                    scenario["heater_type"] == "heatpump"
                                )
                                current_pcm_properties = copy.deepcopy(pcm_defaults)
                                current_pcm_properties["setpoint_temp"] = setpoint_temp_c
                                current_pcm_properties["sa_ratio"] = sa_ratio
                                current_pcm_properties["h"] = h_value
                                current_pcm_properties["enthalpy_lut_file"] = pcm_file_name

                                model_name = convert_dict_to_name(pcm_vol_fraction)
                                heater_label = (
                                    "Heatpump"
                                    if scenario["heater_type"] == "heatpump"
                                    else "Electric"
                                )
                                model_name = (
                                    f"{scenario['name']}_{model_name}_{heater_label}_"
                                    f"SA-{_format_number(sa_ratio)}_H-{_format_number(h_value)}_"
                                    f"setpoint-{setpoint_temp_f:g}F_"
                                    f"{Path(pcm_file_name).stem}_{tank_volume:g}gal_"
                                    f"{task_index}_{test_type}"
                                )
                                current_default_args = add_pcm_model(
                                    current_default_args,
                                    model_name,
                                    pcm_vol_fraction,
                                    current_pcm_properties,
                                )
                                case_path = scenario_folder / f"{model_name}.csv"
                                if test_type == "FHR":
                                    case_paths_by_group.setdefault(group, []).append(case_path)
                                add_task(
                                    current_default_args,
                                    tank_volume,
                                    setpoint_temp_c,
                                    model_name,
                                    group,
                                    test_type,
                                )
                                task_index += 1

    return tasks, case_paths_by_group, baseline_paths_by_group


def run_scenario_tasks(tasks, processes, progress=None):
    """Run one scenario's tasks and return successful timing/result records."""
    process_results = {}
    if not tasks:
        print(f"{GREEN}No simulations to run; all task results were cached{RESET}")
        return process_results

    task_futures = []
    process_count = processes or multiprocessing.cpu_count()
    process_count = max(1, int(process_count))

    print(
        f"{BOLD}{GREEN}Running {len(tasks)} task(s) with {process_count} worker(s)"
        f" for this scenario{RESET}"
    )
    with multiprocessing.Pool(processes=process_count) as pool:
        for task in tasks:
            submission_time = time.perf_counter()
            future = pool.apply_async(
                run_water_heater_process,
                (
                    task["args"],
                    task["tank_volume"],
                    task["setpoint_temp_c"],
                    task["filename"],
                    submission_time,
                ),
            )
            task_futures.append((task, future))
            print(f"{YELLOW}Submitted {task['filename']} to queue{RESET}")

        for task, future in task_futures:
            try:
                title, result, sim_duration, wait_time, total_duration = future.get()
                process_results[title] = {
                    "uef": result,
                    "sim_time": sim_duration,
                    "wait_time": wait_time,
                    "total_task_time": total_duration,
                    "group": task["group"],
                }
                print(f"{GREEN}Successfully completed: {title}{RESET}")
            except Exception as exc:
                print(
                    f"{RED}{task['filename']} error in simulation process: "
                    f"{exc}{RESET}"
                )
            finally:
                if progress is not None:
                    progress.advance()

    return process_results


def _draw_output_cache_is_complete(task):
    """Return whether a task has a usable intermediate draw-output CSV."""
    draw_output_path = task["draw_output_path"]
    return draw_output_path.is_file() and draw_output_path.stat().st_size > 0


def cache_draw_outputs_for_tasks(
    tasks,
    process_results=None,
    workers=None,
):
    """Calculate and cache draw metrics for completed tasks missing a cache."""
    process_results = process_results or {}
    tasks_to_calculate = []
    for task in tasks:
        if not _task_result_is_complete(task) or _draw_output_cache_is_complete(task):
            continue
        tasks_to_calculate.append(task)

    if not tasks_to_calculate:
        return 0

    print(
        f"{CYAN}Calculating and caching hot-water draw outputs for "
        f"{len(tasks_to_calculate)} completed simulation(s){RESET}"
    )
    draw_outputs = {}
    for test_type in sorted({task["test_type"] for task in tasks_to_calculate}):
        file_map = {
            task["filename"]: task["result_path"]
            for task in tasks_to_calculate
            if task["test_type"] == test_type
        }
        draw_outputs.update(
            calculate_hot_water_delivered_from_paths(
                file_map,
                first_hour_test=test_type == "FHR",
                workers=workers,
            )
        )

    cached_count = 0
    for task in tasks_to_calculate:
        file_key = task["filename"]
        metrics = draw_outputs.get(file_key)
        if metrics is None:
            continue
        metrics = dict(metrics)
        metrics["test_type"] = task["test_type"]

        uef = process_results.get(file_key, {}).get("uef")
        if uef is None:
            uef = calculate_uef(
                pd.read_csv(task["result_path"], index_col="Time", parse_dates=True),
                file_key,
            )
        export_draw_outputs_csv(
            {file_key: metrics},
            {file_key: uef},
            task["draw_output_path"],
        )
        cached_count += 1

    return cached_count


def combine_cached_draw_outputs(tasks, csv_path):
    """Append all intermediate draw-output CSVs into one final CSV."""
    intermediate_paths = [
        task["draw_output_path"]
        for task in tasks
        if _draw_output_cache_is_complete(task)
    ]
    if not intermediate_paths:
        print(f"{YELLOW}No cached draw-output CSVs available for final export{RESET}")
        return None, 0

    exported_count = 0
    for intermediate_path in intermediate_paths:
        intermediate_df = pd.read_csv(intermediate_path)
        export_draw_outputs_csv(
            intermediate_df,
            csv_path=csv_path,
            append=exported_count > 0,
        )
        exported_count += len(intermediate_df)

    return csv_path, exported_count


def append_results_csv(source_csv, destination_csv):
    """Append rows from a comparison CSV while preserving the union of columns."""
    source_path = Path(source_csv).expanduser().resolve()
    destination_path = Path(destination_csv).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Comparison results CSV not found: {source_path}")
    if source_path == destination_path:
        raise ValueError("Comparison results CSV must differ from the destination CSV")

    source_df = pd.read_csv(source_path)
    if "test_type" not in source_df.columns:
        if "is_FHR" in source_df.columns:
            is_fhr = source_df["is_FHR"].map(
                lambda value: str(value).strip().lower() in {"true", "1", "yes"}
            )
            source_df.insert(1, "test_type", np.where(is_fhr, "FHR", "UEF"))
        elif "file_key" in source_df.columns:
            source_df.insert(
                1,
                "test_type",
                source_df["file_key"].str.extract(
                    r"_(FHR|UEF)(?:\.csv)?$", expand=False
                ),
            )
    if destination_path.is_file():
        destination_df = pd.read_csv(destination_path)
        columns = list(destination_df.columns)
        columns.extend(
            column for column in source_df.columns if column not in columns
        )
        combined_df = pd.concat(
            [destination_df.reindex(columns=columns), source_df.reindex(columns=columns)],
            ignore_index=True,
        )
    else:
        combined_df = source_df

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(destination_path, index=False)
    return len(source_df)


def resolve_baseline_file(scenario, scenario_folder, config):
    configured_baseline = scenario["baseline_file"] or config.get("baseline_file")
    if configured_baseline is None:
        return None

    candidates = [
        Path(config["config_base"]) / Path(configured_baseline).expanduser(),
        Path.cwd() / Path(configured_baseline).expanduser(),
        Path(configured_baseline).expanduser(),
        scenario_folder / Path(configured_baseline).expanduser().name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[-1].resolve()


def generate_scenario_plots(
    scenario,
    scenario_folder,
    case_paths_by_group,
    baseline_paths_by_group,
    config,
    plot_output_folder,
    image_scale,
    show_plots=False,
):
    """Compare each setpoint/volume group and save all PNGs to one folder."""
    try:
        from bin.compare_2d_plots import process_single_folder
    except Exception as exc:
        print(f"{RED}Unable to import plot runner: {exc}{RESET}")
        return []

    configured_baseline = resolve_baseline_file(scenario, scenario_folder, config)
    saved_plot_results = []
    for group, case_paths in case_paths_by_group.items():
        case_paths = [path for path in case_paths if path.is_file()]
        # An explicitly configured baseline is independent of whether a new
        # no-PCM simulation was requested. Generated baselines are fallback
        # inputs only when no external baseline was configured.
        baseline_path = configured_baseline
        if baseline_path is None:
            baseline_path = baseline_paths_by_group.get(group)
            if baseline_path is not None and not baseline_path.is_file():
                baseline_path = None

        if not case_paths:
            print(f"{YELLOW}No completed case CSVs found for group {group}; skipping plots{RESET}")
            continue
        if baseline_path is None or not baseline_path.is_file():
            print(
                f"{YELLOW}No baseline CSV found for setpoint {group[0]:g}F and "
                f"{group[1]:g}gal; skipping plots{RESET}"
            )
            continue

        # Always stage only this group. This avoids loading old CSVs that may
        # already exist in a reused scenario folder and keeps each baseline
        # matched to its setpoint and tank volume.
        with tempfile.TemporaryDirectory(prefix="plot-input-") as staging_dir:
            staging_path = Path(staging_dir)
            for case_path in case_paths:
                shutil.copy2(case_path, staging_path / case_path.name)
            staged_baseline = staging_path / baseline_path.name
            shutil.copy2(baseline_path, staged_baseline)

            plot_result = process_single_folder(
                staging_path,
                staged_baseline.name,
                show=show_plots,
                plot_output_folder=plot_output_folder,
                image_scale=image_scale,
            )
            saved_plot_results.append(
                {
                    "group": group,
                    "baseline": str(baseline_path),
                    "result": plot_result,
                }
            )

    return saved_plot_results


def run_scenario(
    scenario,
    config,
    processes,
    plot_output_folder,
    image_scale,
    generate_plots=True,
    skip_showing_plots=True,
    export_draw_outputs=False,
    progress=None,
):
    scenario_start = time.perf_counter()
    results_root = resolve_runner_path(config["results_root"], config["config_base"])
    scenario_folder = results_root / scenario_folder_name(scenario)
    scenario_folder.mkdir(parents=True, exist_ok=True)
    if generate_plots:
        plot_output_folder.mkdir(parents=True, exist_ok=True)

    run_default_no_pcm = effective_run_default_no_pcm(scenario, config)
    no_pcm_type = scenario["default_no_pcm_type"] or config["default_no_pcm_type"]
    no_pcm_type = str(no_pcm_type).lower()
    if no_pcm_type not in {"heatpump", "electric"}:
        raise ValueError("default_no_pcm_type must be 'heatpump' or 'electric'")
    tests = effective_scenario_tests(scenario, config)

    print(f"\n{BOLD}{CYAN}Starting scenario: {scenario['name']}{RESET}")
    print(f"Results folder: {scenario_folder}")
    print(f"Plot folder: {plot_output_folder}")
    print(f"Default no-PCM baseline: {'enabled' if run_default_no_pcm else 'disabled'}")

    tasks, case_paths_by_group, baseline_paths_by_group = build_scenario_tasks(
        scenario,
        scenario_folder,
        run_default_no_pcm=run_default_no_pcm,
        no_pcm_type=no_pcm_type,
        tests=tests,
    )

    cache_payload = scenario_cache_payload(
        scenario,
        run_default_no_pcm=run_default_no_pcm,
        no_pcm_type=no_pcm_type,
        tests=tests,
    )
    cached_tasks = cached_scenario_tasks(
        tasks,
        scenario_folder,
        cache_payload,
    )
    if progress is not None:
        progress.start_case(scenario["name"], len(tasks))
        progress.advance(len(cached_tasks), force=True)
    cached_filenames = {task["filename"] for task in cached_tasks}
    tasks_to_run = [
        task for task in tasks if task["filename"] not in cached_filenames
    ]
    if cached_tasks:
        print(
            f"{GREEN}Using {len(cached_tasks)} cached task result(s) from "
            f"{scenario_folder}{RESET}"
        )
    simulation_start = time.perf_counter()
    process_results = run_scenario_tasks(
        tasks_to_run,
        processes,
        progress=progress,
    )
    simulation_seconds = time.perf_counter() - simulation_start
    if progress is not None:
        progress.render(force=True)

    if export_draw_outputs:
        cache_draw_outputs_for_tasks(
            tasks,
            process_results=process_results,
            workers=processes,
        )

    if scenario_results_are_complete(tasks):
        _write_scenario_cache(scenario_folder, cache_payload)
        print(f"{GREEN}Scenario cache is complete: {SCENARIO_CACHE_FILE}{RESET}")

    plot_start = time.perf_counter()
    if not generate_plots:
        plot_results = []
        print(f"{YELLOW}Plot generation skipped by command-line option{RESET}")
    else:
        plot_results = generate_scenario_plots(
            scenario,
            scenario_folder,
            case_paths_by_group,
            baseline_paths_by_group,
            config,
            plot_output_folder,
            image_scale,
            show_plots=not skip_showing_plots,
        )
    plot_seconds = time.perf_counter() - plot_start
    scenario_seconds = time.perf_counter() - scenario_start

    print(
        f"{BOLD}{GREEN}Scenario {scenario['name']} complete: "
        f"{len(process_results) + len(cached_tasks)}/{len(tasks)} simulations "
        f"available ({len(process_results)} run, {len(cached_tasks)} cached), "
        f"{len(plot_results)} plot group(s) generated{RESET}"
    )
    print(
        f"Scenario timing: simulations {simulation_seconds:.2f}s; "
        f"plots {plot_seconds:.2f}s; total {scenario_seconds:.2f}s"
    )
    return {
        "scenario": scenario["name"],
        "folder": str(scenario_folder),
        "tasks": tasks,
        "process_results": process_results,
        "cached_tasks": len(cached_tasks),
        "completed_simulations": len(process_results) + len(cached_tasks),
        "plot_results": plot_results,
        "timing": {
            "simulation_seconds": simulation_seconds,
            "plot_seconds": plot_seconds,
            "total_seconds": scenario_seconds,
        },
    }


def parse_runner_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run multiple PCM water-heater scenarios sequentially, then generate "
            "2-D comparison PNGs."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON runner config containing a 'scenarios' array",
    )
    parser.add_argument(
        "--run-default-no-pcm",
        action="store_true",
        help="Run a no-PCM baseline for every scenario setpoint/tank-volume pair",
    )
    parser.add_argument(
        "--no-pcm-type",
        choices=["heatpump", "electric"],
        default=None,
        help="Water-heater type for the generated no-PCM baseline",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        help="Override the configured root for scenario result folders",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        help="Override the single folder receiving all generated PNGs",
    )
    parser.add_argument(
        "--baseline-file",
        type=Path,
        help="Existing baseline CSV to use for plots, independent of no-PCM generation",
    )
    parser.add_argument(
        "--processes",
        type=int,
        help="Workers for grid points within each scenario; scenarios remain sequential",
    )
    parser.add_argument(
        "--image-scale",
        type=int,
        help="Plotly PNG scale multiplier (default: configured 4)",
    )
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        help="Generate comparison plots for each completed scenario",
    )
    parser.add_argument(
        "--skip-showing-plots",
        action="store_true",
        help="Generate and save plots without opening figures in a web browser",
    )
    parser.add_argument(
        "--export-draw-outputs",
        action="store_true",
        help="Calculate hot-water draw metrics and export all completed results to one CSV",
    )
    parser.add_argument(
        "--draw-outputs-csv",
        type=Path,
        help=(
            "CSV path for draw-output metrics; supplying this option also enables "
            "the draw-output export"
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_runner_args(argv)
    config = load_runner_config(args.config)

    if args.run_default_no_pcm:
        config["run_default_no_pcm"] = True
        for scenario in config["scenarios"]:
            scenario["run_default_no_pcm"] = True
    if args.no_pcm_type is not None:
        config["default_no_pcm_type"] = args.no_pcm_type
        for scenario in config["scenarios"]:
            scenario["default_no_pcm_type"] = args.no_pcm_type
    if args.results_root is not None:
        config["results_root"] = str(args.results_root)
    if args.baseline_file is not None:
        config["baseline_file"] = str(args.baseline_file)
    if args.processes is not None:
        if args.processes < 1:
            raise ValueError("--processes must be at least 1")
        config["processes"] = args.processes
    if args.image_scale is not None:
        if args.image_scale < 1:
            raise ValueError("--image-scale must be at least 1")
        config["image_scale"] = args.image_scale
    if args.draw_outputs_csv is not None:
        config["draw_outputs_csv"] = str(args.draw_outputs_csv)

    export_draw_outputs = bool(config.get("export_draw_outputs", False))
    export_draw_outputs = export_draw_outputs or args.export_draw_outputs
    export_draw_outputs = export_draw_outputs or args.draw_outputs_csv is not None

    plot_output_folder = resolve_runner_path(
        args.plot_output or config["plot_output_folder"],
        config["config_base"],
    )
    processes = config.get("processes")
    image_scale = int(config.get("image_scale", 4))
    if image_scale < 1:
        raise ValueError("image_scale must be at least 1")
    # Support both the Python-style config key and the hyphenated spelling
    # used by older JSON configs. CLI flags take precedence over JSON.
    generate_plots = bool(config.get("generate_plots", True))
    if "generate-plots" in config:
        generate_plots = bool(config["generate-plots"])
    if args.generate_plots:
        generate_plots = True
    skip_showing_plots = bool(
        config.get(
            "skip_showing_plots",
            config.get("skip_plots", config.get("skip-plots", True)),
        )
    )
    if args.skip_showing_plots:
        skip_showing_plots = True

    total_start = time.perf_counter()
    total_case_count = sum(
        scenario_task_count(
            scenario,
            effective_run_default_no_pcm(scenario, config),
            effective_scenario_tests(scenario, config),
        )
        for scenario in config["scenarios"]
    )
    progress = ProgressBars(total_case_count)
    print(f"{BOLD}{GREEN}Starting sequential scenario runner{RESET}")
    print(f"Scenarios: {len(config['scenarios'])}")
    print(f"Total simulation cases: {total_case_count}")
    print(f"All plots will be saved to: {plot_output_folder}")

    run_results = []
    for scenario in config["scenarios"]:
        run_results.append(
            run_scenario(
                scenario,
                config,
                processes=processes,
                plot_output_folder=plot_output_folder,
                image_scale=image_scale,
                generate_plots=generate_plots,
                skip_showing_plots=skip_showing_plots,
                export_draw_outputs=export_draw_outputs,
                progress=progress,
            )
        )

    if export_draw_outputs:
        draw_outputs_csv = resolve_runner_path(
            config["draw_outputs_csv"],
            config["config_base"],
        )
        all_tasks = [task for result in run_results for task in result["tasks"]]
        exported_path, exported_count = combine_cached_draw_outputs(
            all_tasks,
            draw_outputs_csv,
        )
        if exported_path is not None:
            print(
                f"{GREEN}Exported {exported_count} draw-output result(s) to "
                f"{exported_path}{RESET}"
            )
            comparison_csv = config.get("append_results_csv")
            if comparison_csv:
                comparison_csv = resolve_runner_path(
                    comparison_csv,
                    config["config_base"],
                )
                appended_count = append_results_csv(
                    comparison_csv,
                    exported_path,
                )
                print(
                    f"{GREEN}Appended {appended_count} comparison result(s) from "
                    f"{comparison_csv} to {exported_path}{RESET}"
                )

    total_seconds = time.perf_counter() - total_start
    total_simulations = sum(
        result["completed_simulations"] for result in run_results
    )
    total_plots = sum(len(result["plot_results"]) for result in run_results)
    print(
        f"\n{BOLD}{GREEN}All scenarios complete: "
        f"{total_simulations} successful simulations, {total_plots} plot groups, "
        f"{total_seconds:.2f}s total wall-clock time{RESET}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
