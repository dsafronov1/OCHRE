import re
import os
import json
import pandas as pd
import numpy as np
import datetime as dt

def export_draw_outputs_csv(
    draw_outputs,
    uef_totals=None,
    csv_path="../OCHRE_results/results_csv/results.csv",
    append=False,
):

    # Keep the convenient two-argument form used by the older commented
    # example: export_draw_outputs_csv(draw_outputs, "results.csv").
    if isinstance(uef_totals, (str, os.PathLike)):
        csv_path = uef_totals
        uef_totals = None

    csv_path = os.fspath(csv_path)
    csv_dir = os.path.dirname(csv_path)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    """
    Export draw_outputs to a CSV with filename-derived parameters and computed metrics.
    Handles:
      • Full parameterized names, e.g.:
        Case1_0.56_pcm4-9_Heatpump_SA-31.00_H-2134.21_setpoint-140F_ct53_h-T_data_57frac_40gal_9.csv
      • Default-style names, e.g.:
        zDefault_No_PCM_HeatPump_setpoint-140F_40gal_0(.csv)
        zDefault(.csv)
    """

    def _json_default(o):
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        if isinstance(o, (dt.datetime, dt.date)):
            return o.isoformat()
        if isinstance(o, (dt.time,)):
            return o.isoformat()
        if isinstance(o, (dt.timedelta, pd.Timedelta)):
            return o.total_seconds()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        try:
            import pandas as _pd
            if o is _pd.NA or _pd.isna(o):
                return None
        except Exception:
            pass
        if isinstance(o, (set, tuple)):
            return list(o)
        return str(o)

    def _parse_default_name(name):
        out = {
            "file_key": name,
            "test_type": None,
            "is_default": True,
            "case": None,
            "fill_fraction": None,
            "pcm_nodes_start": None,
            "pcm_nodes_end": None,
            "heater_type": None,
            "SA_V_ratio": None,
            "H_conv": None,
            "setpoint_F": None,
            "pcm_material": None,
            "gallons": None,
            "run_index": None,
        }
        base = name.rsplit("/", 1)[-1]
        base = re.sub(r"\.csv$", "", base, flags=re.IGNORECASE)
        if re.fullmatch(r"zDefault", base, flags=re.IGNORECASE):
            out["pcm_material"] = "No_PCM" if "No_PCM" in base else out["pcm_material"]
            return out
        m = re.match(
            r"^zDefault"
            r"(?:_(?P<pcm>No_PCM))?"
            r"(?:_(?P<heater>Heat[Pp]ump|Electric|Resistance|Gas|Hybrid))?"
            r"(?:_setpoint-(?P<sp>\d+)[Ff])?"
            r"(?:_(?P<gal>\d+)\s*[Gg]al(?:_(?P<run>\d+))?)?"
            r"(?:_(?P<test>FHR|UEF))?$",
            base
        )
        if m:
            if m.group("pcm"):
                out["pcm_material"] = "No_PCM"
            if m.group("heater"):
                ht = m.group("heater")
                out["heater_type"] = "HeatPump" if ht.lower().startswith("heatp") else ht
            if m.group("sp"):
                out["setpoint_F"] = int(m.group("sp"))
            if m.group("gal"):
                out["gallons"] = int(m.group("gal"))
            if m.group("run"):
                out["run_index"] = int(m.group("run"))
            if m.group("test"):
                out["test_type"] = m.group("test").upper()
        return out

    def _parse_full_name(name):
        out = {
            "file_key": name,
            "test_type": None,
            "is_default": "zDefault" in name,
            "case": None,
            "fill_fraction": None,
            "pcm_nodes_start": None,
            "pcm_nodes_end": None,
            "heater_type": None,
            "SA_V_ratio": None,
            "H_conv": None,
            "setpoint_F": None,
            "pcm_material": None,
            "gallons": None,
            "run_index": None,
        }
        m = re.search(r"(?:^|_)Case(?P<case>[A-Za-z0-9-.]+)(?:_|$)", name, re.IGNORECASE)
        if m:
            case_token = m.group("case")
            out["case"] = int(case_token) if case_token.isdigit() else case_token
        m = re.search(r"Case[A-Za-z0-9.-]+_(?P<fill>\d*\.?\d+)(?:_|$)", name, re.IGNORECASE)
        if m:
            out["fill_fraction"] = float(m.group("fill"))
        if "No_PCM" in name:
            out["pcm_material"] = "No_PCM"
        else:
            m = re.search(r"(?:^|_)pcm(?P<s>\d+)-(?P<e>\d+)(?:_|$)", name, re.IGNORECASE)
            if m:
                out["pcm_nodes_start"] = int(m.group("s"))
                out["pcm_nodes_end"] = int(m.group("e"))
        m = re.search(r"(?:^|_)(Heat[Pp]ump|Electric|Resistance|Gas|Hybrid)(?:_|$)", name)
        if m:
            ht = m.group(1)
            out["heater_type"] = "HeatPump" if ht.lower().startswith("heatp") else ht
        m = re.search(r"(?:^|_)SA-(?P<sa>\d*\.?\d+)(?:_|$)", name, re.IGNORECASE)
        if m:
            out["SA_V_ratio"] = float(m.group("sa"))
        m = re.search(r"(?:^|_)H-(?P<h>\d*\.?\d+)(?:_|$)", name, re.IGNORECASE)
        if m:
            out["H_conv"] = float(m.group("h"))
        m = re.search(r"(?:^|_)setpoint-(?P<sp>\d+)[Ff](?:_|$)", name)
        if m:
            out["setpoint_F"] = int(m.group("sp"))
        m = re.search(
            r"(?:_|^)(?P<gal>\d+)\s*[Gg]al_(?P<run>\d+)"
            r"(?:_(?P<test>FHR|UEF))?(?:\.csv)?$",
            name,
            re.IGNORECASE,
        )
        if m:
            out["gallons"] = int(m.group("gal"))
            out["run_index"] = int(m.group("run"))
            if m.group("test"):
                out["test_type"] = m.group("test").upper()
        m = re.search(r"setpoint-\d+[Ff]_(?P<mat>.+?)_\d+[Gg]al_", name)
        if m:
            out["pcm_material"] = m.group("mat")
        if out["pcm_material"] is None and "No_PCM" in name:
            out["pcm_material"] = "No_PCM"
        return out

    def parse_file_key(file_key):
        name = file_key.rsplit("/", 1)[-1]
        if name.lower().startswith("zdefault"):
            return _parse_default_name(name)
        return _parse_full_name(name)

    preferred_order = [
        "file_key", "test_type", "is_default", "case", "fill_fraction", "pcm_nodes_start", "pcm_nodes_end",
        "heater_type", "SA_V_ratio", "H_conv", "setpoint_F", "pcm_material", "gallons", "run_index",
        "average_pcm_end_temp", "pcm_soc", "total_water_delivered_volume_L", "total_water_delivered_volume_gal",
        "total_water_FHR_volume_L", "total_water_FHR_volume_gal",
        "total_energy_used_kwh", "total_heat_delivered_J", "total_heat_delivered_kWh",
        "max_possible_hot_water_gal", "uef_all", "uef_last_day", "is_FHR", "num_draw_events", "draw_events_json"
    ]

    if isinstance(draw_outputs, pd.DataFrame):
        # Intermediate cache files are already in export-row form. Accepting
        # them directly lets the final combine step reuse this exporter
        # without recalculating the finalized time series.
        df = draw_outputs.copy()
    else:
        # Older callers pass UEF values as a list in the same order as
        # draw_outputs. Newer callers can pass a mapping keyed by file_key,
        # which is safer when results are collected asynchronously or
        # appended in batches.
        if isinstance(uef_totals, dict):
            uef_by_key = uef_totals
            uef_values = None
        else:
            uef_by_key = None
            uef_values = [] if uef_totals is None else list(uef_totals)

        rows = []
        for row_index, (file_key, metrics) in enumerate(draw_outputs.items()):
            if uef_by_key is not None:
                uef = uef_by_key.get(file_key, {})
            elif row_index < len(uef_values):
                uef = uef_values[row_index]
            else:
                uef = {}
            parsed = parse_file_key(file_key)
            record = parsed.copy()

            # Add UEF metrics immediately before is_FHR
            record["uef_all"] = uef.get("uef_all", np.nan) if isinstance(uef, dict) else np.nan
            record["uef_last_day"] = uef.get("uef_last_day", np.nan) if isinstance(uef, dict) else np.nan

            # Always include is_FHR field, even if missing
            record["is_FHR"] = metrics.get("is_FHR", False)

            for k, v in metrics.items():
                if k == "draw_events":
                    for event in metrics[k]:
                        event.pop("temp_readings", None)
                    record["draw_events_json"] = json.dumps(
                        v, default=_json_default, separators=(",", ":"), ensure_ascii=False
                    )
                else:
                    if isinstance(v, (list, dict, tuple, set, pd.Series, np.ndarray)):
                        record[k] = json.dumps(
                            v, default=_json_default, separators=(",", ":"), ensure_ascii=False
                        )
                    else:
                        try:
                            record[k] = (
                                _json_default(v)
                                if isinstance(v, (pd.Timestamp, dt.datetime, dt.date, dt.time, dt.timedelta, np.generic))
                                else v
                            )
                        except Exception:
                            record[k] = v

            rows.append(record)

        df = pd.DataFrame(rows)
    if "test_type" not in df.columns:
        inferred_test_type = pd.Series(index=df.index, dtype="object")
        if "file_key" in df.columns:
            inferred_test_type = df["file_key"].astype(str).str.extract(
                r"_(FHR|UEF)(?:\.csv)?$", expand=False
            )
        if "is_FHR" in df.columns:
            is_fhr = df["is_FHR"].map(
                lambda value: str(value).strip().lower() in {"true", "1", "yes"}
            )
            inferred_test_type = inferred_test_type.fillna(
                pd.Series(np.where(is_fhr, "FHR", "UEF"), index=df.index)
            )
        df.insert(1 if "file_key" in df.columns else 0, "test_type", inferred_test_type)

    existing_pref = [c for c in preferred_order if c in df.columns]
    extras = [c for c in df.columns if c not in existing_pref]
    df = df[existing_pref + extras]

    if append and os.path.isfile(csv_path):
        # Reindex both frames against the union of columns so append remains
        # valid if a later batch contains a metric not present in the first.
        existing_df = pd.read_csv(csv_path)
        columns = list(existing_df.columns)
        columns.extend(column for column in df.columns if column not in columns)
        df = pd.concat(
            [existing_df.reindex(columns=columns), df.reindex(columns=columns)],
            ignore_index=True,
        )

    df.to_csv(csv_path, index=False)
    return csv_path
