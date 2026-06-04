from __future__ import annotations
import pandas as pd

def prepare_minute_level_data(df, participant_col="participant_id", time_col="timestamp",
                              hr_col="heart_rate", event_col="is_event"):
    required = [participant_col, time_col, hr_col, event_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    out[hr_col] = pd.to_numeric(out[hr_col], errors="coerce")
    out[event_col] = out[event_col].fillna(0).astype(int)
    out = out.dropna(subset=[participant_col, time_col, hr_col])
    out = out.sort_values([participant_col, time_col]).reset_index(drop=True)
    return out
