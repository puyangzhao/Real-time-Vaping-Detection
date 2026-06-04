from __future__ import annotations
import numpy as np
import pandas as pd

def construct_segments(df, participant_col="participant_id", time_col="timestamp",
                       hr_col="heart_rate", event_col="is_event",
                       window_minutes=60, min_observed_fraction=0.8):
    """Create one 60-minute retrospective segment at each minute.

    For each participant, all monitoring days are treated as a continuous
    minute-level sequence. Each valid endpoint t defines window (t-60, t].
    Endpoints within 60 minutes before any event are labeled vaping-related.
    Overlapping pre-event intervals are combined by union; no duplicated rows.
    """
    segments = []
    min_count = int(np.ceil(window_minutes * min_observed_fraction))

    for pid, g in df.groupby(participant_col):
        g = g.sort_values(time_col).copy().set_index(time_col)
        full_index = pd.date_range(g.index.min(), g.index.max(), freq="min")
        g = g.reindex(full_index)
        g[participant_col] = pid
        g[event_col] = g[event_col].fillna(0).astype(int)

        event_times = g.index[g[event_col] == 1]
        endpoint_label = pd.Series(False, index=g.index)
        for te in event_times:
            start = te - pd.Timedelta(minutes=window_minutes - 1)
            endpoint_label.loc[(endpoint_label.index >= start) & (endpoint_label.index <= te)] = True

        hr = g[hr_col]
        for endpoint in g.index:
            start = endpoint - pd.Timedelta(minutes=window_minutes - 1)
            window = hr.loc[start:endpoint]
            if len(window) < window_minutes:
                continue
            if window.notna().sum() < min_count:
                continue
            segments.append({
                participant_col: pid,
                "segment_end_time": endpoint,
                "date": endpoint.date(),
                "is_vaping": int(endpoint_label.loc[endpoint]),
                "n_observed": int(window.notna().sum()),
                "window_start_time": start,
                "window_end_time": endpoint,
                "heart_rate_values": window.values.astype(float),
            })
    return pd.DataFrame(segments)
