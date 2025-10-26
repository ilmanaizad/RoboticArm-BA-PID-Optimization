#!/usr/bin/env python3
"""
analyze_and_export.py

Reads a joint_states CSV (either clean per-joint columns or packed arrays),
normalizes time to t=0, selects analysis windows (widened or auto),
computes step/ramp metrics, and exports:

  <out>_timeseries_export.csv
  <out>_metrics_summary.csv

Modes:
  --mode stagger   : Trajectory 1 (step-stagger) → 3 windows with margins
  --mode ramp      : Trajectory 2 (single-joint ramp) → one window for ramp joint
  --mode auto      : No assumption; auto-detect per joint

Usage examples:

# Trajectory 1 (stagger), hold=6s, targets 1.0 0.8 -0.6, margins -0.5/+2.0 s
python3 analyze_and_export.py \
  --input traj1_pid.csv \
  --out-prefix traj1_pid \
  --label "Trajectory 1 | PID" \
  --mode stagger --hold 6.0 \
  --targets 1.0 0.8 -0.6 \
  --joints joint1 joint2 joint3 \
  --margin-start -0.5 --margin-end 2.0

# Trajectory 2 (ramp on joint3), window = full duration (or auto)
python3 analyze_and_export.py \
  --input traj2_baseline.csv \
  --out-prefix traj2_baseline \
  --label "Trajectory 2 | Baseline" \
  --mode ramp --ramp-joint joint3 \
  --joints joint1 joint2 joint3
"""

import argparse
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# ===================== Metrics helpers =====================

def metrics_step_segment(t: np.ndarray, y: np.ndarray, y0: float, y1: float) -> Dict[str, float]:
    """
    Step metrics using robust bands:
      - rise: first time |y - y1| <= 5% of step magnitude (enter 5% band)
      - settling: last time |y - y1| > 2% of step magnitude (2% band)
      - overshoot: max excursion beyond target (as % of step magnitude, clipped at 0)
      - sse: sum of squared error vs final target over the segment
    """
    if t.size < 3:
        return dict(rise=np.nan, settling=np.nan, overshoot=np.nan, sse=np.nan)

    mag = float(y1 - y0)
    if mag == 0.0:
        return dict(rise=np.nan, settling=np.nan, overshoot=np.nan, sse=np.nan)

    sign = 1.0 if mag >= 0 else -1.0
    yN = (y - y0) * sign
    mag = abs(mag)

    # --- Rise time: enter ±5% band around target
    # equivalently: yN within [0.95*mag, 1.05*mag]
    lower = 0.95 * mag
    upper = 1.05 * mag
    idx_rise = np.where((yN >= lower) & (yN <= upper))[0]
    rise = (t[idx_rise[0]] - t[0]) if idx_rise.size > 0 else np.nan

    # --- Overshoot (%): amount above target (no negatives)
    peak = float(np.max(yN))
    overshoot = max(0.0, (peak - mag) / mag * 100.0)

    # --- Settling time: last time outside ±2% band around target
    band2 = 0.02 * mag
    outside = np.where(np.abs(yN - mag) > band2)[0]
    settling = (t[outside[-1]] - t[0]) if outside.size > 0 else 0.0

    # --- SSE over the window vs final target
    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.0
    sse = float(np.sum((y - y1) ** 2) * max(dt, 0.0))

    return dict(rise=rise, settling=settling, overshoot=overshoot, sse=sse)


# ===================== CSV parsing helpers =====================

def try_parse_float_list(s):
    if isinstance(s, (list, tuple, np.ndarray)):
        return [float(x) for x in s]
    if not isinstance(s, str):
        return []
    out = []
    for p in s.strip().split():
        try:
            out.append(float(p))
        except Exception:
            if isinstance(p, str) and p.lower() == 'nan':
                out.append(float('nan'))
    return out

def try_parse_str_list(s):
    if isinstance(s, (list, tuple, np.ndarray)):
        return [str(x) for x in s]
    if not isinstance(s, str):
        return []
    return s.strip().split()

def coerce_time(df: pd.DataFrame) -> np.ndarray:
    # time_sec
    for c in df.columns:
        if c.lower() == 'time_sec':
            return pd.to_numeric(df[c], errors='coerce').astype(float).values

    # sec + nsec/nanosec
    sec_col, nsec_col = None, None
    for c in df.columns:
        cl = c.lower()
        if cl == 'sec': sec_col = c
        if cl in ('nsec', 'nanosec'): nsec_col = c
    if sec_col and nsec_col:
        sec = pd.to_numeric(df[sec_col], errors='coerce').astype(float).values
        nsec = pd.to_numeric(df[nsec_col], errors='coerce').astype(float).values
        return sec + nsec*1e-9

    raise ValueError("CSV needs 'time_sec' or ('sec' + 'nsec'/'nanosec').")

def expand_packed(df: pd.DataFrame, joints: List[str]) -> pd.DataFrame:
    """If columns are packed (joint_names/positions), expand to numeric per-joint."""
    lower = {c.lower(): c for c in df.columns}
    jnames = lower.get('joint_names')
    pos    = lower.get('positions')

    if not jnames or not pos:
        return df  # already expanded

    t = coerce_time(df)
    out = pd.DataFrame({'time_sec': t})
    for j in joints: out[j] = np.nan

    for i in range(len(df)):
        names = try_parse_str_list(df.iloc[i][jnames])
        vals  = try_parse_float_list(df.iloc[i][pos])
        mp = {n: v for n, v in zip(names, vals)}
        for j in joints:
            if j in mp:
                out.at[i, j] = mp[j]

    # carry-forward to fill sparse rows
    out[joints] = out[joints].fillna(method='ffill').fillna(method='bfill')
    return out

def ensure_timeseries(df: pd.DataFrame, joints: List[str]) -> pd.DataFrame:
    """Return df with columns: time_sec + each joint (numeric)."""
    lower = {c.lower(): c for c in df.columns}
    if 'time_sec' in lower and all(j in lower for j in joints):
        cols = ['time_sec'] + joints
        return df[[lower[c] for c in cols]].rename(columns={lower['time_sec']: 'time_sec'})

    df2 = expand_packed(df, joints)
    lower2 = {c.lower(): c for c in df2.columns}
    if 'time_sec' in lower2 and all(j in lower2 for j in joints):
        cols = ['time_sec'] + joints
        return df2[[lower2[c] for c in cols]].rename(columns={'time_sec': 'time_sec'})

    raise ValueError("Could not construct timeseries. Ensure per-joint columns exist or packed columns can be expanded.")

# ===================== Window selection helpers =====================

def detect_step_window(t: np.ndarray, y: np.ndarray, min_step=0.05, hold=6.0) -> Optional[Tuple[float, float]]:
    """
    Finds first significant change in y(t); returns window [t0, t0+hold]
    """
    if t.size < 5: return None
    dy = np.abs(y - y[0])
    idx = np.where(dy > min_step)[0]
    if idx.size == 0: return None
    t0 = float(t[idx[0]])
    return (t0, min(float(t[-1]), t0 + float(hold)))

def build_windows_stagger(hold: float, margin_start: float, margin_end: float) -> Dict[str, Tuple[float,float]]:
    # base stagger: [0..hold], [hold..2hold], [2hold..3hold]
    # widen with margins: [0+ms .. hold+me], etc. (ms can be negative)
    w = {}
    # order is joint1, joint2, joint3
    w[0] = (max(0.0, 0.0 + margin_start), hold + margin_end)
    w[1] = (max(0.0, hold + margin_start), 2*hold + margin_end)
    w[2] = (max(0.0, 2*hold + margin_start), 3*hold + margin_end)
    return w

# ===================== Main analysis =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out-prefix', required=True)
    ap.add_argument('--label', default='')
    ap.add_argument('--joints', nargs='+', default=['joint1','joint2','joint3'])
    ap.add_argument('--mode', choices=['stagger','ramp','auto'], default='stagger')
    ap.add_argument('--hold', type=float, default=6.0, help='Per-step duration (stagger) or analysis window for auto.')
    ap.add_argument('--targets', nargs='+', type=float, default=[1.0, 0.8, -0.6],
                    help='Final targets for step (stagger mode): len must match joints.')
    ap.add_argument('--ramp-joint', default='joint3', help='Joint to analyze in ramp mode.')
    ap.add_argument('--margin-start', type=float, default=-0.5, help='Window start margin (stagger), can be negative.')
    ap.add_argument('--margin-end', type=float, default=2.0, help='Window end margin (stagger).')
    args = ap.parse_args()

    df_raw = pd.read_csv(args.input)
    df_ts = ensure_timeseries(df_raw, args.joints)

    # Normalize time so first sample is t=0
    t0 = float(df_ts['time_sec'].iloc[0])
    df_ts['time_sec'] = df_ts['time_sec'] - t0

    # Export clean timeseries
    ts_out = f"{args.out_prefix}_timeseries_export.csv"
    df_ts[['time_sec'] + args.joints].to_csv(ts_out, index=False)

    t = df_ts['time_sec'].values
    metrics_rows = []

    if args.mode == 'stagger':
        # Build widened windows (with margins)
        windows_map = build_windows_stagger(args.hold, args.margin_start, args.margin_end)
        # For each joint in order, compute step metrics; fallback to auto if window is empty/flat
        for idx, (j, target) in enumerate(zip(args.joints, args.targets)):
            t0w, t1w = windows_map[idx]
            mask = (t >= t0w) & (t <= t1w)
            y = df_ts[j].values
            used_mode = 'fixed'
            # Fallback to auto-detect if nothing or too flat (<1e-3 span)
            if (mask.sum() < 5) or (np.nanmax(y[mask]) - np.nanmin(y[mask]) < 1e-3):
                auto = detect_step_window(t, y, min_step=0.05, hold=args.hold)
                if auto is not None:
                    t0w, t1w = auto
                    mask = (t >= t0w) & (t <= t1w)
                    used_mode = 'auto'
            tt, yy = t[mask], y[mask]
            if tt.size == 0:
                m = dict(rise=np.nan, settling=np.nan, overshoot=np.nan, sse=np.nan)
            else:
                # y0 = value at t0w
                start_idx = np.argmax(t >= t0w)
                y0 = df_ts[j].values[start_idx]
                m = metrics_step_segment(tt, yy, float(y0), float(target))
            metrics_rows.append({
                'label': args.label, 'mode': 'stagger', 'window_mode': used_mode,
                'joint': j, 'target': target,
                'rise_time': m['rise'], 'settling_time': m['settling'],
                'overshoot': m['overshoot'], 'sse': m['sse'],
                'window_start': t0w, 'window_end': t1w
            })

    elif args.mode == 'ramp':
        # Analyze a single ramp joint (others optional / ignored here)
        j = args.ramp_joint
        if j not in args.joints:
            raise ValueError(f"--ramp-joint '{j}' not in joints list {args.joints}")
        # Use full duration, or detect first change and use [t0, t0+hold]
        y = df_ts[j].values
        t0w, t1w = 0.0, float(t[-1])
        auto = detect_step_window(t, y, min_step=0.02, hold=args.hold)  # smaller threshold for ramps
        if auto is not None:
            t0w, t1w = auto
        mask = (t >= t0w) & (t <= t1w)
        tt, yy = t[mask], y[mask]
        m = metrics_ramp_segment(tt, yy) if tt.size else dict(ramp_sse=np.nan, end_error=np.nan)
        metrics_rows.append({
            'label': args.label, 'mode': 'ramp', 'window_mode': 'auto' if auto else 'full',
            'joint': j, 'target': np.nan,
            'rise_time': np.nan, 'settling_time': np.nan, 'overshoot': np.nan,
            'sse': m.get('sse', np.nan),
            'ramp_sse': m.get('ramp_sse', np.nan), 'end_error': m.get('end_error', np.nan),
            'window_start': t0w, 'window_end': t1w
        })

    else:  # auto for each joint (generic fallback)
        for j in args.joints:
            y = df_ts[j].values
            auto = detect_step_window(t, y, min_step=0.05, hold=args.hold)
            if auto is None:
                metrics_rows.append({
                    'label': args.label, 'mode': 'auto', 'window_mode': 'none',
                    'joint': j, 'target': np.nan,
                    'rise_time': np.nan, 'settling_time': np.nan, 'overshoot': np.nan,
                    'sse': np.nan, 'window_start': np.nan, 'window_end': np.nan
                })
                continue
            t0w, t1w = auto
            mask = (t >= t0w) & (t <= t1w)
            tt, yy = t[mask], y[mask]
            start_idx = np.argmax(t >= t0w)
            y0 = df_ts[j].values[start_idx]
            # Without a declared target, use final tt end as target surrogate
            m = metrics_step_segment(tt, yy, float(y0), float(yy[-1]))
            metrics_rows.append({
                'label': args.label, 'mode': 'auto', 'window_mode': 'auto',
                'joint': j, 'target': float(yy[-1]),
                'rise_time': m['rise'], 'settling_time': m['settling'],
                'overshoot': m['overshoot'], 'sse': m['sse'],
                'window_start': t0w, 'window_end': t1w
            })

    mx_out = f"{args.out_prefix}_metrics_summary.csv"
    pd.DataFrame(metrics_rows).to_csv(mx_out, index=False)

if __name__ == '__main__':
    main()
