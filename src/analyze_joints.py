import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Path to your CSV (converted from rosbag)
raw_csv = "joint_states.csv"

# Load CSV
df = pd.read_csv(raw_csv)

# --- STEP 1: Parse positions (split string into floats) ---
def parse_array(val):
    try:
        return [float(x) for x in str(val).split()]
    except Exception:
        return [np.nan, np.nan, np.nan]

df["positions_split"] = df["positions"].apply(parse_array)

# Expand into separate joint columns
df["joint1_pos"] = df["positions_split"].apply(lambda x: float(x[0]) if len(x) > 0 else np.nan)
df["joint2_pos"] = df["positions_split"].apply(lambda x: float(x[1]) if len(x) > 1 else np.nan)
df["joint3_pos"] = df["positions_split"].apply(lambda x: float(x[2]) if len(x) > 2 else np.nan)

# Build clean DataFrame
df_clean = pd.DataFrame({
    "time_sec": df["sec"] + df["nsec"] * 1e-9,
    "joint1_pos": df["joint1_pos"],
    "joint2_pos": df["joint2_pos"],
    "joint3_pos": df["joint3_pos"]
}).dropna()

# Normalize time
df_clean["time_sec"] -= df_clean["time_sec"].iloc[0]

# --- STEP 2: Plot positions ---
plt.figure(figsize=(10,6))
for j in ["joint1_pos", "joint2_pos", "joint3_pos"]:
    plt.plot(df_clean["time_sec"].to_numpy(), df_clean[j].to_numpy(), label=j)
plt.xlabel("Time (s)")
plt.ylabel("Joint Position (rad)")
plt.title("Joint Positions over Time")
plt.legend()
plt.grid(True)
plt.show()

# --- STEP 3: Step response metrics ---
def step_metrics(time, signal, tol=0.05):
    """Compute rise time, settling time, overshoot for one joint."""
    y0 = signal.iloc[0]
    yfinal = signal.iloc[-1]
    amp = yfinal - y0

    if abs(amp) < 1e-6:
        return {"rise_time": None, "settling_time": None, "overshoot": None}

    # Rise time
    t10 = t90 = None
    for t, y in zip(time, signal):
        if t10 is None and (y - y0) / amp >= 0.1:
            t10 = t
        if t90 is None and (y - y0) / amp >= 0.9:
            t90 = t
            break
    rise_time = t90 - t10 if (t10 and t90) else None

    # Settling time
    lower, upper = yfinal - tol * abs(amp), yfinal + tol * abs(amp)
    settling_time = None
    for t, y in zip(reversed(time.tolist()), reversed(signal.tolist())):
        if not (lower <= y <= upper):
            settling_time = t
            break

    # Overshoot
    max_val = max(signal) if amp > 0 else min(signal)
    overshoot = (max_val - yfinal) / abs(amp) * 100

    return {
        "rise_time": rise_time,
        "settling_time": settling_time,
        "overshoot": overshoot
    }

# Print metrics for each joint
for j in ["joint1_pos", "joint2_pos", "joint3_pos"]:
    m = step_metrics(df_clean["time_sec"], df_clean[j])
    print(f"=== {j} ===")
    print(f"Rise time     : {m['rise_time']:.3f} s" if m['rise_time'] else "Rise time     : N/A")
    print(f"Settling time : {m['settling_time']:.3f} s" if m['settling_time'] else "Settling time : N/A")
    print(f"Overshoot     : {m['overshoot']:.1f} %" if m['overshoot'] is not None else "Overshoot     : N/A")
    print()
