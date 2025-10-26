#!/usr/bin/env python3
import argparse, random, math, subprocess, time, json
import numpy as np
import pandas as pd
from pathlib import Path

# put near the top of the file
JTC_NODE = "/joint_trajectory_controller"   # node to set params on
JTC_NAME = "joint_trajectory_controller"    # controller name for switch

def _try(cmd):
    import subprocess
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode == 0

def _restart_jtc():
    # stop then start the controller so params re-apply on activation
    _try(["ros2","control","switch_controllers","--stop", JTC_NAME,"--strict"])
    _try(["ros2","control","switch_controllers","--start",JTC_NAME,"--strict"])

def set_pid(gains, restart_controller=True):
    """
    gains = [P1,I1,D1, P2,I2,D2, P3,I3,D3]
    Writes to:
      /joint_trajectory_controller gains.jointX.{p,i,d}
    """
    import subprocess
    for k, j in enumerate(["joint1","joint2","joint3"]):
        P, I, D = gains[3*k:3*k+3]
        if not _try(["ros2","param","set", JTC_NODE, f"gains.{j}.p", str(float(P))]): raise RuntimeError(f"set {j}.p failed")
        if not _try(["ros2","param","set", JTC_NODE, f"gains.{j}.i", str(float(I))]): raise RuntimeError(f"set {j}.i failed")
        if not _try(["ros2","param","set", JTC_NODE, f"gains.{j}.d", str(float(D))]): raise RuntimeError(f"set {j}.d failed")

    if restart_controller:
        _restart_jtc()

# ---- CONFIG you may need to tweak ----
JOINTS = ["joint1","joint2","joint3"]

# BA hyperparams (keep small first)
N_BATS   = 6
N_ITERS  = 6
FMIN, FMAX = 0.0, 2.0
ALPHA, GAMMA = 0.9, 0.9

# Bounds per term (tune if unstable/weak)
BOUNDS = { "p": (0.5, 40.0), "i": (0.0, 5.0), "d": (0.0, 3.0) }

# Trajectory & analysis settings (Trajectory 1: step-stagger)
HOLD = 6.0
TARGETS = [1.3963, 0.7854, 0.7854]
MARGIN_START = -1.0
MARGIN_END   = 6.0

def run_traj_and_export(tmp_prefix):
    # 1) run trajectory (blocking)
    subprocess.run([
    "ros2","run","robotic_arm","trajectory2_client.py","--",
    "--j2-deg","45","--j3-deg","45","--j1-deg","80",
    "--hold","6.0","--return-time","6.0","--start-delay","1.0"
], check=True)

    # 2) Convert latest bag if needed OR assume you already convert externally.
    #    Here we assume you will pass a CSV already created by your new bag_to_csv,
    #    OR you prefer to re-export from a bag.
    # For robustness, we call your analyze script directly on an *existing* CSV path:
    pass

def analyze_csv(csv_path, out_prefix, label):
    # Call your analyzer (new version you use) to export timeseries+metrics
    cmd = [
        "python3", str(Path(__file__).with_name("analyze_and_export.py")),
        "--input", csv_path,
        "--out-prefix", out_prefix,
        "--label", label,
        "--mode", "stagger",
        "--hold", str(HOLD),
        "--targets", *[str(x) for x in TARGETS],
        "--joints", *JOINTS,
        "--margin-start", str(MARGIN_START),
        "--margin-end", str(MARGIN_END),
    ]
    subprocess.run(cmd, check=True)
    # Load metrics for cost
    mx = pd.read_csv(f"{out_prefix}_metrics_summary.csv")
    return mx

def compute_cost(mx: pd.DataFrame):
    # Expect rows per joint
    # Robustly handle missing values
    def get(v): 
        return 0.0 if pd.isna(v) else float(v)
    cost = 0.0
    for j in JOINTS:
        row = mx[mx["joint"]==j]
        if row.empty:
            cost += 1e6
            continue
        rise   = get(row["rise_time"].iloc[0])
        settle = get(row["settling_time"].iloc[0])
        sse    = get(row["sse"].iloc[0])
        over   = get(row["overshoot"].iloc[0])
        over_pen = max(0.0, over - 2.0)  # 2% free
        # weights
        cost += 0.45*settle + 0.35*rise + 0.15*sse + 0.05*over_pen
    return cost

def project_bounds(g):
    out = []
    for k, val in enumerate(g):
        key = ["p","i","d"][k%3]
        lo, hi = BOUNDS[key]
        out.append(min(max(val, lo), hi))
    return out

def evaluate_candidate(gains, bag_folder, tmpname):
    """
    Evaluate a candidate:
      - set PID
      - record a quick bag for the stagger trajectory
      - convert bag to CSV
      - analyze CSV
      - return scalar cost
    """
    set_pid(gains)
    # record bag
    bag = Path(bag_folder) / tmpname
    # Start recording
    rec = subprocess.Popen(["ros2","bag","record","-o", str(bag), "/joint_states"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.5)
    # Play trajectory
    subprocess.run([
    "ros2","run","robotic_arm","trajectory2_client.py","--",
    "--j2-deg","45","--j3-deg","45","--j1-deg","80",
    "--hold","6.0","--return-time","6.0","--start-delay","1.0"
], check=True)

    # Stop recording
    rec.terminate()
    try:
        rec.wait(timeout=2.0)
    except Exception:
        rec.kill()

    # Convert bag to CSV (your fixed exporter writes time_sec,joint1,2,3)
    csv_path = f"{tmpname}.csv"
    subprocess.run([
        "python3", str(Path(__file__).with_name("bag_to_csv.py")),
        "-b", str(bag), "-o", csv_path
    ], check=True)

    # Analyze
    mx = analyze_csv(csv_path, out_prefix=tmpname, label="BA-PID | candidate")
    cost = compute_cost(mx)
    return cost

def bat_optimize(bag_folder, seed=None):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    dim = 9
    def rand_vec():
        vec=[]
        for k in range(dim):
            key = ["p","i","d"][k%3]
            lo, hi = BOUNDS[key]
            vec.append(random.uniform(lo, hi))
        return vec

    X = [rand_vec() for _ in range(N_BATS)]
    V = [[0.0]*dim for _ in range(N_BATS)]
    A = [1.0]*N_BATS
    r = [0.5]*N_BATS

    # initial evaluate (use short tmp names)
    J = []
    for i in range(N_BATS):
        tmpname = f"ba_tmp_bat{i}"
        J.append(evaluate_candidate(X[i], bag_folder, tmpname))
    best_i = int(np.argmin(J))
    best_x, best_j = X[best_i][:], J[best_i]

    print(f"[init] best J={best_j:.4f} gains={np.round(best_x,3)}")

    for t in range(1, N_ITERS+1):
        Abar = sum(A)/len(A)
        for i in range(N_BATS):
            f = FMIN + (FMAX - FMIN)*random.random()
            # v & x update
            V[i] = [V[i][d] + (X[i][d] - best_x[d]) * f for d in range(dim)]
            cand = [X[i][d] + V[i][d] for d in range(dim)]
            # local random walk
            if random.random() < r[i]:
                cand = [best_x[d] + random.gauss(0, 0.02) * Abar for d in range(dim)]
            cand = project_bounds(cand)
            # evaluate
            tmpname = f"ba_tmp_t{t}_bat{i}"
            j_cand = evaluate_candidate(cand, bag_folder, tmpname)

            # acceptance
            if (j_cand <= J[i]) and (random.random() < A[i]):
                X[i], J[i] = cand, j_cand
                A[i] *= ALPHA
                r[i] = r[i] * (1 - math.exp(-GAMMA * t))

            if J[i] < best_j:
                best_x, best_j = X[i][:], J[i]
        print(f"[iter {t}] best J={best_j:.4f} gains={np.round(best_x,3)}")
    return best_x, best_j

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag-dir", default=str(Path.cwd()), help="Directory where temp bags will be written")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--final-prefix", default="traj2_bapid")
    args = ap.parse_args()

    best_g, best_cost = bat_optimize(args.bag_dir, args.seed)
    print("\nBEST:", best_g, "J=", best_cost)

    # Apply best gains and export official run
    set_pid(best_g)

    # record final bag
    final_bag = Path(args.bag_dir) / (args.final_prefix + "_bag")
    rec = subprocess.Popen(["ros2","bag","record","-o", str(final_bag), "/joint_states"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.5)
    subprocess.run([
    "ros2","run","robotic_arm","trajectory2_client.py","--",
    "--j2-deg","45","--j3-deg","45","--j1-deg","80",
    "--hold","6.0","--return-time","6.0","--start-delay","1.0"
], check=True)

    rec.terminate()
    try:
        rec.wait(timeout=2.0)
    except Exception:
        rec.kill()

    # convert + analyze
    final_csv = args.final_prefix + ".csv"
    subprocess.run([
        "python3", str(Path(__file__).with_name("bag_to_csv.py")),
        "-b", str(final_bag), "-o", final_csv
    ], check=True)

    analyze_csv(final_csv, out_prefix=args.final_prefix, label="Trajectory 2 | BA-PID")
    print(f"\n✅ Final exports written: {args.final_prefix}_timeseries_export.csv and {args.final_prefix}_metrics_summary.csv")

if __name__ == "__main__":
    main()
