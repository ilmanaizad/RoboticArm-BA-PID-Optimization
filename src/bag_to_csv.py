#!/usr/bin/env python3
"""
bag_to_csv.py
Convert a ROS 2 bag to a clean per-joint CSV for /joint_states.

Usage:
  python3 bag_to_csv.py -b <bag_folder> -o <out.csv>
Options:
  --topic   (default: /joint_states)
  --joints  (default: joint1 joint2 joint3)  e.g. --joints joint1 joint2 joint3
"""

from pathlib import Path
import argparse
import csv
import math

from rosbags.highlevel import AnyReader

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-b", "--bag", required=True, help="Path to ROS 2 bag folder")
    ap.add_argument("-o", "--out", required=True, help="Output CSV path")
    ap.add_argument("--topic", default="/joint_states", help="JointStates topic name")
    ap.add_argument("--joints", nargs="+", default=["joint1", "joint2", "joint3"],
                    help="Ordered joint names to export as columns")
    return ap.parse_args()

def main():
    args = parse_args()
    bag_path = Path(args.bag)
    out_csv = args.out
    topic = args.topic
    joint_list = list(args.joints)

    if not bag_path.exists():
        raise SystemExit(f"Bag folder not found: {bag_path}")

    # Open bag
    with AnyReader([bag_path]) as reader, open(out_csv, "w", newline="") as f:
        # Prepare connections for the topic
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            raise SystemExit(f"No connections for topic '{topic}' in bag: {bag_path}")

        writer = csv.writer(f)
        header = ["time_sec"] + joint_list
        writer.writerow(header)

        # Keep last seen values to carry forward if a joint is missing in a message
        last_vals = {j: math.nan for j in joint_list}

        count = 0
        for connection, timestamp, rawdata in reader.messages(connections=conns):
            msg = reader.deserialize(rawdata, connection.msgtype)

            # Timestamp in seconds (float)
            t = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

            # Build name->position mapping safely
            names = list(msg.name) if hasattr(msg, "name") else []
            pos = list(msg.position) if hasattr(msg, "position") else []
            name_to_pos = {n: float(p) for n, p in zip(names, pos)}

            row = [t]
            for j in joint_list:
                val = name_to_pos.get(j, last_vals.get(j, math.nan))
                row.append(val)
                last_vals[j] = val
            writer.writerow(row)
            count += 1

    print(f"✅ Saved {out_csv} with {count} rows and columns {header}")

if __name__ == "__main__":
    main()
