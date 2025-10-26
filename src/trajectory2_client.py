#!/usr/bin/env python3
import argparse
import math
import time
import subprocess
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

JOINTS = ['joint1', 'joint2', 'joint3']

def jt_point(pos, t):
    p = JointTrajectoryPoint()
    p.positions = pos
    p.time_from_start = Duration(sec=int(t), nanosec=int((t - int(t)) * 1e9))
    return p

class StepStaggerSeq(Node):
    """
    Sequence: joint2 -> joint3 -> joint1 (degrees), hold after each, then return to origin.
    One-shot command. Optional: record bag and convert to CSV afterwards.
    """
    def __init__(self, j2_deg, j3_deg, j1_deg, hold, return_time, start_delay):
        super().__init__('traj_step_stagger_seq')

        # store targets in radians; array is [j1, j2, j3]
        self.targets = [
            math.radians(j1_deg),
            math.radians(j2_deg),
            math.radians(j3_deg),
        ]
        self.hold = float(hold)
        self.return_time = float(return_time)
        self.start_delay = float(start_delay)

        self.ac = ActionClient(
            self, FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory'
        )

    def wait(self):
        self.get_logger().info('Waiting for /joint_trajectory_controller/follow_joint_trajectory...')
        self.ac.wait_for_server()
        self.get_logger().info('...connected')

    def build_points(self):
        pts = []
        t = 0.0
        base = [0.0, 0.0, 0.0]

        # small start delay so the controller latches the first point
        pts.append(jt_point(base, t))
        t += self.start_delay

        # 1) step joint2 -> hold
        p1 = base.copy()
        p1[1] = self.targets[1]  # joint2 target
        pts.append(jt_point(p1, t + self.hold))
        t += self.hold

        # 2) step joint3 -> hold
        p2 = p1.copy()
        p2[2] = self.targets[2]  # joint3 target
        pts.append(jt_point(p2, t + self.hold))
        t += self.hold

        # 3) step joint1 -> hold
        p3 = p2.copy()
        p3[0] = self.targets[0]  # joint1 target
        pts.append(jt_point(p3, t + self.hold))
        t += self.hold

        # 4) return to origin
        pts.append(jt_point(base, t + self.return_time))

        return pts

    def send(self):
        pts = self.build_points()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINTS
        goal.trajectory.points = pts

        self.get_logger().info(
            f"Sending step-stagger (j2→j3→j1): "
            f"j2={math.degrees(self.targets[1]):.1f}°, "
            f"j3={math.degrees(self.targets[2]):.1f}°, "
            f"j1={math.degrees(self.targets[0]):.1f}°, "
            f"hold={self.hold}s, return={self.return_time}s, start_delay={self.start_delay}s, "
            f"points={len(pts)}"
        )

        send_fut = self.ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_fut)
        gh = send_fut.result()
        if not gh or not gh.accepted:
            self.get_logger().error('Goal was rejected')
            return False

        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut)
        res = res_fut.result()
        self.get_logger().info(
            f'Trajectory finished: goal_status={res.status}, '
            f'error_code={res.result.error_code}, message="{res.result.error_string}"'
        )
        return True

def record_and_convert(bag_out, csv_out):
    """
    Optional helper: record /joint_states while the trajectory runs, then convert bag -> CSV
    using your existing bag_to_csv.py that accepts -b (bag path) and -o (csv path).
    """
    bag_out = Path(bag_out).expanduser().resolve()
    csv_out = Path(csv_out).expanduser().resolve()

    # Start recording
    rec = subprocess.Popen(
        ["ros2", "bag", "record", "-o", str(bag_out), "/joint_states"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    time.sleep(0.5)  # let recorder spin up
    return rec, bag_out, csv_out

def stop_recording_and_convert(rec_proc, bag_out, csv_out, script_dir: Path, bag2csv_override: str = ''):
    try:
        rec_proc.terminate(); rec_proc.wait(timeout=2.0)
    except Exception:
        rec_proc.kill()

    candidates = []
    if bag2csv_override:
        candidates.append(Path(bag2csv_override))
    # same folder as this script (works if you run from source)
    candidates.append(script_dir / "bag_to_csv.py")
    # typical source path
    candidates.append(Path.home() / "ros2_ws" / "src" / "robotic_arm" / "src" / "bag_to_csv.py")

    for p in candidates:
        if p.exists():
            subprocess.run(["python3", str(p), "-b", str(bag_out), "-o", str(csv_out)], check=True)
            return
    print(f"[WARN] bag_to_csv.py not found in {', '.join(str(c) for c in candidates)}; skipping conversion.")


def main():
    ap = argparse.ArgumentParser(
        description='Step-stagger variant: joint2 → joint3 → joint1 (degrees), hold, return to 0.'
    )
    # Defaults you asked for: j2=45°, j3=??° (set default 45), j1=80°
    ap.add_argument('--j2-deg', type=float, default=45.0, help='Joint2 target [deg]')
    ap.add_argument('--j3-deg', type=float, default=45.0, help='Joint3 target [deg]')
    ap.add_argument('--j1-deg', type=float, default=80.0, help='Joint1 target [deg]')
    ap.add_argument('--hold', type=float, default=6.0, help='Hold time after each step [s]')
    ap.add_argument('--return-time', type=float, default=6.0, help='Return-to-zero duration [s]')
    ap.add_argument('--start-delay', type=float, default=1.0, help='Initial delay before first step [s]')

    # One-shot recording & conversion (optional)
    ap.add_argument('--record-bag', type=str, default='', help='If set, record to this bag path (no extension)')
    ap.add_argument('--csv-out', type=str, default='', help='If set, convert bag to this CSV path after run')
    ap.add_argument('--bag2csv-path', type=str, default='',
                help='Path to bag_to_csv.py (if not in the same folder)')


    args, unknown = ap.parse_known_args()

    # Init ROS with remaining args
    rclpy.init(args=unknown)
    node = StepStaggerSeq(
        j2_deg=args.j2_deg,
        j3_deg=args.j3_deg,
        j1_deg=args.j1_deg,
        hold=args.hold,
        return_time=args.return_time,
        start_delay=args.start_delay
    )
    node.wait()

    # (Optional) start bag recording
    rec = None
    bag_out = csv_out = None
    if args.record_bag:
        rec, bag_out, csv_out = record_and_convert(args.record_bag, args.csv_out or (args.record_bag + ".csv"))

    ok = node.send()

    # Stop recording and convert if requested
    if rec is not None:
        stop_recording_and_convert(rec, bag_out, csv_out, Path(__file__).parent, args.bag2csv_path)


    node.destroy_node()
    rclpy.shutdown()

    # exit code
    if not ok:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
