#!/usr/bin/env python3
import time
import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory

# Controller expects all joint names; we’ll command joint3 while keeping joint1/2 at 0
JOINTS = ['joint1', 'joint2', 'joint3']
ACTION_NAME = '/joint_trajectory_controller/follow_joint_trajectory'


def make_point(positions, tsec):
    """Helper to build a JointTrajectoryPoint with strictly increasing time_from_start."""
    p = JointTrajectoryPoint()
    p.positions = positions
    sec = int(tsec)
    nsec = int((tsec - sec) * 1e9)
    p.time_from_start = Duration(sec=sec, nanosec=nsec)
    return p


def ramp_points(target=0.6, ramp_time=2.0, hold_time=10.0, return_time=2.0, dt_hint=0.2):
    """
    Build a linear ramp up → hold → ramp down trajectory for joint3.
    Joint1 and joint2 stay at 0.0.
    """
    pts = []

    # Choose integer counts so that n*dt == segment_time exactly
    n_up = max(1, round(ramp_time / dt_hint))
    n_dn = max(1, round(return_time / dt_hint))
    dt_up = ramp_time / n_up
    dt_dn = return_time / n_dn

    # Ramp up: t in [0, ramp_time]
    for k in range(n_up + 1):
        alpha = k / n_up
        t = k * dt_up
        pos3 = target * alpha
        pts.append(make_point([0.0, 0.0, pos3], t))

    # Hold: single point at ramp_time + hold_time
    t_hold_end = ramp_time + hold_time
    pts.append(make_point([0.0, 0.0, target], t_hold_end))

    # Ramp down: t in (t_hold_end, t_hold_end + return_time]
    for k in range(1, n_dn + 1):
        alpha = k / n_dn
        t = t_hold_end + k * dt_dn
        pos3 = target * (1.0 - alpha)
        pts.append(make_point([0.0, 0.0, pos3], t))

    return pts


class TrajClient(Node):
    def __init__(self):
        super().__init__('traj3_client')
        self.ac = ActionClient(self, FollowJointTrajectory, ACTION_NAME)

    def wait(self):
        self.get_logger().info(f'Waiting for action server {ACTION_NAME}...')
        self.ac.wait_for_server()
        self.get_logger().info('...connected')

    def send(self, name, points):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINTS
        goal.trajectory.points = points

        self.get_logger().info(f'Sending trajectory: {name} '
                               f'({len(points)} points, T={points[-1].time_from_start.sec + points[-1].time_from_start.nanosec*1e-9:.2f}s)')
        fut = self.ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut)
        gh = fut.result()
        if not gh or not gh.accepted:
            self.get_logger().error('Goal rejected')
            return

        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut)
        res = res_fut.result()  # GetResult
        status = res.status
        result_msg = res.result  # FollowJointTrajectory.Result
        self.get_logger().info(
            f'Trajectory "{name}" finished: '
            f'goal_status={status}, error_code={result_msg.error_code}, '
            f'message="{result_msg.error_string}"'
        )


def main():
    rclpy.init()
    node = TrajClient()
    try:
        node.wait()

        # Ramped test for joint3: change ramp_time to try different shapes
        pts = ramp_points(target=0.6, ramp_time=0.2, hold_time=10.0, return_time=2.0, dt_hint=0.2)
        node.send('ramped_step_joint3', pts)

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
