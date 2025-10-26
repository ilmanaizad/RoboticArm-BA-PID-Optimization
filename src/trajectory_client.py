#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
import math, time

JOINTS = ['joint1', 'joint2', 'joint3']

def point(pos, t):
    p = JointTrajectoryPoint()
    p.positions = pos
    p.time_from_start = Duration(sec=int(t), nanosec=int((t - int(t)) * 1e9))
    return p

class TrajClient(Node):
    def __init__(self):
        super().__init__('traj_client')
        self.ac = ActionClient(self, FollowJointTrajectory,
                               '/joint_trajectory_controller/follow_joint_trajectory')

    def wait(self):
        self.get_logger().info('Waiting for action server...')
        self.ac.wait_for_server()
        self.get_logger().info('...connected')

    def send(self, name, points):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINTS
        goal.trajectory.points = points

        self.get_logger().info(f'Sending trajectory: {name}')
        send_future = self.ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f'Goal rejected: {name}')
            return

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        get_result = result_future.result()    # GetResult response
        status = get_result.status             # Goal status code (from action_msgs/GoalStatus)
        result_msg = get_result.result         # FollowJointTrajectory.Result

        self.get_logger().info(
            f'Trajectory "{name}" finished: '
            f'goal_status={status}, error_code={result_msg.error_code}, '
            f'message="{result_msg.error_string}"'
        )

    # ---- Test patterns ----
    def step_stagger_no_return(self, step=1.0, hold=6.0, return_time=6.0):

        base = [0.0, 0.0, 0.0]
        # Use the same targets you used before
        targets = [1.0, 0.8, -0.6] if step == 1.0 else [step, step, step]

        pts = []
        t = 0.0

        # start at base
        pts.append(point(base, t))

        # step joint1 -> hold
        t += hold
        p1 = base.copy()
        p1[0] = targets[0]
        pts.append(point(p1, t))

        # step joint2 -> hold
        t += hold
        p2 = p1.copy()
        p2[1] = targets[1]
        pts.append(point(p2, t))

        # step joint3 -> final hold
        t += hold
        p3 = p2.copy()
        p3[2] = targets[2]
        pts.append(point(p3, t))

         # Return to base
        t += return_time
        pts.append(point(base, t))

        self.send('stagger_j1_j2_j3_no_return', pts)

    def sine_tracking(self, amp=0.2, freq=0.25, duration=12.0):
        pts = [point([0.0,0.0,0.0], 0.0)]
        t = 0.5
        while t <= duration:
            s = amp * math.sin(2*math.pi*freq*t)
            pts.append(point([s, s, s], t))
            t += 0.5
        self.send('sine_tracking', pts)

def main():
    rclpy.init()
    node = TrajClient()
    node.wait()

    # A) per-joint step tests
    node.step_stagger_no_return(step=1.0, hold=6.0)

    # C) optional sine tracking
    # node.sine_tracking(amp=0.2, freq=0.25, duration=12.0)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
