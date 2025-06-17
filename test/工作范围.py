import rospy
import moveit_commander
from geometry_msgs.msg import Pose
import numpy as np
import csv

rospy.init_node("reachable_points_checker")
robot = moveit_commander.RobotCommander()
group = moveit_commander.MoveGroupCommander("manipulator")

reachable_points = []

# 定义工作空间范围（单位：米）
x_range = np.linspace(0.2, 0.6, 10)
y_range = np.linspace(-0.3, 0.3, 10)
z_range = np.linspace(0.1, 0.5, 10)

for x in x_range:
    for y in y_range:
        for z in z_range:
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z

            # 设置一个简单的朝下姿态（与桌面平行）
            pose.orientation.x = 0.707
            pose.orientation.w = 0.707

            group.set_pose_target(pose)
            plan = group.plan()

            if isinstance(plan, tuple):
                plan = plan[1]

            if hasattr(plan, "joint_trajectory") and len(plan.joint_trajectory.points) > 0:
                reachable_points.append((x, y, z))
                print(f"✔ 可达点: x={x:.3f}, y={y:.3f}, z={z:.3f}")

print(f"✅ 可达点总数: {len(reachable_points)}")

# 保存为CSV文件
with open("reachable_points.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["x", "y", "z"])
    writer.writerows(reachable_points)

print("📁 已保存到 reachable_points.csv")
