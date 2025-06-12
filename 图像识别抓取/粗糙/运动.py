#!/usr/bin/env python

import sys
import rospy
import moveit_commander
import geometry_msgs.msg
from moveit_commander.conversions import pose_to_list

def main():
    rospy.init_node("z1_go_to_pose_node", anonymous=True)
    moveit_commander.roscpp_initialize(sys.argv)

    # 选择 MoveIt 中配置的规划组名称（通常为 "manipulator"）
    move_group = moveit_commander.MoveGroupCommander("manipulator")

    # 可选：调节运动速度
    move_group.set_max_velocity_scaling_factor(0.2)
    move_group.set_max_acceleration_scaling_factor(0.3)

    # ==== ✅ 设置目标位置（单位：米） ====
    target_x = 0.35
    target_y = 0.1
    target_z = 0.42

    target_pose = geometry_msgs.msg.Pose()
    target_pose.position.x = target_x
    target_pose.position.y = target_y
    target_pose.position.z = target_z

    # 夹爪朝前（姿态平行地面，适合抓桌面物体）
    target_pose.orientation.x = 0.707
    target_pose.orientation.y = 0.0
    target_pose.orientation.z = 0.0
    target_pose.orientation.w = 0.707

    # 设置目标并规划
    move_group.set_pose_target(target_pose)

    plan = move_group.plan()
    if plan and len(plan.joint_trajectory.points) > 0:
        rospy.loginfo("[✔] 规划成功，正在执行运动...")
        move_group.go(wait=True)
    else:
        rospy.logwarn("[✘] 规划失败，检查目标是否在可达空间")

    move_group.stop()
    move_group.clear_pose_targets()
    moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    main()
