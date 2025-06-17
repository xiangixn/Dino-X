#!/usr/bin/env python3

import rospy
import actionlib
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
import sys
import termios
import tty

def get_key():
    """非阻塞获取键盘按键"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key

def main():
    rospy.init_node("z1_gripper_keyboard_ctrl")
    client = actionlib.SimpleActionClient("z1_gripper", GripperCommandAction)

    rospy.loginfo("Waiting for gripper action server...")
    client.wait_for_server()
    rospy.loginfo("Connected to gripper action server!")

    print("====== Keyboard Control ======")
    print("[o] Open Gripper")
    print("[c] Close Gripper")
    print("[q] Quit")
    print("==============================")

    while not rospy.is_shutdown():
        key = get_key()
        goal = GripperCommandGoal()

        if key == 'o':
            goal.command.position = -1.0  # 张开
            goal.command.max_effort = 10.0
            rospy.loginfo("Sending OPEN: angle = %.2f rad, effort = %.2f", goal.command.position, goal.command.max_effort)
        elif key == 'c':
            goal.command.position = 0.0   # 闭合
            goal.command.max_effort = 10.0
            rospy.loginfo("Sending CLOSE: angle = %.2f rad, effort = %.2f", goal.command.position, goal.command.max_effort)
        elif key == 'q':
            print("Exiting...")
            break
        else:
            continue

        client.send_goal(goal)
        client.wait_for_result()
        result = client.get_result()
        rospy.loginfo("Result - angle: %.2f, effort: %.2f", result.position, result.effort)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass