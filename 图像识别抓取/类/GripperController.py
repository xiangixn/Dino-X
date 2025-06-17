import rospy
import actionlib
from control_msgs.msg import GripperCommandAction, GripperCommandGoal

class GripperController:
    def __init__(self, gripper_action_name="z1_gripper"):
        rospy.loginfo(f"[Gripper] 初始化 Gripper Action Client: {gripper_action_name}")
        self.client = actionlib.SimpleActionClient(gripper_action_name, GripperCommandAction)
        self.client.wait_for_server()
        rospy.loginfo("[Gripper] Gripper Action Server 已连接")

    def move_gripper(self, position: float, max_effort: float = 10.0) -> bool:
        """
        控制夹爪移动到指定角度位置（单位：rad）
        注意：position=-1 表示闭合，0 表示张开（与默认方向相反）
        """
        rospy.loginfo(f"[Gripper] 发送目标: angle={position:.2f} rad, effort={max_effort:.1f} N·m")
        goal = GripperCommandGoal()
        goal.command.position = position
        goal.command.max_effort = max_effort

        self.client.send_goal(goal)
        self.client.wait_for_result()

        result = self.client.get_result()
        if self.client.get_state() == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo(f"[Gripper] 执行成功，实际角度: {result.position:.3f}，施力: {result.effort:.3f}")
            return True
        else:
            rospy.logwarn(f"[Gripper] 执行失败，当前状态: {self.client.get_state()}")
            return False

    def open(self):
        """张开夹爪：位置 0，默认力度"""
        return self.move_gripper(position=0.0, max_effort=10.0)

    def close(self):
        """闭合夹爪：位置 -1，力度减小避免损坏"""
        return self.move_gripper(position=-0.5, max_effort=2.0)