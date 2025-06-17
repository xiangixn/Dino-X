#!/usr/bin/env python3
import rospy
from time import sleep
from 图像识别抓取.类.GripperController import GripperController  

def main():
    rospy.init_node("gripper_test_node")

    # 初始化夹爪控制器
    gripper = GripperController()

    rospy.loginfo("[Main] 开始测试夹爪控制")

    # 1. 张开夹爪
    rospy.loginfo("[Main] 正在张开夹爪...")
    if gripper.open():
        rospy.loginfo("[Main] 夹爪张开成功")
    else:
        rospy.logwarn("[Main] 夹爪张开失败")

    rospy.sleep(2.0)

    # 2. 闭合夹爪
    rospy.loginfo("[Main] 正在闭合夹爪...")
    if gripper.close():
        rospy.loginfo("[Main] 夹爪闭合成功")
    else:
        rospy.logwarn("[Main] 夹爪闭合失败")

    rospy.loginfo("[Main] 夹爪测试完成")

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
