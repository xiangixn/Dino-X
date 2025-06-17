from moveit_commander import MoveGroupCommander
import rospy

rospy.init_node("test_moveit_pose")
move_group = MoveGroupCommander("manipulator")  # 替换为你的 MoveIt 组名
current_pose = move_group.get_current_pose().pose

x = current_pose.position.x
y = current_pose.position.y
z = current_pose.position.z

print(f"当前末端执行器位置: x={x:.3f}, y={y:.3f}, z={z:.3f}")
