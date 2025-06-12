import rospy
import moveit_commander
import geometry_msgs.msg
import numpy as np
import sys

class RobotArm:
    def __init__(self, group_name="manipulator"):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.loginfo("[Init] 初始化 MoveIt 控制器")
        self.move_group = moveit_commander.MoveGroupCommander(group_name)

        # 设置参考坐标系为 link00（你的机械臂 base）
        self.move_group.set_pose_reference_frame("link00")
        rospy.loginfo(f"[Init] 当前规划参考坐标系: {self.move_group.get_planning_frame()}")

        self._configure_arm()

        # 标定得到的相机 → base（link00）变换参数
        self.R = np.array([ [ 0.71019493 , -0.02226755 , -0.70365284],
                            [ 0.15695671 , 0.97935068  , 0.12742381] ,
                            [ 0.68628547 , -0.20093878 , 0.69902493]])
        
        self.T = np.array([ [[0.37621825],
                            [0.09743894],
                            [0.24983783]]])

    def _configure_arm(self):
        self.move_group.set_max_velocity_scaling_factor(0.2)
        self.move_group.set_max_acceleration_scaling_factor(0.3)
        self.move_group.set_planning_time(5.0)
        rospy.loginfo("[Config] 设置最大速度和加速度缩放因子")

    def transform_camera_to_base(self, cam_coords: tuple[float, float, float]) -> tuple[float, float, float]:
        X_cam = np.array(cam_coords).reshape(3, 1)
        X_base = self.R @ X_cam + self.T
        rospy.loginfo(f"[Transform] 相机坐标: {cam_coords} → Base坐标: {X_base.ravel()}")
        return tuple(X_base.ravel())

    def move_to(self, x: float, y: float, z: float, orientation: tuple = (0.707, 0, 0, 0.707)) -> bool:
        x_offset = 0
        x += x_offset
        rospy.loginfo(f"[Motion] 尝试移动到目标点: x={x:.3f}, y={y:.3f}, z={z:.3f}")

        pose = geometry_msgs.msg.Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.x = 0 #orientation[0] #0.707
        pose.orientation.y = 0 #orientation[1]
        pose.orientation.z = 0 #orientation[2]
        pose.orientation.w = 1 #orientation[3] #0.707
        print(f"[Motion] 目标位姿: {pose}")

        self.move_group.set_pose_target(pose)
        plan = self.move_group.plan()

        if isinstance(plan, tuple):
            plan = plan[1]

        if not hasattr(plan, "joint_trajectory") or len(plan.joint_trajectory.points) == 0:
            rospy.logwarn("[Motion] 路径规划失败：无有效轨迹点")
            return False

        success = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

        if success:
            rospy.loginfo("[Motion] 运动成功完成")
        else:
            rospy.logwarn("[Motion] 执行路径失败")

        return success

    
    def compute_orientation_from_mask(self, mask: np.ndarray) -> tuple[float, float, float, float]:
        """
        根据掩码计算目标主轴方向对应的四元数
        """
        import cv2
        from scipy.spatial.transform import Rotation as R

        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise ValueError("掩码区域为空，无法拟合方向")

        largest_contour = max(contours, key=cv2.contourArea)
        if len(largest_contour) < 5:
            raise ValueError("轮廓点太少，无法拟合椭圆")

        ellipse = cv2.fitEllipse(largest_contour)
        angle_deg = ellipse[2]
        angle_rad = np.deg2rad(angle_deg)

        x_axis = np.array([np.cos(angle_rad), np.sin(angle_rad), 0])
        default_x = np.array([1, 0, 0])
        v = np.cross(default_x, x_axis)
        c = np.dot(default_x, x_axis)
        s = np.linalg.norm(v)

        if s < 1e-6:
            rot_matrix = np.eye(3) if c > 0 else -np.eye(3)
        else:
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            rot_matrix = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))

        quat = R.from_matrix(rot_matrix).as_quat()
        return tuple(quat)


    def shutdown(self):
        rospy.loginfo("[Shutdown] 正在关闭 MoveIt 控制器")
        moveit_commander.roscpp_shutdown()
