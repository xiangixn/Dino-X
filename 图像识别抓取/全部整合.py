import numpy as np
import cv2
import pyrealsense2 as rs
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from rle_util import rle_to_array
from scipy.ndimage import center_of_mass
import supervision as sv
from dds_cloudapi_sdk import Config, Client
from dds_cloudapi_sdk.image_resizer import image_to_base64
from dds_cloudapi_sdk.tasks.v2_task import V2Task
import rospy
import moveit_commander
import geometry_msgs.msg
from moveit_commander.conversions import pose_to_list
import sys

# 相机内参数据结构
@dataclass
class CameraIntrinsics:
    width: int    
    height: int  
    fx: float     
    fy: float     
    ppx: float    
    ppy: float    

# 检测结果数据结构
@dataclass
class DetectionResult:
    masks: np.ndarray      # 物体掩码数组 [N, H, W] 
    class_ids: np.ndarray  # 类别ID数组 [N]
    class_names: List[str] # 类别名称列表

# 物体中心点数据结构
@dataclass
class ObjectCentroid:
    pixel_coords: Tuple[int, int]    # 中心点像素坐标 (x, y)
    rgb: Tuple[int, int, int]       # RGB颜色值 (r, g, b)
    depth_mm: float                 # 深度值(毫米)
    world_coords: Tuple[float, float, float]  # 世界坐标系 (X, Y, Z) 单位米

# RealSense相机控制类
class RealsenseCamera:
    def __init__(self):
        """初始化RealSense相机"""
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.profile = self.pipeline.start(config)
        
        # 跳过前10帧等待自动曝光稳定
        for _ in range(10):
            self.pipeline.wait_for_frames()
    
    def capture(self) -> Tuple[np.ndarray, np.ndarray, CameraIntrinsics]:
        """捕获一帧彩色和深度图像"""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        # 检查帧是否有效
        if not color_frame or not depth_frame:
            raise RuntimeError("获取彩色或深度帧失败")
        
        # 转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())  # 彩色图像 (480, 640, 3)
        depth_image = np.asanyarray(depth_frame.get_data())  # 深度图像 (480, 640)
        
        # 获取相机内参，并封装为CameraIntrinsics对象
        intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        cam_intrinsics = CameraIntrinsics(
            width=intrinsics.width,
            height=intrinsics.height,
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            ppx=intrinsics.ppx,
            ppy=intrinsics.ppy
        )
        
        return color_image, depth_image, cam_intrinsics
    
    def close(self):
        """关闭相机"""
        self.pipeline.stop()

# DINO物体检测器类
class DINOObjectDetector:
    def __init__(self, api_token: str):
        """初始化检测器"""
        # 创建SDK配置
        config = Config(api_token)
        # 创建客户端
        self.client = Client(config)
    
    def detect(self, image: np.ndarray, text_prompt: str) -> DetectionResult:
        """执行物体检测"""
        # 将图像转换为base64编码
        image_base64 = image_to_base64(image)
        
        # 准备API请求参数
        api_path = "/v2/task/dinox/detection"
        api_body = {
            "model": "DINO-X-1.0",  # 使用DINO-X-1.0模型
            "image": image_base64,   # 输入图像
            "prompt": {"type": "text", "text": text_prompt},  # 文本提示
            "targets": ["mask"],     # 输出掩码
            "bbox_threshold": 0.25, # 边界框阈值
            "iou_threshold": 0.8     # IOU阈值
        }
        
        # 创建并运行任务
        task = V2Task(api_path=api_path, api_body=api_body)
        self.client.run_task(task)
        # 获取结果
        result = task.result
        objects = result["objects"]
        
        # 处理掩码数据
        masks = []
        class_ids = []
        # 分割文本提示中的类别
        classes = [x.strip().lower() for x in text_prompt.split('.') if x]
        # 创建类别名到ID的映射
        class_name_to_id = {name: i for i, name in enumerate(classes)}
        
        # 处理每个检测到的物体
        for obj in objects:
            # 解析RLE编码的掩码
            mask_rle = obj["mask"]
            mask_array = rle_to_array(mask_rle["counts"], mask_rle["size"][0] * mask_rle["size"][1])
            mask = mask_array.reshape(mask_rle["size"])
            masks.append(mask.astype(bool))
            
            # 获取类别名称并转换为ID
            cls_name = obj["category"].lower().strip()
            class_ids.append(class_name_to_id.get(cls_name, -1))  # -1表示未知类别
        
        # 返回检测结果
        return DetectionResult(
            masks=np.array(masks),
            class_ids=np.array(class_ids),
            class_names=classes
        )

# 物体定位器类
class ObjectLocalizer:
    def __init__(self, intrinsics: CameraIntrinsics):
        """初始化定位器"""
        self.intrinsics = intrinsics  # 相机内参
    
    def calculate_centroids(self, 
                          color_image: np.ndarray,
                          depth_image: np.ndarray,
                          detection_result: DetectionResult) -> List[ObjectCentroid]:
        """计算物体中心点的3D坐标"""
        centroids = []
        
        # 遍历每个检测到的物体
        for i, mask in enumerate(detection_result.masks):
            # 跳过空掩码
            if mask.sum() == 0:
                centroids.append(None)
                continue
            
            # 计算掩码的质心 (cy是行/y，cx是列/x)
            cy, cx = center_of_mass(mask)
            cx, cy = int(cx), int(cy)
            
            # 获取RGB颜色值 (OpenCV是BGR顺序)
            b, g, r = color_image[cy, cx]
            # 获取深度值(毫米)
            depth_mm = depth_image[cy, cx]
            
            # 将像素坐标转换为3D相机坐标
            point3d = rs.rs2_deproject_pixel_to_point(
                [self.intrinsics.fx, self.intrinsics.fy, self.intrinsics.ppx, self.intrinsics.ppy],
                [cx, cy],
                depth_mm / 1000.0  # 毫米转米
            )
            X, Y, Z = point3d  # 单位: 米
            
            # 保存中心点信息
            centroids.append(ObjectCentroid(
                pixel_coords=(cx, cy),
                rgb=(r, g, b),
                depth_mm=depth_mm,
                world_coords=(X, Y, Z)
            ))
        
        return centroids
    
    def visualize(self, 
                 color_image: np.ndarray,
                 centroids: List[ObjectCentroid]) -> np.ndarray:
        """在图像上可视化中心点"""
        # 创建图像副本
        annotated = color_image.copy()
        
        # 绘制每个中心点
        for centroid in centroids:
            if centroid is None:
                continue
                
            # 获取像素坐标
            cx, cy = centroid.pixel_coords
            # 绘制红色圆点标记中心点
            cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
            # 在中心点旁边显示深度值
            cv2.putText(annotated, f"{centroid.depth_mm:.0f}mm", 
                       (cx+6, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return annotated

# 机械臂控制类
class RobotArmController:
    def __init__(self, group_name="manipulator"):
        """初始化机械臂控制器"""
        # 初始化MoveIt
        moveit_commander.roscpp_initialize(sys.argv)
        # 创建MoveGroup
        self.move_group = moveit_commander.MoveGroupCommander(group_name)
        # 设置运动参数
        self.move_group.set_max_velocity_scaling_factor(0.2)  # 最大速度比例
        self.move_group.set_max_acceleration_scaling_factor(0.3)  # 最大加速度比例
    
    def move_to_position(self, x: float, y: float, z: float):
        """移动机械臂到指定位置"""
        # 创建目标位姿
        target_pose = geometry_msgs.msg.Pose()
        # 设置位置 (单位: 米)
        target_pose.position.x = x
        target_pose.position.y = y
        target_pose.position.z = z
        
        # 设置姿态 (夹爪朝前)
        target_pose.orientation.x = 0.707  # 四元数表示
        target_pose.orientation.y = 0.0
        target_pose.orientation.z = 0.0
        target_pose.orientation.w = 0.707
        
        # 设置目标位姿
        self.move_group.set_pose_target(target_pose)
        # 规划路径
        plan = self.move_group.plan()
        
        # 检查规划是否成功
        if plan and len(plan.joint_trajectory.points) > 0:
            rospy.loginfo("规划成功，正在执行运动...")
            self.move_group.go(wait=True)  # 执行运动
        else:
            rospy.logwarn("规划失败，请检查目标是否可达")
        
        # 停止并清除目标
        self.move_group.stop()
        self.move_group.clear_pose_targets()
    
    def shutdown(self):
        """关闭控制器"""
        moveit_commander.roscpp_shutdown()

def main():
    """主函数"""
    # 初始化ROS节点
    rospy.init_node("object_detection_and_grasping")
    
    try:
        # 1. 初始化各组件
        camera = RealsenseCamera()  # 相机
        detector = DINOObjectDetector(api_token="8f14d9c9dcd2c27a5f3a4fb29a762c58")  # 检测器
        arm_controller = RobotArmController()  # 机械臂控制器
        
        # 2. 捕获图像
        color_image, depth_image, intrinsics = camera.capture()
        
        # 3. 检测物体 (这里检测"apple")
        detection_result = detector.detect(color_image, text_prompt="apple")
        
        # 4. 计算物体3D位置
        localizer = ObjectLocalizer(intrinsics)
        centroids = localizer.calculate_centroids(color_image, depth_image, detection_result)
        
        # 5. 可视化结果
        annotated_image = localizer.visualize(color_image, centroids)
        cv2.imwrite("annotated_result.jpg", annotated_image)  # 保存可视化结果
        
        # 6. 控制机械臂移动到第一个检测到的物体
        if centroids and centroids[0] is not None:
            x, y, z = centroids[0].world_coords  # 获取世界坐标
            arm_controller.move_to_position(x, y, z)  # 移动机械臂
        
    except Exception as e:
        rospy.logerr(f"发生错误: {str(e)}")
    finally:
        # 确保资源释放
        camera.close()
        arm_controller.shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()