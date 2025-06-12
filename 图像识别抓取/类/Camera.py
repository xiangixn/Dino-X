import pyrealsense2 as rs
import numpy as np
import cv2
from dataclasses import dataclass
from datetime import datetime
import os

# 定义一个结构体，用于存储相机内参信息
@dataclass
class CameraIntrinsics:
    width: int   
    height: int   
    fx: float    
    fy: float     
    ppx: float   
    ppy: float
    raw_intrinsics: rs.intrinsics    

# RealSense 相机封装类
class RealSenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()      
        config = rs.config()                
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.profile = self.pipeline.start(config)
        self._warmup_camera()

    # 跳过前几帧，等待自动曝光等稳定
    def _warmup_camera(self):
        for _ in range(10):
            self.pipeline.wait_for_frames()

    # 获取一帧图像（RGB 和深度）以及相机内参
    def capture_frame(self) -> tuple[np.ndarray, np.ndarray, CameraIntrinsics]:
        frames = self.pipeline.wait_for_frames()  
        color_frame = frames.get_color_frame()    
        depth_frame = frames.get_depth_frame()    
        
        if not color_frame or not depth_frame:
            raise RuntimeError("无法获取彩色图像或深度图像")

        # 将 RealSense 图像帧转换为 NumPy 数组
        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())
        # 获取相机内参
        intrinsics = self._get_intrinsics()
        
        return color_img, depth_img, intrinsics

    # 获取彩色图像的相机内参并封装为 CameraIntrinsics 类
    def _get_intrinsics(self) -> CameraIntrinsics:
        stream_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = stream_profile.get_intrinsics()

        return CameraIntrinsics(
            width=intrinsics.width,
            height=intrinsics.height,
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            ppx=intrinsics.ppx,
            ppy=intrinsics.ppy,
            raw_intrinsics=intrinsics  #  把原始内参对象传进去
        )

    
    # 保存 RGB 图像到指定路径（带时间戳）
    def save_rgb_image(self, color_img: np.ndarray, save_dir: str = "output"):
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"rgb_{timestamp}.png")
        cv2.imwrite(save_path, color_img)
        print(f"[✔] RGB 图像已保存到: {save_path}")


    # 支持 with 上下文管理器，自动关闭 pipeline
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pipeline.stop()
