import numpy as np
import cv2
from pathlib import Path
from scipy.ndimage import center_of_mass
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# 获取彩色相机的内参（也可以改成 rs.stream.depth 获取深度相机内参）
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

pipeline.stop()

# ==== 路径设置 ====
MASK_PATH = Path("./outputs/masks.npy")
CAMERA_DIR = Path("./camera_data")
IMG_PATH = str(CAMERA_DIR / "camera_capture_d435.png")
DEPTH_PATH = str(CAMERA_DIR / "camera_depth_d435.npy")
OUTPUT_PATH = Path("./outputs/centroid_rgbd_vis.jpg")

# ==== 加载图像与掩码 ====
masks = np.load(MASK_PATH)              # shape: [N, H, W], bool
img = cv2.imread(IMG_PATH)              # shape: (H, W, 3), BGR
depth = np.load(DEPTH_PATH)            # shape: (H, W), uint16 (单位: mm)

centroids = []

# ==== 遍历掩码，计算质心并提取 RGBD + XYZ 信息 ====
for i, mask in enumerate(masks):
    if mask.sum() == 0:
        centroids.append(None)
        print(f"[!] Mask {i} 无有效区域，跳过")
        continue

    # 质心坐标 (cy 是行/y，cx 是列/x)
    cy, cx = center_of_mass(mask)
    cx, cy = int(cx), int(cy)
    centroids.append((cx, cy))

    # 提取 RGB 值（OpenCV 是 BGR 顺序）
    b, g, r = img[cy, cx]

    # 提取深度值（单位：毫米 → 米）
    depth_mm = depth[cy, cx]
    depth_m = depth_mm / 1000.0

    # 转换为 3D 相机坐标
    point3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth_m)
    X, Y, Z = point3d  # 单位：米

    # 可视化标记
    cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
    cv2.putText(img, f"{depth_mm}mm", (cx+6, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # 打印 RGBD + 3D 坐标
    print(f"Mask {i} ➤ 像素质心: ({cx}, {cy}) | RGB: ({r}, {g}, {b}) | Depth: {depth_mm} mm")
    print(f"        ➤ 空间坐标: X={X:.3f} m, Y={Y:.3f} m, Z={Z:.3f} m")

# ==== 保存标注后的图像 ====
cv2.imwrite(str(OUTPUT_PATH), img)
