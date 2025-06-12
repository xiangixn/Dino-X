import pyrealsense2 as rs

# ==== D435 图像采集 ====
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

# 跳过最开始几帧（用于自动曝光稳定）
for _ in range(10):
    pipeline.wait_for_frames()

frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()
pipeline.stop()

if not color_frame:
    raise RuntimeError("[❌] 未能获取 D435 彩色图像。")

# 将图像转为 numpy 数组并保存
color_image = np.asanyarray(color_frame.get_data())
IMG_PATH = "camera_capture_d435.png"
cv2.imwrite(IMG_PATH, color_image)