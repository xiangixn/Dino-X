from dds_cloudapi_sdk import Config, Client
from dds_cloudapi_sdk.image_resizer import image_to_base64
from dds_cloudapi_sdk.tasks.v2_task import V2Task

import supervision as sv
import numpy as np
import cv2, os
from pathlib import Path
from rle_util import rle_to_array
import pyrealsense2 as rs

# ==== D435 图像采集 ====
# 初始化 RealSense 流
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

# 跳过前几帧
for _ in range(10):
    pipeline.wait_for_frames()

frames = pipeline.wait_for_frames()

color_frame = frames.get_color_frame()
depth_frame = frames.get_depth_frame()
pipeline.stop()

if not color_frame or not depth_frame:
    raise RuntimeError("[❌] 未能获取 D435 彩色图像或深度图像。")

# 转为 numpy 数组
color_image = np.asanyarray(color_frame.get_data())         # shape: (480, 640, 3)
depth_image = np.asanyarray(depth_frame.get_data())         # shape: (480, 640)

# 保存图像
CAMERA_DIR = Path("./camera_data")
CAMERA_DIR.mkdir(parents=True, exist_ok=True)

IMG_PATH = str(CAMERA_DIR / "camera_capture_d435.png")
DEPTH_PATH_PNG = str(CAMERA_DIR / "camera_depth_d435.png")
DEPTH_PATH_NPY = str(CAMERA_DIR / "camera_depth_d435.npy")

cv2.imwrite(IMG_PATH, color_image)
cv2.imwrite(DEPTH_PATH_PNG, depth_image)
np.save(DEPTH_PATH_NPY, depth_image)

print(f"[✔] RGB 图像已保存到: {IMG_PATH}")
print(f"[✔] Depth 图像已保存到: {DEPTH_PATH_PNG} / {DEPTH_PATH_NPY}")



# ==== 配置部分 ====
API_TOKEN = "8f14d9c9dcd2c27a5f3a4fb29a762c58"  
# IMG_PATH = "1.png"
TEXT_PROMPT = "cap"  # 类别之间用 . 分隔，如 "duck . bottle"
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==== 初始化 SDK ====
config = Config(API_TOKEN)
client = Client(config)

# ==== 图像转 base64 并构造请求 ====
image = image_to_base64(IMG_PATH)
api_path = "/v2/task/dinox/detection"
api_body = {
    "model": "DINO-X-1.0",
    "image": image,
    "prompt": {"type": "text", "text": TEXT_PROMPT},
    "targets": ["mask"],
    "bbox_threshold": 0.25,
    "iou_threshold": 0.8
}
task = V2Task(api_path=api_path, api_body=api_body)
client.run_task(task)
result = task.result
objects = result["objects"]

# ==== 掩码提取 ====
masks = []
class_ids = []
classes = [x.strip().lower() for x in TEXT_PROMPT.split('.') if x]
class_name_to_id = {name: i for i, name in enumerate(classes)}

for obj in objects:
    mask_rle = obj["mask"]
    mask_array = rle_to_array(mask_rle["counts"], mask_rle["size"][0] * mask_rle["size"][1])
    mask = mask_array.reshape(mask_rle["size"])
    masks.append(mask.astype(bool))

    cls_name = obj["category"].lower().strip()
    class_ids.append(class_name_to_id.get(cls_name, -1))  # 用 -1 代表未匹配类别

# ==== Supervision 可视化（只用掩码） ====
masks = np.array(masks)
np.save(OUTPUT_DIR / "masks.npy", masks)

class_ids = np.array(class_ids)
img = cv2.imread(IMG_PATH)
dummy_boxes = np.zeros((len(masks), 4), dtype=np.int32)  # 提供假的框以符合 API 要求

detections = sv.Detections(
    xyxy=dummy_boxes,
    mask=masks,
    class_id=class_ids,
)

mask_annotator = sv.MaskAnnotator()
annotated_frame = mask_annotator.annotate(scene=img.copy(), detections=detections)

# ==== 保存输出 ====
output_path = OUTPUT_DIR / "annotated_mask_only.jpg"
cv2.imwrite(str(output_path), annotated_frame)
print(f"[✔] 掩码图像已保存到: {output_path}")
