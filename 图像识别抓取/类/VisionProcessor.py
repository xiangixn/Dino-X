import numpy as np
import cv2
import os
import pyrealsense2 as rs
from scipy.ndimage import center_of_mass
from dataclasses import dataclass
from typing import List, Optional
from rle_util import rle_to_array
from io import BytesIO
from PIL import Image
from Camera import CameraIntrinsics
from dds_cloudapi_sdk.image_resizer import image_to_base64
from dds_cloudapi_sdk.tasks.v2_task import V2Task


@dataclass
class ObjectCentroid:
    pixel_coords: tuple[int, int]
    rgb: tuple[int, int, int]
    depth_mm: float
    world_coords: tuple[float, float, float]


class ObjectDetector:
    def __init__(self, api_token: str):
        from dds_cloudapi_sdk import Config, Client
        self.client = Client(Config(api_token))

    def detect_objects(self, image: np.ndarray, text_prompt: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        buffer = BytesIO()
        image_pil.save(buffer, format='PNG')
        buffer.seek(0)
        image_base64 = image_to_base64(buffer)

        task = V2Task(
            api_path="/v2/task/dinox/detection",
            api_body={
                "model": "DINO-X-1.0",
                "image": image_base64,
                "prompt": {"type": "text", "text": text_prompt},
                "targets": ["mask"],
                "bbox_threshold": 0.25,
                "iou_threshold": 0.8
            }
        )
        self.client.run_task(task)
        return self._process_result(task.result, text_prompt)

    def _process_result(self, result, text_prompt) -> tuple[np.ndarray, np.ndarray, list[str]]:
        objects = result["objects"]
        classes = [x.strip().lower() for x in text_prompt.split('.') if x]
        class_map = {name: i for i, name in enumerate(classes)}

        masks, class_ids = [], []
        for obj in objects:
            mask_rle = obj["mask"]
            mask = rle_to_array(mask_rle["counts"], mask_rle["size"][0] * mask_rle["size"][1])
            masks.append(mask.reshape(mask_rle["size"]).astype(bool))
            class_ids.append(class_map.get(obj["category"].lower().strip(), -1))

        return np.array(masks), np.array(class_ids), classes


class ObjectLocalizer:
    def __init__(self, intrinsics: CameraIntrinsics):
        self.intrinsics = intrinsics

    def locate_objects(self, color_img: np.ndarray, depth_img: np.ndarray, masks: np.ndarray) -> List[Optional[ObjectCentroid]]:
        centroids = []
        for mask in masks:
            if not mask.any():
                centroids.append(None)
                continue

            cy, cx = center_of_mass(mask)
            cx, cy = int(cx), int(cy)

            b, g, r = color_img[cy, cx]
            depth_mm = depth_img[cy, cx]

            x, y, z = rs.rs2_deproject_pixel_to_point(
                self.intrinsics.raw_intrinsics,  # 修复参数错误
                [cx, cy],
                depth_mm / 1000.0
            )

            centroids.append(ObjectCentroid(
                pixel_coords=(cx, cy),
                rgb=(r, g, b),
                depth_mm=depth_mm,
                world_coords=(x, y, z)
            ))

        print(f"[INFO] 目标质心在相机坐标系下的坐标: x={x:.3f}, y={y:.3f}, z={z:.3f}")
        
        return centroids

    def visualize_masks_with_centroids(self, color_img: np.ndarray, masks: np.ndarray,
                                       centroids: List[Optional[ObjectCentroid]],
                                       save_path: str = "output/mask_centroid_vis.jpg"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        overlay = color_img.copy()

        for i, (mask, centroid) in enumerate(zip(masks, centroids)):
            if centroid is None:
                continue
            colored_mask = np.zeros_like(color_img, dtype=np.uint8)
            colored_mask[mask] = [0, 0, 255]
            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)
            cx, cy = centroid.pixel_coords
            cv2.circle(overlay, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(overlay, f"({cx},{cy})", (cx+5, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imwrite(save_path, overlay)
        print(f"[✔] 掩码可视化图已保存到: {save_path}")
