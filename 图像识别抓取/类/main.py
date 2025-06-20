from Camera import RealSenseCamera, CameraIntrinsics    
from VisionProcessor import ObjectDetector, ObjectLocalizer
from RobotArm import RobotArm
import rospy
import os
from datetime import datetime
from voice_txt import VoiceCommand
from GripperController import GripperController


def main():
    rospy.init_node("object_grasping_system")

    arm = None  # 提前声明，避免 finally 中报错

    try:
        with RealSenseCamera() as camera:
            detector = ObjectDetector("3805134fdf09d4b607da5578f449699b")  # api
            arm = RobotArm()
            gripper = GripperController()

            # === 固定文字提示（未使用语音）===
            prompt = "apple"
            # voice = VoiceCommand()
            # prompt = voice.run()   # 语音转英文

            # === 获取图像帧 ===
            color_img, depth_img, intrinsics = camera.capture_frame()

            # === 保存 RGB 图像 ===
            camera.save_rgb_image(color_img)

            # === 目标检测 ===
            masks, class_ids, _ = detector.detect_objects(color_img, prompt)

            # === 获取质心 ===
            localizer = ObjectLocalizer(intrinsics)
            centroids = localizer.locate_objects(color_img, depth_img, masks)

            # === 可视化掩码 + 质心 ===
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            vis_path = os.path.join("output", f"mask_centroid_{timestamp}.jpg")
            localizer.visualize_masks_with_centroids(color_img, masks, centroids, vis_path)

            # === 控制机械臂移动到目标质心位置 ===
            if centroids and (target := centroids[0]):
                x_cam, y_cam, z_cam = target.world_coords
                x_base, y_base, z_base = arm.transform_camera_to_base((x_cam, y_cam, z_cam))
                mask = masks[0]
                quat = arm.compute_orientation_from_mask(mask)
                gripper.open()
                arm.move_to(x_base, y_base, z_base, orientation=quat)
                rospy.sleep(1.0)
                gripper.close()
                rospy.sleep(5.0)
                arm.move_to(0.068, 0.011, 0.449)

            else:
                print("未检测到目标或掩码为空。")

            # arm.move_to(0.5, 0.0, 0.5)  # 在前方中等高度，肯定能到


    except Exception as e:
        print(f"系统运行出错: {e}")
    finally:
        if arm:  # 只有初始化成功才调用 shutdown
            arm.shutdown()

if __name__ == "__main__":
    main()
