# coding=utf-8
import json
import logging, os, sys
import time
import numpy as np
import cv2
import pyrealsense2 as rs

from libs.log_setting import CommonLog
from libs.auxiliary import create_folder_with_date, get_ip, popup_message

import rospy
import tf2_ros
from tf.transformations import euler_from_quaternion

# ========== 路径准备 ==========
cam0_origin_path = create_folder_with_date()

# ========== 日志系统 ==========
logger_ = logging.getLogger(__name__)
logger_ = CommonLog(logger_)

# ========== TF 获取机械臂姿态 ==========
def get_pose_from_tf(base_frame="link00", ee_frame="link06"):
    try:
        tf_buffer = get_pose_from_tf.tf_buffer
        if tf_buffer is None:
            tf_buffer = tf2_ros.Buffer()
            tf2_ros.TransformListener(tf_buffer)
            get_pose_from_tf.tf_buffer = tf_buffer
            rospy.sleep(1.0)

        trans = tf_buffer.lookup_transform(base_frame, ee_frame, rospy.Time(0), rospy.Duration(1.0))
        t = trans.transform.translation
        r = trans.transform.rotation

        quat = [r.x, r.y, r.z, r.w]
        roll, pitch, yaw = euler_from_quaternion(quat)

        pose = [t.x, t.y, t.z, roll, pitch, yaw]
        return True, pose
    except Exception as e:
        logger_.error(f"[TF] 获取姿态失败: {e}")
        return False, f"TF 获取姿态失败: {e}"

get_pose_from_tf.tf_buffer = None

# ========== 图像回调函数 ==========
def callback(frame):
    scaling_factor = 2.0
    global count

    cv_img = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    cv2.imshow("Capture_Video", cv_img)

    k = cv2.waitKey(30) & 0xFF

    if k == ord('s'):
        state, pose = get_pose_from_tf("link00", "link06")
        logger_.info(f'获取状态：{"成功" if state else "失败"}，{f"当前位姿为{pose}" if state else None}')
        if state:
            filename = os.path.join(cam0_origin_path, "poses.txt")
            with open(filename, 'a+') as f:
                pose_ = [str(i) for i in pose]
                new_line = f'{",".join(pose_)}\n'
                f.write(new_line)

            image_path = os.path.join(cam0_origin_path, f"{str(count)}.jpg")
            cv2.imwrite(image_path, cv_img)
            logger_.info(f"===采集第{count}次数据！")
            count += 1
    elif k == ord('q'):
        logger_.info("用户按下 q 键，退出采集")
        rospy.signal_shutdown("用户退出")

# ========== 相机主循环 ==========
def displayD435():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        pipeline.start(config)
    except Exception as e:
        logger_.error(f"相机连接异常：{e}")
        popup_message("提醒", "相机连接异常")
        sys.exit(1)

    global count
    count = 1

    logger_.info(f"开始手眼标定程序，当前程序版号 V1.0.0")

    try:
        while not rospy.is_shutdown():
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            callback(color_image)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

# ========== 主函数 ==========
if __name__ == '__main__':
    rospy.init_node("hand_eye_tf_capture", anonymous=True)

    logger_.info("初始化 ROS 节点成功")

    displayD435()
