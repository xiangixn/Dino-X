# DINO-X-Z1: Language-Driven Visual Grasping with Unitree Z1

结合 DINO-X、语音识别与视觉感知技术，构建一个支持自然语言输入并驱动 Unitree Z1 机械臂执行目标抓取任务的智能机器人系统。

> 本项目参考自 [MIAA_SIM (rekep 分支)](https://github.com/Khlann/MIAA_SIM/tree/rekep)，并针对实际硬件部署进行了拓展优化。

---

## 项目简介 / Introduction

系统整体流程：

1. 用户通过麦克风输入语音指令；
2. 使用语义提取模块（如豆包 API）提取目标关键词；
3. 使用 Realsense 相机获取图像
4. 关键词传递给 DINO-X，进行目标检测与掩码生成；
5. 计算掩码质心作为抓取点；
6. 控制 Unitree Z1 机械臂完成抓取动作。

系统特点：

- 自然语言语音交互  
- 多模态理解与图像目标定位  
- 通用目标抓取能力，模块化部署  

---

## 项目结构 / Project Structure

```text
project_root/
├── camera_data/                         # 相机数据保存路径
├── 图像识别抓取/
│   ├── 类/                              # 整理好的主控代码
│   ├── 粗糙/                            # 初版测试代码
│   ├── 全部整合.py                      # 抓取完整流程脚本
│   └── test.py                          # 单模块测试脚本
├── 标定/hand_eye_calibration-main/     # 手眼标定工具包
├── 语音转文字/                          # 本地语音转文字工具（含 ffmpeg）
├── 工作范围.py                          # 抓取工作空间可视化
├── main.py                              # 主程序入口
````

---

## 安装说明 / Installation

### 系统要求

* Ubuntu 20.04+
* ROS Noetic
* Python 3.8+

### 安装步骤

#### 1. 安装 Z1 控制环境

参考官方文档：[Z1 ROS 安装指南](https://github.com/unitreerobotics/z1_ros/blob/noetic/doc/setup.md)

#### 2. 安装 RealSense SDK

```bash
sudo apt-get install librealsense2-dkms librealsense2-utils librealsense2-dev librealsense2-dbg
```

#### 3. 安装手眼标定依赖

```bash
cd 标定/hand_eye_calibration-main/
pip install -r requirements.txt
```

---

## 硬件支持 / Hardware Support

* Unitree Z1 机械臂
* Intel RealSense D435 / D405 相机
* USB 麦克风（或支持文本输入）

---

## 快速上手 / Quick Start

### 1. 启动 Z1 控制

```bash
source /opt/ros/noetic/setup.bash
source ~/z1_ws/devel/setup.bash
roslaunch z1_bringup real_arm.launch rviz:=true
```

---

### 2. 手眼标定（首次运行必做）

我们做的是“眼在手外”标定，因此需要打印棋盘格标定板，并固定在机械臂末端（夹爪上），且其相对位置必须在采集过程中保持不变。

标定步骤如下：

1. 打印棋盘格标定板：
   [https://calib.io/pages/camera-calibration-pattern-generator](https://calib.io/pages/camera-calibration-pattern-generator)

2. 修改 `Config.yaml` 文件，配置棋盘格尺寸、角点数等参数

3. 连接相机并采集图像：

```bash
python collect_data.py
```

按下 `s` 键保存图像，建议采集 15\~20 组图像

4. 计算手眼变换矩阵：

```bash
python compute_to_hand.py
```

5. 将结果写入 `RobotArm.py` 中：

```python
self.R = np.array([[...], [...], [...]])
self.T = np.array([[...], [...], [...]])
```

---

### 3. 配置 API Key

* 获取 DINO-X API：
  [https://cloud.deepdataspace.com/apply-token?from=github](https://cloud.deepdataspace.com/apply-token?from=github)

* 获取豆包 API：
  [https://console.volcengine.com/ark/region\:ark+cn-beijing/apiKey](https://console.volcengine.com/ark/region:ark+cn-beijing/apiKey)

修改 `main.py` 中如下代码：

```python
detector = ObjectDetector("你的DINOX_API")
api_key = "你的豆包API")
```

---

### 4. 运行主程序

```bash
cd 图像识别抓取/类/
python main.py
```

---

## 联系方式 / Contact

如有问题请联系：

* 邮箱：[widebbie0923@mail.scut.edu.cn](mailto:widebbie0923@mail.scut.edu.cn)


