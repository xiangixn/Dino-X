# 📘 DINO-X-Z1：支持语音输入的视觉抓取系统

结合 DINO-X、语音识别与视觉感知技术，构建一个支持自然语言输入并驱动 Unitree Z1 机械臂执行目标抓取任务的智能机器人系统。

本项目参考自 [MIAA_SIM 项目](https://github.com/Khlann/MIAA_SIM/tree/rekep)，并结合实际硬件部署需求进行修改和拓展。

---

## 📖 项目简介 / Introduction

本项目实现了一个结合语音输入与视觉检测的通用抓取系统。用户通过麦克风输入语音指令，经语义提取模块（基于豆包等大模型）提取目标关键词，传递给 DINO-X 进行图像中目标检测与掩码生成。通过 Realsense 相机获取的图像，在掩码区域内计算质心作为抓取点，并驱动 Unitree Z1 机械臂完成抓取动作。系统特点是自然语言交互、目标自定位、低成本部署。

---

## 🧠 项目结构 / Project Structure

```text
project_root/
├── camera_data/                         # 相机采集数据保存路径
├── 图像识别抓取/                        # 目标检测与抓取控制模块
│   ├── 类/                              # 整理好的完整抓取代码
│   ├── 粗糙/                            # 初版代码
│   ├── rle_util.py                      
│   ├── 全部整合.py                      # 完整流程整合脚本
│   └── test.py                          # 模块测试脚本
├── 标定/hand_eye_calibration-main/      # 相机与机械臂的手眼标定工具（眼在手外）
├── 语音转文字/ffmpeg-n7.1.../           # 本地语音转文字模块（含 ffmpeg）
├── 工作范围.py                          # 工作范围/抓取区域预设脚本
└── main.py                               # 主运行脚本，调用语音识别 + 视觉检测 + 控制流程
```

---

## 🔧 安装方式 / Installation

### 基本依赖
```
todo
```

### 硬件支持
- ✅ Unitree Z1 机械臂
- ✅ Realsense D435 / D405 相机
- ✅ 麦克风输入设备（或支持文本输入）

---

## 🚀 快速上手 / Quick Start

```bash
# z1控制

# 抓取

```

---

## 📎 引用 / Citation

```bibtex
@misc{dino_x_z1_2025,
  title={DINO-X-Z1: Language-Driven Vision-Based Robotic Grasping},
  author={Your Name et al.},
  year={2025},
  note={Internal Project}
}
```

---

## 🙋 联系方式 / Contact

如有问题请联系：widebbie0923@mail.scut.edu.cn
