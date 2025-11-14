# AI-Visual Detection System

## 简介

这是一个基于YOLOv12的AI视频智能检测系统，具备目标检测、跟踪和多种行为分析功能。

## 功能特性

- 基于YOLOv12的目标检测
- 多目标跟踪（ByteTrack）
- 徘徊行为检测与报警
- 离岗检测与报警
- 人员聚集检测与报警
- 支持多种目标类别检测（人、车等）
- 可视化显示检测结果和报警信息

## 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装lap库（ByteTrack需要）
pip install lap>=0.5.12

# 安装PyTorch相关库（根据你的环境选择合适的版本）
# CUDA 12.1版本
pip install torch==2.8.0+cu121 torchvision==0.13.0+cu121 torchaudio==2.0.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# CPU版本
pip install torch==2.8.0+cpu torchvision==0.13.0+cpu torchaudio==2.0.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
```

## 运行系统

系统包含多个功能模块，可以根据需要选择运行：

```bash
# 运行徘徊检测
python main.py

# 运行离岗检测
python leave_gather_detection/leave.py

# 运行人员聚集检测
python leave_gather_detection/main.py

# 同时运行离岗和人员聚集检测
python leave_gather_detection/whole.py
```
```bash
# 运行web服务器
python api/cv_api.py
python api/flask_app.py
```

## 模块说明

系统功能已按用途分类组织：

1. **徘徊检测模块** - 位于 `detector/` 目录
   - 核心实现在 `detector/core.py`
   - 使用说明请查看 `detector/README.md`

2. **离岗和聚集检测模块** - 位于 `leave_gather_detection/` 目录
   - 离岗检测: `leave.py`
   - 人员聚集检测: `main.py`
   - 综合检测: `whole.py`
   - 使用说明请查看 `leave_gather_detection/README.md`

## 故障排除

1. **numpy兼容性问题 (AttributeError: module 'numpy' has no attribute 'object')**:
   - 系统已自动处理此问题，在detector/core.py和test.py中添加了兼容性代码
   - 无需额外操作

2. **ByteTrack跟踪器不可用**:
   - 确保已安装lap库: `pip install lap>=0.5.12`
   - 系统会自动检测并使用ByteTrack，如果不可用会回退到基础跟踪方法
3. **CUDA相关问题**:
   - 如果没有GPU或CUDA不可用，系统会自动回退到CPU运行
   - 确保安装了正确版本的PyTorch和CUDA驱动

## 应用场景

- 安全监控
- 入侵检测
- 公共空间监控
- 智慧城市应用
