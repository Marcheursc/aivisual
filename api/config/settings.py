"""
项目配置文件
"""

import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据目录配置
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_videos")

# 模型配置
MODEL_DIR = os.path.join(BASE_DIR, "..", "yolov12")
DEFAULT_MODEL = "yolov12n.pt"

# 创建必要的目录
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 默认参数配置
DEFAULT_LOITERING_THRESHOLD = 20  # 徘徊检测阈值(秒)
DEFAULT_LEAVE_THRESHOLD = 5       # 离岗检测阈值(秒)
DEFAULT_GATHER_THRESHOLD = 5      # 聚集检测阈值(人)

# 默认ROI区域配置
DEFAULT_LEAVE_ROI = [(600, 100), (1000, 100), (1000, 700), (600, 700)]
DEFAULT_GATHER_ROI = [(220, 300), (700, 300), (700, 700), (200, 700)]
