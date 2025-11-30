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
DEFAULT_BANNER_CONFIDENCE_THRESHOLD = 0.5  # 横幅检测置信度阈值
DEFAULT_BANNER_IOU_THRESHOLD = 0.45  # 横幅检测IOU阈值

# 默认ROI区域配置
DEFAULT_LEAVE_ROI = [(600, 100), (1000, 100), (1000, 700), (600, 700)]
DEFAULT_GATHER_ROI = [(220, 300), (700, 300), (700, 700), (200, 700)]
DEFAULT_BANNER_ROI = [(0, 0), (1280, 0), (1280, 720), (0, 720)]  # 默认全屏检测

# RabbitMQ配置
RABBITMQ_HOST = "127.0.0.1"
RABBITMQ_PORT = 5672
RABBITMQ_USERNAME = "guest"
RABBITMQ_PASSWORD = "guest"
RABBITMQ_VIRTUAL_HOST = "/"

# 交换机配置
RABBITMQ_EXCHANGE = "alarm_exchange"
RABBITMQ_EXCHANGE_TYPE = "direct"

# 队列配置
RABBITMQ_QUEUE = "alarm_queue"
RABBITMQ_ROUTING_KEY = "alarm_routing_key"

# 消息配置
RABBITMQ_MESSAGE_DURABLE = True

# 连接重试配置
RABBITMQ_CONNECTION_RETRIES = 3
RABBITMQ_CONNECTION_RETRY_DELAY = 5
