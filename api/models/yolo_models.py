"""
YOLO 模型加载和管理模块
"""

import os
import sys
from ultralytics import YOLO
import torch

# 全局模型缓存，键为 (模型绝对路径, 设备)
_MODEL_CACHE = {}


class YOLOModelManager:
    """YOLO 模型管理器"""

    def __init__(self, model_dir="yolov12"):
        """
        初始化模型管理器

        Args:
            model_dir: 模型文件目录
        """
        # 获取项目根目录的绝对路径
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = os.path.join(project_root, model_dir) if model_dir == "yolov12" else model_dir
        # 使用全局缓存
        self.models = _MODEL_CACHE

    def load_model(self, model_name="yolov12n.pt", device='cuda', model_dir=None):
        """
        加载 YOLO 模型

        Args:
            model_name: 模型文件名
            device: 运行设备 ('cuda' 或 'cpu')

        Returns:
            YOLO 模型实例
        """
        # 检查设备可用性
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA is not available, falling back to CPU")
            device = 'cpu'

        load_dir = model_dir if model_dir else self.model_dir
        model_path = os.path.join(load_dir, model_name)
        cache_key = (os.path.abspath(model_path), device)
        print(f"Attempting to load model from: {model_path}")

        if cache_key not in self.models:
            print(f"Loading YOLO model from {model_path}...")
            try:
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                    
                model = YOLO(model_path)
                model.to(device)
                self.models[cache_key] = {
                    'model': model,
                    'device': device
                }
                print(f"Model {model_name} loaded successfully!")
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
                raise

        return self.models[cache_key]['model']

    def get_model_device(self, model_name="yolov12n.pt"):
        """
        获取模型运行设备

        Args:
            model_name: 模型文件名

        Returns:
            设备名称
        """
        for (path, device) in self.models:
            if path.endswith(model_name):
                return device
        return None

    def set_model_classes(self, model_name="yolov12n.pt", classes=None):
        """
        设置模型检测类别

        Args:
            model_name: 模型文件名
            classes: 类别列表
        """
        if not classes:
            return
        for (path, _device), entry in self.models.items():
            if path.endswith(model_name):
                entry['model'].set_classes(classes)
