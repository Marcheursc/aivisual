"""
YOLO 模型加载和管理模块
"""

import os
import sys
from ultralytics import YOLO
import torch


class YOLOModelManager:
    """YOLO 模型管理器"""

    def __init__(self, model_dir="yolov12"):
        """
        初始化模型管理器

        Args:
            model_dir: 模型文件目录
        """
        self.model_dir = model_dir
        self.models = {}

    def load_model(self, model_name="yolov12n.pt", device='cuda'):
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

        model_path = os.path.join(self.model_dir, model_name)

        if model_name not in self.models:
            print(f"Loading YOLO model from {model_path}...")
            try:
                model = YOLO(model_path)
                model.to(device)
                self.models[model_name] = {
                    'model': model,
                    'device': device
                }
                print(f"Model {model_name} loaded successfully!")
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
                raise

        return self.models[model_name]['model']

    def get_model_device(self, model_name="yolov12n.pt"):
        """
        获取模型运行设备

        Args:
            model_name: 模型文件名

        Returns:
            设备名称
        """
        if model_name in self.models:
            return self.models[model_name]['device']
        return None

    def set_model_classes(self, model_name="yolov12n.pt", classes=None):
        """
        设置模型检测类别

        Args:
            model_name: 模型文件名
            classes: 类别列表
        """
        if model_name in self.models and classes:
            model = self.models[model_name]['model']
            model.set_classes(classes)
