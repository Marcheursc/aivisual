"""
横幅检测算法实现
基于YOLOv12的目标检测算法
"""

import cv2
import numpy as np
import sys
import torch
import os
from ultralytics import YOLO

class BannerDetector:
    def __init__(self, model_path="yolov12/yolov12n.pt", conf_threshold=0.3, iou_threshold=0.45, img_size=640, device='cuda'):
        """
        初始化横幅检测器

        Args:
            model_path (str): YOLOv12模型路径
            conf_threshold (float): 置信度阈值
            iou_threshold (float): NMS IoU阈值
            img_size (int): 图像处理尺寸
            device (str): 运行设备 ('cuda' 或 'cpu')
        """
        print(f"Loading YOLOv12 model from {model_path}...")
        try:
            # 检查设备可用性
            if device == 'cuda' and not torch.cuda.is_available():
                print("CUDA is not available, falling back to CPU")
                device = 'cpu'

            print(f"Using device: {device}")

            # 加载YOLOv12模型
            self.model = YOLO(model_path)
            self.device = device

            # 将模型移动到指定设备
            self.model.to(device)

            print(f"Model loaded successfully! Detecting classes: {list(self.model.names.values())}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please ensure you have downloaded the yolov12 model file")
            sys.exit(1)

        # 配置参数
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size  # 降低图像尺寸以提高速度

        # 存储检测到的横幅信息
        self.detected_banners = []

    def detect_banner(self, frame):
        """
        检测视频帧中的横幅

        Args:
            frame: 视频帧

        Returns:
            results: 检测结果
            banners: 横幅信息
        """
        # 使用YOLOv12检测目标
        results = self.model(
            frame,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False  # 关闭推理日志输出
        )

        banners = []

        for r in results:
            boxes = r.boxes  # 检测框信息
            if boxes is not None:
                for box in boxes:
                    # 获取检测框坐标（转换为整数）
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # 获取置信度和类别
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    cls_name = self.model.names[cls]  # 类别名称

                    # 只处理横幅类别
                    if cls_name == "banner":
                        banners.append({
                            'box': (x1, y1, x2, y2),
                            'confidence': conf,
                            'class': cls_name,
                            'area': (x2 - x1) * (y2 - y1)
                        })

        # 更新检测到的横幅信息
        self.detected_banners = banners

        return results, banners

    def draw_detections(self, frame, banners):
        """
        在帧上绘制检测结果

        Args:
            frame: 视频帧
            banners: 检测到的横幅信息

        Returns:
            frame_with_detections: 绘制了检测结果的视频帧
        """
        # 绘制检测框
        for banner in banners:
            x1, y1, x2, y2 = banner['box']
            conf = banner['confidence']
            cls_name = banner['class']

            # 绘制检测框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制标签和置信度
            label_text = f"{cls_name}: {conf:.2f}"
            # 绘制文本背景（提高可读性）
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            text_bg_x2 = x1 + text_size[0] + 6
            text_bg_y2 = y1 - text_size[1] - 6
            cv2.rectangle(frame, (x1, y1), (text_bg_x2, text_bg_y2), (0, 255, 0), -1)

            # 绘制文本
            cv2.putText(
                frame, label_text, (x1 + 3, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )

        return frame