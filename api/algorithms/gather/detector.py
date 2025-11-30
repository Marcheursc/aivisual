"""
聚集检测算法实现
基于YOLOv12的目标检测算法
"""

import cv2
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO
import torch
import logging
import time

# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GatherDetector:
    def __init__(self, model_path="yolov12/yolov12n.pt", device='cuda', img_size=640):
        """
        初始化聚集检测器

        Args:
            model_path (str): YOLOv12模型路径
            device (str): 运行设备 ('cuda' 或 'cpu')
            img_size (int): 图像处理尺寸（较小的尺寸可以提高速度）
        """
        # 检查设备可用性
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA is not available, falling back to CPU")
            device = 'cpu'

        # 使用 YOLOModelManager 加载模型
        from ...models.yolo_models import YOLOModelManager
        model_manager = YOLOModelManager(model_dir=os.path.dirname(model_path) or "yolov12")
        self.model = model_manager.load_model(os.path.basename(model_path), device)
        self.device = device

        # 设置检测类别为人员
        if hasattr(self.model, 'set_classes'):
            self.model.set_classes(["person"])

        self.img_size = img_size
        
        # 用于控制告警频率的变量
        self.last_alarm_time = 0
        self.alarm_interval = 10  # 告警间隔时间（秒）

    def point_in_roi(self, point, roi):
        """
        判断点是否在ROI区域内（射线法）

        Args:
            point: (x, y) 坐标
            roi: ROI区域顶点列表 [(x1, y1), (x2, y2), ...]

        Returns:
            bool: 点是否在ROI内
        """
        x, y = point
        n = len(roi)
        inside = False
        for i in range(n):
            j = (i + 1) % n
            xi, yi = roi[i]
            xj, yj = roi[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
        return inside

    def detect_gather(self, frame, roi, gather_threshold):
        """
        检测人员聚集情况

        Args:
            frame: 视频帧
            roi: ROI区域 [(x1, y1), (x2, y2), ...]
            gather_threshold: 聚集人数阈值

        Returns:
            dict: 检测结果
        """
        logger.info(f"开始聚集检测，ROI: {roi}, 阈值: {gather_threshold}")

        # 检测行人，降低置信度阈值提高检测灵敏度
        results = self.model(frame, classes=[0], conf=0.1, verbose=False)
        logger.info(f"YOLO检测结果: 检测到 {len(results[0].boxes)} 个目标")

        person_boxes = []
        roi_person_boxes = []  # 仅存储ROI区域内的人员框
        for box in results[0].boxes:
            cls = int(box.cls[0])
            if cls == 0:  # 只处理人员类别
                box_coords = box.xyxy.cpu().numpy()[0]
                person_boxes.append(box_coords)
                # 检查该人员是否在ROI区域内
                x1, y1, x2, y2 = box_coords.astype(int)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                if self.point_in_roi(center, roi):
                    roi_person_boxes.append(box_coords)

        logger.info(f"检测到人员数量: {len(person_boxes)}")
        logger.info(f"ROI内人员数量: {len(roi_person_boxes)}")

        # ROI内人数
        roi_person_count = len(roi_person_boxes)

        # 判断是否触发聚集警报（带频率控制）
        current_time = time.time()
        should_trigger_alert = (
            roi_person_count >= gather_threshold and 
            (current_time - self.last_alarm_time) >= self.alarm_interval
        )
        
        alert_triggered = False
        if should_trigger_alert:
            alert_triggered = True
            self.last_alarm_time = current_time
            logger.info(f"触发聚集告警，已更新上次告警时间")

        logger.info(f"聚集检测结果: {roi_person_count} >= {gather_threshold} = {alert_triggered}")

        return {
            'roi_person_count': roi_person_count,
            'alert_triggered': alert_triggered,
            'person_boxes': person_boxes,  # 所有检测到的人员框
            'roi_person_boxes': roi_person_boxes  # 仅ROI区域内的人员框
        }