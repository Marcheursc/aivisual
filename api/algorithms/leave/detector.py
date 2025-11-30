"""
离岗检测算法实现
基于YOLOv12的目标检测算法
"""

import cv2
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO
import torch


class LeaveDetector:
    def __init__(self, model_path="yolov12/yolov12n.pt", device='cuda', img_size=640):
        """
        初始化离岗检测器

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

    def point_in_roi(self, point, roi):
        """
        判断点是否在ROI区域内（射线法）

        Args:
            point: (x, y) 坐标
            roi: ROI区域顶点列表 [(x1, y1), (x2, y2), ...]

        Returns:
            bool: 点是否在ROI内
        """
        # 如果ROI为None，认为所有点都在ROI内（兼容处理）
        if roi is None:
            return True
            
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

    def detect_leave(self, frame, roi, absence_start_time, absence_threshold):
        """
        检测离岗情况

        Args:
            frame: 视频帧
            roi: ROI区域 [(x1, y1), (x2, y2), ...]
            absence_start_time: 开始脱岗时间
            absence_threshold: 脱岗判定阈值（秒）

        Returns:
            dict: 检测结果
        """
        # 检测行人
        results = self.model(frame, classes=[0], verbose=False)
        person_boxes = []
        for box in results[0].boxes:
            cls = int(box.cls[0])
            if cls == 0:  # 只处理人员类别
                person_boxes.append(box.xyxy.cpu().numpy()[0])

        # 统计ROI内人数
        roi_person_count = 0
        for box in person_boxes:
            x1, y1, x2, y2 = box.astype(int)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            if self.point_in_roi(center, roi):
                roi_person_count += 1

        # 更新状态检测逻辑
        current_time = datetime.now()
        status = "脱岗" if roi_person_count == 0 else "在岗"

        if roi_person_count > 0:
            # 有人在岗，重置脱岗时间
            absence_start_time = None
            alert_triggered = False
        else:
            # 无人在岗
            if absence_start_time is None:
                absence_start_time = current_time

            absence_duration = (current_time - absence_start_time).total_seconds()

            if absence_duration >= absence_threshold:
                alert_triggered = True
            else:
                alert_triggered = False

        return {
            'status': status,
            'roi_person_count': roi_person_count,
            'absence_start_time': absence_start_time,
            'alert_triggered': alert_triggered,
            'person_boxes': person_boxes
        }
