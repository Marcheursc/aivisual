"""
聚集检测算法实现
基于YOLOv12的目标检测算法，包含完整的人员跟踪和停留时间计算
"""

import cv2
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO
import torch
import logging
import time
from collections import defaultdict

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
        
        # 初始化行人跟踪相关变量
        self.person_tracker = {}  # 跟踪字典 {id: {enter_time, total_roi_time, last_seen_frame, in_roi, last_box}}
        self.next_id = 1  # 下一个行人ID
        self.max_distance = 50  # 跟踪匹配的最大距离（像素）
        self.frame_count = 0  # 帧计数器
        self.fps = 30.0  # 默认帧率，会在detect_gather中更新

    def center_distance(self, box1, box2):
        """
        计算两个边界框中心的欧氏距离
        """
        x1, y1, x2, y2 = box1
        center1 = ((x1 + x2) // 2, (y1 + y2) // 2)
        x3, y3, x4, y4 = box2
        center2 = ((x3 + x4) // 2, (y3 + y4) // 2)
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

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

    def detect_gather(self, frame, roi, gather_threshold, loitering_time_threshold=3, tracking_disappear_threshold=10, fps=30.0):
        """
        检测人员聚集情况（包含完整的跟踪和停留时间计算）

        Args:
            frame: 视频帧
            roi: ROI区域 [(x1, y1), (x2, y2), ...]
            gather_threshold: 聚集人数阈值
            loitering_time_threshold: 停留时间阈值（秒）
            tracking_disappear_threshold: 跟踪消失阈值（帧数）
            fps: 视频帧率

        Returns:
            dict: 检测结果，包含详细的跟踪信息
        """
        self.frame_count += 1
        self.fps = fps
        frame_interval = 1.0 / fps
        current_frame = self.frame_count

        logger.info(f"开始聚集检测，ROI: {roi}, 阈值: {gather_threshold}, 停留阈值: {loitering_time_threshold}s")

        # 检测行人，降低置信度阈值提高检测灵敏度
        results = self.model(frame, classes=[0], conf=0.1, verbose=False)
        logger.info(f"YOLO检测结果: 检测到 {len(results[0].boxes)} 个目标")

        # 提取检测结果中的边界框（双重过滤：只保留类别0）
        current_boxes = []
        for box in results[0].boxes:
            cls = int(box.cls[0])  # 获取类别ID
            if cls == 0:  # 仅保留行人
                current_boxes.append(box.xyxy.cpu().numpy()[0])  # 提取边界框（x1,y1,x2,y2）

        logger.info(f"检测到人员数量: {len(current_boxes)}")

        # 更新跟踪器
        new_tracker = {}
        used_ids = set()
        
        # 匹配现有跟踪与当前检测
        for box in current_boxes:
            x1, y1, x2, y2 = box.astype(int)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            in_roi = self.point_in_roi(center, roi)
            
            # 寻找最匹配的现有跟踪目标
            best_match_id = None
            min_dist = self.max_distance
            
            for pid, data in self.person_tracker.items():
                if pid not in used_ids:
                    dist = self.center_distance(box, data.get('last_box', []))
                    if dist < min_dist:
                        min_dist = dist
                        best_match_id = pid
            
            if best_match_id is not None:
                # 更新现有跟踪
                used_ids.add(best_match_id)
                data = self.person_tracker[best_match_id]
                
                # 更新停留时间
                if in_roi:
                    if not data.get('in_roi', False):
                        # 刚进入ROI，记录进入时间
                        data['enter_time'] = current_frame * frame_interval
                        data['in_roi'] = True
                    else:
                        # 已在ROI内，累计停留时间
                        data['total_roi_time'] = (current_frame * frame_interval) - data['enter_time']
                else:
                    data['in_roi'] = False
                    
                data['last_seen_frame'] = current_frame
                data['last_box'] = box
                new_tracker[best_match_id] = data
            else:
                # 新目标
                new_data = {
                    'enter_time': current_frame * frame_interval if in_roi else None,
                    'total_roi_time': 0.0 if in_roi else 0.0,
                    'last_seen_frame': current_frame,
                    'in_roi': in_roi,
                    'last_box': box
                }
                new_tracker[self.next_id] = new_data
                self.next_id += 1
        
        # 保留未匹配但未超过消失阈值的目标
        for pid, data in self.person_tracker.items():
            if pid not in used_ids and current_frame - data['last_seen_frame'] <= tracking_disappear_threshold:
                new_tracker[pid] = data
        
        self.person_tracker = new_tracker
        
        # 统计符合条件的停留人数（在ROI内且停留时间超过阈值）
        roi_person_count = 0
        qualified_persons = []  # 存储符合条件的人员信息
        for pid, data in self.person_tracker.items():
            if data['in_roi'] and data['total_roi_time'] >= loitering_time_threshold:
                roi_person_count += 1
                qualified_persons.append({
                    'id': pid,
                    'box': data['last_box'],
                    'total_roi_time': data['total_roi_time']
                })

        logger.info(f"ROI内符合条件的人员数量: {roi_person_count}")

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
            'current_boxes': current_boxes,  # 所有检测到的人员框
            'qualified_persons': qualified_persons,  # 符合条件的人员信息（包含ID、框、停留时间）
            'person_tracker': self.person_tracker.copy(),  # 跟踪器状态
            'frame_count': current_frame,
            'fps': fps
        }