"""
视频处理核心模块
统一处理各种视频分析任务
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime
from typing import Optional, List, Tuple
from .loitering_detection import LoiteringDetector
from .leave_detection import LeaveDetector
from .gather_detection import GatherDetector


class VideoProcessor:
    """
    视频处理器，统一处理各种检测任务
    """

    def __init__(self, model_path: str = "yolov12/yolov12n.pt"):
        """
        初始化视频处理器

        Args:
            model_path: 模型路径
        """
        self.model_path = model_path

    def _get_loitering_detector(self, loitering_time_threshold: int = 20):
        """
        获取徘徊检测器实例

        Args:
            loitering_time_threshold: 徘徊时间阈值（秒）

        Returns:
            LoiteringDetector: 徘徊检测器实例
        """
        return LoiteringDetector(
            model_path=self.model_path,
            loitering_time_threshold=loitering_time_threshold
        )

    def _get_leave_detector(self):
        """
        获取离岗检测器实例

        Returns:
            LeaveDetector: 离岗检测器实例
        """
        return LeaveDetector(model_path=self.model_path)

    def _get_gather_detector(self):
        """
        获取聚集检测器实例

        Returns:
            GatherDetector: 聚集检测器实例
        """
        return GatherDetector(model_path=self.model_path)

    def process_loitering_video(self,
                               video_path: str,
                               output_path: str,
                               loitering_time_threshold: int = 20,
                               device: str = 'cuda') -> str:
        """
        处理徘徊检测视频

        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
            loitering_time_threshold: 徘徊时间阈值（秒）
            device: 运行设备

        Returns:
            str: 处理后的视频路径
        """
        # 初始化检测器
        detector = LoiteringDetector(
            model_path=self.model_path,
            loitering_time_threshold=loitering_time_threshold,
            device=device
        )

        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"无法打开视频文件: {video_path}")

        # 获取视频参数
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_time = frame_count / fps

            # 执行徘徊检测
            detections, alarms = detector.detect_loitering(frame, frame_time)

            # 在帧上绘制检测结果
            annotated_frame = self._draw_loitering_detections(frame, detections, alarms)

            # 写入处理后的帧
            out.write(annotated_frame)

        # 释放资源
        cap.release()
        out.release()

        return output_path

    def process_leave_video(self,
                           video_path: str,
                           output_path: str,
                           roi: Optional[List[Tuple[int, int]]] = None,
                           absence_threshold: int = 5,
                           device: str = 'cuda') -> str:
        """
        处理离岗检测视频

        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
            roi: ROI区域 [(x1, y1), (x2, y2), ...]
            absence_threshold: 脱岗判定阈值（秒）
            device: 运行设备

        Returns:
            str: 处理后的视频路径
        """
        # 默认ROI区域
        if roi is None:
            roi = [(600, 100), (1000, 100), (1000, 700), (600, 700)]

        # 初始化检测器
        detector = LeaveDetector(model_path=self.model_path, device=device)

        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"无法打开视频文件: {video_path}")

        # 获取视频参数
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 状态变量
        absence_start_time = None
        alert_triggered = False

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_time = frame_count / fps

            # 调整图像尺寸以提高处理速度
            h, w = frame.shape[:2]
            scale = detector.img_size / max(h, w) if hasattr(detector, 'img_size') else 1
            if scale < 1:
                new_w, new_h = int(w * scale), int(h * scale)
                processed_frame = cv2.resize(frame, (new_w, new_h))
            else:
                processed_frame = frame
                scale = 1

            # 执行离岗检测
            result = detector.detect_leave(processed_frame, roi, absence_start_time, absence_threshold)
            absence_start_time = result['absence_start_time']

            # 在帧上绘制检测结果
            annotated_frame = self._draw_leave_detections(
                frame, roi, result['status'], result['roi_person_count'],
                absence_start_time, absence_threshold, result['alert_triggered']
            )

            # 绘制检测到的人员框
            for box in result['person_boxes']:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 写入处理后的帧
            out.write(annotated_frame)

        # 释放资源
        cap.release()
        out.release()

        return output_path

    def process_gather_video(self,
                            video_path: str,
                            output_path: str,
                            roi: Optional[List[Tuple[int, int]]] = None,
                            gather_threshold: int = 5,
                            device: str = 'cuda') -> str:
        """
        处理聚集检测视频

        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
            roi: ROI区域 [(x1, y1), (x2, y2), ...]
            gather_threshold: 聚集人数阈值
            device: 运行设备

        Returns:
            str: 处理后的视频路径
        """
        # 默认ROI区域
        if roi is None:
            roi = [(220, 300), (700, 300), (700, 700), (200, 700)]

        # 初始化检测器
        detector = GatherDetector(model_path=self.model_path, device=device)

        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"无法打开视频文件: {video_path}")

        # 获取视频参数
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_time = frame_count / fps

            # 调整图像尺寸以提高处理速度
            h, w = frame.shape[:2]
            scale = detector.img_size / max(h, w) if hasattr(detector, 'img_size') else 1
            if scale < 1:
                new_w, new_h = int(w * scale), int(h * scale)
                processed_frame = cv2.resize(frame, (new_w, new_h))
            else:
                processed_frame = frame
                scale = 1

            # 执行聚集检测
            result = detector.detect_gather(processed_frame, roi, gather_threshold)

            # 在帧上绘制检测结果
            annotated_frame = self._draw_gather_detections(
                frame, roi, result['roi_person_count'], gather_threshold, result['alert_triggered']
            )

            # 绘制检测到的人员框
            for box in result['person_boxes']:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # 写入处理后的帧
            out.write(annotated_frame)

        # 释放资源
        cap.release()
        out.release()

        return output_path

    def _draw_loitering_detections(self, frame, detections, alarms):
        """
        在帧上绘制徘徊检测结果
        """
        # 绘制检测框
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection[:4])
            class_name = detection[5]
            confidence = detection[4]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name}: {confidence:.2f}",
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 绘制警报
        for obj_id, alarm in alarms.items():
            x1, y1, x2, y2 = map(int, alarm['position'])
            duration = alarm['duration']
            cv2.putText(frame, f"Loitering: {duration:.1f}s",
                       (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame

    def _draw_leave_detections(self, frame, roi, status, person_count,
                              absence_start_time, absence_threshold, alert_triggered):
        """
        在帧上绘制离岗检测结果
        """
        # 绘制ROI区域
        if roi is not None:
            roi_np = np.array(roi, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [roi_np], True, (0, 255, 0), 2)

        # 绘制状态信息
        color = (0, 255, 0) if status == "在岗" else (0, 0, 255)
        cv2.putText(frame, f"状态: {status}", (30, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # 绘制脱岗时长
        if absence_start_time is not None:
            absence_duration = (datetime.now() - absence_start_time).total_seconds()
            cv2.putText(frame, f"脱岗时长: {absence_duration:.1f}秒", (30, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 绘制警报
        if alert_triggered:
            cv2.putText(frame, "⚠️ 警告：人员脱岗！", (frame.shape[1] // 2 - 150, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        return frame

    def _draw_gather_detections(self, frame, roi, person_count, gather_threshold, alert_triggered):
        """
        在帧上绘制聚集检测结果
        """
        # 绘制ROI区域
        if roi is not None:
            roi_np = np.array(roi, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [roi_np], True, (0, 255, 0), 2)

        # 绘制人数信息
        cv2.putText(frame, f"ROI内人数: {person_count}", (30, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # 绘制警报
        if alert_triggered:
            cv2.putText(frame, "警告：人员聚集！", (30, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        return frame
