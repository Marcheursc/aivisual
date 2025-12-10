"""
聚集检测视频处理模块
处理聚集检测的视频处理逻辑
"""

from typing import Optional, List, Tuple
from ..video_processing.core import VideoProcessorCore
from ..video_processing.utils import draw_detection_box, put_text
from .detector import GatherDetector
import cv2
import numpy as np


def process_gather_video(
        model_path: str,
        video_path: str,
        output_path: str,
        roi: Optional[List[Tuple[int, int]]] = None,
        gather_threshold: int = 5,
        loitering_time_threshold: int = 3,
        tracking_disappear_threshold: int = 10,
        device: str = 'cuda'
) -> str:
    """
    处理聚集检测视频

    Args:
        model_path: 模型路径
        video_path: 输入视频路径
        output_path: 输出视频路径
        roi: ROI区域 [(x1, y1), (x2, y2), ...]
        gather_threshold: 聚集人数阈值
        loitering_time_threshold: 停留时间阈值（秒）
        tracking_disappear_threshold: 跟踪消失阈值（帧数）
        device: 运行设备

    Returns:
        str: 处理后的视频路径
    """
    # 默认ROI区域
    if roi is None:
        roi = [(220, 300), (700, 300), (700, 700), (200, 700)]

    # 初始化检测器
    detector = GatherDetector(model_path=model_path, device=device)

    # 初始化核心处理器
    core = VideoProcessorCore(model_path)

    # 打开视频文件
    cap = core.open_video_capture(video_path)

    # 获取视频参数
    fps, width, height = core.get_video_properties(cap)

    # 初始化视频写入器
    out = core.create_video_writer(output_path, fps, width, height)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_time = frame_count / fps

        # 直接在原始帧上执行聚集检测，不进行缩放
        result = detector.detect_gather(frame, roi, gather_threshold, loitering_time_threshold, tracking_disappear_threshold, fps)

        # 在帧上绘制检测结果（包含所有状态的颜色框）
        annotated_frame = draw_gather_detections(
            frame, roi, result['roi_person_count'], gather_threshold, result['alert_triggered'],
            result['qualified_persons'], loitering_time_threshold,
            current_boxes=result['current_boxes'], person_tracker=result['person_tracker']
        )

        # 写入处理后的帧
        out.write(annotated_frame)

    # 释放资源
    core.release_resources(cap, out)

    return output_path


def draw_gather_detections(frame, roi, roi_person_count, gather_threshold, alert_triggered, qualified_persons=None, loitering_time_threshold=3, current_boxes=None, person_tracker=None):
    """
    在帧上绘制聚集检测结果
    """
    # 绘制ROI区域
    if len(roi) >= 3:
        pts = np.array(roi, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

    # 如果有当前检测框和跟踪器数据，使用多色框逻辑
    if current_boxes is not None and person_tracker is not None and len(current_boxes) > 0:
        for i, box in enumerate(current_boxes):
            x1, y1, x2, y2 = box.astype(int)
            
            # 获取跟踪器中的数据
            person_id = i + 1
            if person_id in person_tracker:
                tracker_data = person_tracker[person_id]
                in_roi = tracker_data.get('in_roi', False)
                total_roi_time = tracker_data.get('total_roi_time', 0)
                
                # 根据状态选择颜色
                if in_roi and total_roi_time >= loitering_time_threshold:
                    color = (0, 0, 255)  # 红色 - 停留时间达标
                elif in_roi:
                    color = (0, 255, 255)  # 黄色 - 在ROI内但未达标
                else:
                    color = (0, 255, 0)  # 绿色 - 在ROI外
                
                # 绘制边框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 显示停留时间
                if in_roi:
                    text = f"{total_roi_time:.1f}s"
                    cv2.putText(frame, text, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    else:
        # 原有逻辑：只绘制qualified_persons的红色框
        for person in qualified_persons:
            x1, y1, x2, y2 = person['box'].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # 显示停留时间
            if 'total_roi_time' in person:
                text = f"{person['total_roi_time']:.1f}s"
                cv2.putText(frame, text, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # 绘制符合条件的人员框和停留时间（兼容原有逻辑）
    if qualified_persons:
        for person in qualified_persons:
            box = person['box']
            total_roi_time = person['total_roi_time']
            
            # 绘制边界框（红色表示已计入聚集人数）
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # 显示停留时间
            put_text(frame, f"{total_roi_time:.1f}s", (x1, y1-10), 
                    color=(0, 255, 255), thickness=1, font_scale=0.6)

    # 显示统计信息
    put_text(frame, f"ROI内停留人数: {roi_person_count}", (30, 50), 
             color=(255, 0, 0), thickness=2, font_scale=1.0)
    put_text(frame, f"停留阈值: {loitering_time_threshold}秒", (30, 90), 
             color=(0, 255, 255), thickness=1, font_scale=0.8)

    # 聚集预警（超过阈值时）
    if alert_triggered:
        put_text(frame, "警告：人员聚集！", (30, 130), 
                color=(0, 0, 255), thickness=3, font_scale=1.2)

    return frame
