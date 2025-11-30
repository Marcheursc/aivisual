"""
徘徊检测视频处理模块
处理徘徊检测的视频处理逻辑
"""

from typing import Optional, List, Tuple
from ..video_processing.core import VideoProcessorCore
from ..video_processing.utils import draw_detection_box, put_text
from .detector import LoiteringDetector


def process_loitering_video(
        model_name: str,
        video_path: str,
        output_path: str,
        loitering_time_threshold: int = 20,
        device: str = 'cuda'
) -> str:
    """
    处理徘徊检测视频

    Args:
        model_name: 模型文件名
        video_path: 输入视频路径
        output_path: 输出视频路径
        loitering_time_threshold: 徘徊时间阈值（秒）
        device: 运行设备

    Returns:
        str: 处理后的视频路径
    """
    # 初始化检测器
    detector = LoiteringDetector(
        model_name=model_name,
        loitering_time_threshold=loitering_time_threshold,
        device=device
    )

    # 初始化核心处理器
    core = VideoProcessorCore(model_name)

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

        # 执行徘徊检测
        detections, alarms = detector.detect_loitering(frame, frame_time)

        # 在帧上绘制检测结果
        annotated_frame = draw_loitering_detections(frame, detections, alarms)

        # 写入处理后的帧
        out.write(annotated_frame)

    # 释放资源
    core.release_resources(cap, out)

    return output_path


def draw_loitering_detections(frame, detections, alarms):
    """
    在帧上绘制徘徊检测结果
    """
    # 创建一个集合来存储触发警报的对象ID
    alarm_object_ids = set(alarms.keys())

    # 绘制检测框
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection[:4])
        class_name = detection[5]
        confidence = detection[4]

        # 检查是否有对象ID（在检测结果的第7个位置）
        object_id = detection[6] if len(detection) > 6 else None

        # 如果该对象触发了警报，则使用红色边框，否则使用绿色
        color = (0, 0, 255) if object_id in alarm_object_ids else (0, 255, 0)

        frame = draw_detection_box(frame, [x1, y1, x2, y2], color, 2)
        frame = put_text(frame, f"{class_name}: {confidence:.2f}", (x1, y1 - 10), 0.5, color, 2)

    # 绘制警报
    for obj_id, alarm in alarms.items():
        x1, y1, x2, y2 = map(int, alarm['position'])
        duration = alarm['duration']
        frame = put_text(frame, f"Loitering: {duration:.1f}s", (x1, y2 + 20), 0.5, (0, 0, 255), 2)

    return frame
