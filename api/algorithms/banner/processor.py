"""
横幅检测视频处理模块
处理横幅检测的视频处理逻辑
"""

from typing import Optional, List, Tuple
from ..video_processing.core import VideoProcessorCore
from ..video_processing.utils import draw_detection_box, put_text
from .detector import BannerDetector
import cv2


def process_banner_video(
        model_path: str,
        video_path: str,
        output_path: str,
        roi: Optional[List[Tuple[int, int]]] = None,
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.45,
        device: str = 'cuda'
) -> str:
    """
    处理横幅检测视频

    Args:
        model_path: 模型路径
        video_path: 输入视频路径
        output_path: 输出视频路径
        roi: ROI区域 [(x1, y1), (x2, y2), ...]
        conf_threshold: 置信度阈值
        iou_threshold: NMS IoU阈值
        device: 运行设备

    Returns:
        str: 处理后的视频路径
    """
    print(f"开始横幅检测处理: {video_path}")
    print(f"输出路径: {output_path}")

    # 初始化检测器
    print("初始化BannerDetector...")
    detector = BannerDetector(
        model_path=None,  # 横幅检测使用专用的banner_weight.pt模型
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        device=device
    )
    print("BannerDetector初始化完成")

    # 初始化核心处理器
    core = VideoProcessorCore(model_path)

    # 打开视频文件
    cap = core.open_video_capture(video_path)

    # 获取视频参数
    fps, width, height = core.get_video_properties(cap)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")

    # 初始化视频写入器
    out = core.create_video_writer(output_path, fps, width, height)

    frame_count = 0
    total_banners = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 == 0:  # 每30帧输出一次进度
            print(f"处理进度: {frame_count}/{total_frames} 帧")

        # 执行横幅检测
        results, banners = detector.detect_banner(frame)

        # 调试输出
        if banners:
            print(f"第{frame_count}帧检测到 {len(banners)} 个横幅")
            for i, banner in enumerate(banners):
                print(f"  横幅{i + 1}: 类别={banner['class']}, 置信度={banner['confidence']:.2f}, "
                      f"位置={banner['box']}, 宽高比={banner.get('aspect_ratio', 0):.2f}")
            total_banners += len(banners)

        # 在帧上绘制检测结果
        annotated_frame = draw_banner_detections(frame, banners)

        # 写入处理后的帧
        out.write(annotated_frame)

    # 释放资源
    core.release_resources(cap, out)

    print(f"横幅检测处理完成!")
    print(f"总帧数: {frame_count}, 检测到横幅的总次数: {total_banners}")

    return output_path


def draw_banner_detections(frame, banners):
    """
    在帧上绘制横幅检测结果
    """
    # 绘制检测到的横幅边界框
    for banner in banners:
        x1, y1, x2, y2 = map(int, banner['box'])
        confidence = banner['confidence']
        class_name = banner['class']
        
        frame = draw_detection_box(frame, [x1, y1, x2, y2], (0, 0, 255), 2)
        frame = put_text(frame, f'{class_name}: {confidence:.2f}', (x1, y1 - 10), 0.5, (0, 0, 255), 2)

    return frame
