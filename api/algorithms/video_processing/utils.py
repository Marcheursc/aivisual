"""
视频处理工具模块
提供通用的视频处理工具函数
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import os
from PIL import Image, ImageDraw, ImageFont


def draw_roi(frame: np.ndarray, roi: Optional[List[Tuple[int, int]]]) -> np.ndarray:
    """
    在帧上绘制ROI区域

    Args:
        frame: 输入帧
        roi: ROI区域坐标点列表

    Returns:
        np.ndarray: 绘制后的帧
    """
    if roi is not None:
        roi_np = np.array(roi, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [roi_np], True, (0, 255, 0), 2)
    return frame


def draw_detection_box(frame: np.ndarray, box: List[int], color: Tuple[int, int, int] = (0, 255, 0),
                       thickness: int = 2) -> np.ndarray:
    """
    在帧上绘制检测框

    Args:
        frame: 输入帧
        box: 检测框坐标 [x1, y1, x2, y2]
        color: 框的颜色 (B, G, R)
        thickness: 框的线宽

    Returns:
        np.ndarray: 绘制后的帧
    """
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame


def put_text(frame: np.ndarray, text: str, position: Tuple[int, int],
             font_scale: float = 0.6, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 1) -> np.ndarray:
    """
    在帧上添加文本（支持中英文）

    Args:
        frame: 输入帧
        text: 要添加的文本
        position: 文本位置 (x, y)
        font_scale: 字体大小
        color: 文本颜色 (B, G, R)
        thickness: 文本线宽

    Returns:
        np.ndarray: 添加文本后的帧
    """
    x, y = position
    
    # 检查是否包含中文字符
    if any('\u4e00' <= char <= '\u9fff' for char in text):
        # 使用PIL绘制中文
        return put_text_chinese(frame, text, position, font_scale, color)
    else:
        # 使用OpenCV绘制英文
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return frame


def put_text_chinese(frame: np.ndarray, text: str, position: Tuple[int, int],
                    font_scale: float = 0.6, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    在帧上添加中文文本（使用PIL）

    Args:
        frame: 输入帧
        text: 要添加的中文文本
        position: 文本位置 (x, y)
        font_scale: 字体大小缩放
        color: 文本颜色 (B, G, R)

    Returns:
        np.ndarray: 添加文本后的帧
    """
    x, y = position
    
    # 转换为PIL图像
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 尝试使用系统字体
    font_size = int(font_scale * 20)
    try:
        # Windows系统字体路径
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
            "C:/Windows/Fonts/msyh.ttc",   # 微软雅黑
        ]
        
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break
        
        if font is None:
            # 使用默认字体
            font = ImageFont.load_default()
            
    except Exception:
        font = ImageFont.load_default()
    
    # 绘制文本
    draw.text((x, y - font_size), text, font=font, fill=color)
    
    # 转换回OpenCV格式
    frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return frame


def draw_text_background(frame: np.ndarray, text: str, position: Tuple[int, int],
                         font_scale: float = 0.6, text_color: Tuple[int, int, int] = (255, 255, 255),
                         bg_color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    在帧上绘制带背景的文本

    Args:
        frame: 输入帧
        text: 要添加的文本
        position: 文本位置 (x, y)
        font_scale: 字体大小
        text_color: 文本颜色 (B, G, R)
        bg_color: 背景颜色 (B, G, R)

    Returns:
        np.ndarray: 添加文本后的帧
    """
    x, y = position
    # 计算文本大小
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
    text_bg_x2 = x + text_size[0] + 6
    text_bg_y2 = y - text_size[1] - 6

    # 绘制文本背景
    cv2.rectangle(frame, (x, y), (text_bg_x2, text_bg_y2), bg_color, -1)

    # 绘制文本
    cv2.putText(frame, text, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

    return frame
