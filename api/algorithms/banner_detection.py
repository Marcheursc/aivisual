"""
横幅检测算法实现
基于YOLOv12的目标检测算法
完全按照bannerdetect.py的实现方式
"""

import cv2
import numpy as np
import sys
import torch
import os
from ultralytics import YOLO

class BannerDetector:
    def __init__(self, model_path=None, conf_threshold=0.3, iou_threshold=0.45, img_size=640, device='cuda'):
        """
        初始化横幅检测器

        Args:
            model_path (str): YOLOv12模型路径，如果为None则使用best.pt
            conf_threshold (float): 置信度阈值
            iou_threshold (float): NMS IoU阈值
            img_size (int): 图像处理尺寸
            device (str): 运行设备 ('cuda' 或 'cpu')
        """
        # 如果没有提供模型路径，使用横幅检测专用的best.pt模型
        if model_path is None:
            model_path = "yolov12/best.pt"
        
        print(f"[BannerDetector] 开始初始化，模型路径: {model_path}")
        try:
            # 检查设备可用性
            if device == 'cuda' and not torch.cuda.is_available():
                print("[BannerDetector] CUDA不可用，回退到CPU")
                device = 'cpu'

            print(f"[BannerDetector] 使用设备: {device}")

            # 加载YOLOv12模型
            print(f"[BannerDetector] 正在加载YOLO模型...")
            self.model = YOLO(model_path)
            self.device = device

            # 将模型移动到指定设备
            self.model.to(device)

            # 打印模型信息
            print(f"[BannerDetector] 模型加载成功!")
            print(f"[BannerDetector] 可用类别总数: {len(self.model.names)}")
            print(f"[BannerDetector] 所有类别: {list(self.model.names.values())}")
            
        except Exception as e:
            print(f"[BannerDetector] 模型加载错误: {e}")
            print("[BannerDetector] 请确保已下载YOLOv12模型文件")
            sys.exit(1)

        # 配置参数（与bannerdetect.py保持一致）
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        
        # 绘制参数（与bannerdetect.py保持一致）
        self.SHOW_LABEL = True  # 是否显示检测标签
        self.SHOW_CONF = True  # 是否显示置信度
        self.BOX_COLOR = (0, 255, 0)  # 检测框颜色（BGR：绿色）
        self.TEXT_COLOR = (255, 255, 255)  # 文本颜色（白色）
        self.LINE_WIDTH = 2  # 检测框线宽
        self.FONT_SCALE = 0.6  # 字体大小

        # 存储检测到的横幅信息
        self.detected_banners = []
        print(f"[BannerDetector] 初始化完成")

    def detect_banner(self, frame):
        """
        检测视频帧中的横幅（完全按照bannerdetect.py的实现）

        Args:
            frame: 视频帧

        Returns:
            results: 检测结果
            banners: 横幅信息
        """
        # 使用YOLOv12检测目标（完全按照bannerdetect.py的方式）
        results = self.model(
            frame,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False  # 关闭推理日志输出
        )

        # 解析检测结果（完全按照bannerdetect.py的方式）
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
                    
                    # 存储检测信息（与bannerdetect.py完全一致）
                    banners.append({
                        'box': (x1, y1, x2, y2),
                        'confidence': conf,
                        'class': cls_name
                    })
        
        # 更新检测到的信息
        self.detected_banners = banners

        return results, banners

    def draw_detections(self, frame, banners):
        """
        在帧上绘制检测结果（完全按照bannerdetect.py的方式）

        Args:
            frame: 视频帧
            banners: 检测到的信息

        Returns:
            frame_with_detections: 绘制了检测结果的视频帧
        """
        # 解析检测结果并绘制（完全按照bannerdetect.py的方式）
        for banner in banners:
            x1, y1, x2, y2 = banner['box']
            conf = banner['confidence']
            cls_name = banner['class']
            
            # 绘制检测框
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.BOX_COLOR, self.LINE_WIDTH)
            
            # 绘制标签和置信度（完全按照bannerdetect.py的方式）
            if self.SHOW_LABEL or self.SHOW_CONF:
                label_text = ""
                if self.SHOW_LABEL:
                    label_text += cls_name
                if self.SHOW_CONF:
                    label_text += f" ({conf:.2f})" if label_text else f"{conf:.2f}"
                
                # 绘制文本背景（提高可读性）
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, 1)[0]
                text_bg_x2 = x1 + text_size[0] + 6
                text_bg_y2 = y1 - text_size[1] - 6
                cv2.rectangle(frame, (x1, y1), (text_bg_x2, text_bg_y2), self.BOX_COLOR, -1)
                
                # 绘制文本
                cv2.putText(
                    frame, label_text, (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.TEXT_COLOR, 1
                )
        
        # 添加帧信息（左上角，完全按照bannerdetect.py的方式）
        frame_info = f"YOLOv12 Banner Detection | Conf: {self.conf_threshold}"
        cv2.putText(frame, frame_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame