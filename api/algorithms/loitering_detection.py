"""
徘徊检测算法实现
基于YOLOv12和ByteTrack的目标检测与跟踪算法
"""

import cv2
import numpy as np
import sys
import torch
import os

# 确保优先使用项目中的yolov12模块而不是site-packages中的
project_yolov12_path = os.path.join(os.path.dirname(__file__), '..', '..', 'yolov12')
if project_yolov12_path not in sys.path:
    sys.path.insert(0, project_yolov12_path)

try:
    # 优先使用YOLOv12的ultralytics
    from ultralytics import YOLO
    print("Successfully imported Ultralytics library from YOLOv12")
except ImportError as e:
    try:
        # 如果失败，尝试使用系统安装的ultralytics
        import sys
        # 移除yolov12路径以避免冲突
        yolov12_path = os.path.join(os.path.dirname(__file__), '..', '..', 'yolov12')
        if yolov12_path in sys.path:
            sys.path.remove(yolov12_path)
        from ultralytics import YOLO
        print("Successfully imported Ultralytics library from system")
    except ImportError as e:
        print("Error importing Ultralytics library:", e)
        print("Please install required dependencies with: pip install ultralytics")
        sys.exit(1)

# 尝试导入ByteTrack
BYTETRACK_AVAILABLE = False
try:
    # 从yolov12的ultralytics导入
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'yolov12'))
    from ultralytics.trackers.byte_tracker import BYTETracker
    BYTETRACK_AVAILABLE = True
    print("Successfully imported ByteTrack tracker from ultralytics.trackers")
except ImportError as e:
    print("ByteTrack not available:", str(e))
    print("Using basic tracking")

# 定义Args类用于ByteTrack参数配置
class Args:
    def __init__(self, track_high_thresh=0.5, track_low_thresh=0.1, new_track_thresh=0.6,
                 track_buffer=30, match_thresh=0.8, fuse_score=True):
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.fuse_score = fuse_score


class LoiteringDetector:
    def __init__(self, model_path="yolov12/yolov12n.pt", loitering_time_threshold=20, target_classes=["person"],
                 conf_threshold=0.5, img_size=640, device='cuda', detection_region=None, use_bytetrack=True):
        """
        初始化徘徊检测器

        Args:
            model_path (str): YOLOv12模型路径
            loitering_time_threshold (int): 徘徊时间阈值（秒）
            target_classes (list): 要检测的目标类别列表
            conf_threshold (float): 置信度阈值
            img_size (int): 图像处理尺寸（较小的尺寸可以提高速度）
            device (str): 运行设备 ('cuda' 或 'cpu')
            detection_region (tuple): 检测区域 (x, y, width, height) 或 None 表示全图检测
            use_bytetrack (bool): 是否使用ByteTrack跟踪器
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

            # 设置检测类别
            self.target_classes = target_classes
            if hasattr(self.model, 'set_classes'):
                self.model.set_classes(target_classes)
            print(f"Model loaded successfully! Detecting classes: {target_classes}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please ensure you have downloaded the yolov12 model file")
            sys.exit(1)

        # 配置参数
        self.conf_threshold = conf_threshold
        self.img_size = img_size  # 降低图像尺寸以提高速度

        # 徘徊时间阈值（秒）
        self.loitering_time_threshold = loitering_time_threshold

        # 存储跟踪对象的信息
        self.tracked_objects = {}
        self.loitering_alarms = {}

        # 跟踪ID计数器
        self.next_object_id = 0

        # 检测区域
        self.detection_region = detection_region

        # 不同类别的颜色
        self.class_colors = {
            "person": (0, 255, 0),      # 绿色
            "car": (255, 0, 0),         # 蓝色
            "truck": (0, 0, 255),       # 红色
            "bus": (255, 255, 0),       # 青色
            "motorcycle": (255, 0, 255), # 紫色
            "bicycle": (0, 255, 255),   # 黄色
        }

        # 为未定义颜色的类别生成随机颜色
        for cls in target_classes:
            if cls not in self.class_colors:
                self.class_colors[cls] = tuple(np.random.randint(0, 255, 3).tolist())

        # ByteTrack跟踪器
        self.use_bytetrack = use_bytetrack and BYTETRACK_AVAILABLE
        if self.use_bytetrack:
            try:
                # 使用兼容的参数初始化BYTETracker
                args = Args(track_high_thresh=0.5, track_low_thresh=0.1, new_track_thresh=0.6,
                           track_buffer=30, match_thresh=0.8, fuse_score=True)
                self.tracker = BYTETracker(args)
                self.frame_id = 0
                print("ByteTrack tracker initialized successfully")
            except Exception as e:
                print(f"Error initializing BYTETracker: {e}")
                self.use_bytetrack = False
                print("Falling back to basic tracking")

    def calculate_iou(self, box1, box2):
        """
        计算两个边界框的交并比(IoU)

        Args:
            box1, box2: 边界框坐标 [x1, y1, x2, y2]

        Returns:
            iou: 交并比
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - intersection_area

        iou = intersection_area / union_area if union_area > 0 else 0
        return iou

    def is_in_detection_region(self, box):
        """
        检查边界框是否在检测区域内

        Args:
            box: 边界框坐标 [x1, y1, x2, y2]

        Returns:
            bool: 是否在检测区域内
        """
        if self.detection_region is None:
            return True

        rx, ry, rw, rh = self.detection_region
        bx1, by1, bx2, by2 = box

        # 检查边界框中心点是否在检测区域内
        center_x = (bx1 + bx2) / 2
        center_y = (by1 + by2) / 2

        return (rx <= center_x <= rx + rw) and (ry <= center_y <= ry + rh)

    def assign_object_id(self, detected_box, class_name, iou_threshold=0.5):
        """
        为检测到的对象分配ID

        Args:
            detected_box: 检测到的边界框 [x1, y1, x2, y2]
            class_name: 对象类别名称
            iou_threshold: IoU阈值用于匹配

        Returns:
            object_id: 分配的对象ID
        """
        # 只与同类对象进行匹配
        if not self.tracked_objects:
            new_id = self.next_object_id
            self.next_object_id += 1
            return new_id

        best_match_id = None
        best_iou = 0

        # 查找最佳匹配的对象（仅同类）
        for obj_id, obj_info in self.tracked_objects.items():
            # 检查是否为同类
            if obj_info.get('class') == class_name:
                iou = self.calculate_iou(detected_box, obj_info['last_position'])
                if iou > best_iou and iou > iou_threshold:
                    best_iou = iou
                    best_match_id = obj_id

        if best_match_id is not None:
            return best_match_id
        else:
            new_id = self.next_object_id
            self.next_object_id += 1
            return new_id

    def update_tracked_objects(self, detections, frame_time):
        """
        更新跟踪对象信息

        Args:
            detections: 检测结果列表
            frame_time: 当前帧时间
        """
        current_frame_objects = set()

        # 处理当前帧的检测结果
        for detection in detections:
            box = detection[:4]  # [x1, y1, x2, y2]
            class_name = detection[5]  # 类别名称
            object_id = detection[6] if len(detection) > 6 else self.assign_object_id(box, class_name)

            # 如果使用ByteTrack但没有跟踪ID，则跳过
            if self.use_bytetrack and object_id is None:
                continue

            current_frame_objects.add(object_id)

            # 更新对象信息
            if object_id not in self.tracked_objects:
                self.tracked_objects[object_id] = {
                    'class': class_name,
                    'first_seen': frame_time,
                    'last_position': box,
                    'positions': [box],
                    'timestamps': [frame_time]
                }
            else:
                self.tracked_objects[object_id]['last_position'] = box
                self.tracked_objects[object_id]['positions'].append(box)
                self.tracked_objects[object_id]['timestamps'].append(frame_time)

                # 只对人员进行徘徊检测
                if class_name == "person":
                    # 检查最近一段时间内的移动距离
                    if len(self.tracked_objects[object_id]['timestamps']) > 1:
                        recent_positions = self.tracked_objects[object_id]['positions'][-10:]  # 检查最近10个位置
                        recent_timestamps = self.tracked_objects[object_id]['timestamps'][-10:]

                        if len(recent_positions) > 1:
                            # 计算最近位置之间的距离
                            total_distance = 0
                            for i in range(1, len(recent_positions)):
                                pos1 = recent_positions[i-1]
                                pos2 = recent_positions[i]
                                # 计算中心点距离
                                center1 = [(pos1[0] + pos1[2]) / 2, (pos1[1] + pos1[3]) / 2]
                                center2 = [(pos2[0] + pos2[2]) / 2, (pos2[1] + pos2[3]) / 2]
                                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                                total_distance += distance

                            # 如果在阈值时间内移动距离很小，则认为在徘徊
                            time_diff = recent_timestamps[-1] - recent_timestamps[0]
                            if time_diff > 0:
                                # 检查是否在指定区域内停留超过阈值时间
                                first_seen_time = self.tracked_objects[object_id]['first_seen']
                                total_time = frame_time - first_seen_time

                                if total_time > self.loitering_time_threshold and total_distance < 50:  # 50像素作为移动阈值
                                    self.loitering_alarms[object_id] = {
                                        'start_time': first_seen_time,
                                        'current_time': frame_time,
                                        'duration': total_time,
                                        'position': box,
                                        'class': class_name
                                    }
                                # 如果移动距离较大，且之前有警报，则移除警报
                                elif total_distance >= 50 and object_id in self.loitering_alarms:
                                    del self.loitering_alarms[object_id]

        # 移除不在当前帧中的对象的警报
        removed_objects = set(self.loitering_alarms.keys()) - current_frame_objects
        for obj_id in removed_objects:
            if obj_id in self.loitering_alarms:
                del self.loitering_alarms[obj_id]

    def detect_loitering(self, frame, frame_time):
        """
        检测视频帧中的徘徊行为

        Args:
            frame: 视频帧
            frame_time: 帧时间戳

        Returns:
            results: 检测结果
            alarms: 徘徊警报
        """
        # 调整图像尺寸以提高处理速度
        h, w = frame.shape[:2]
        scale = self.img_size / max(h, w)
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            resized_frame = cv2.resize(frame, (new_w, new_h))
        else:
            resized_frame = frame
            scale = 1

        # 使用YOLOv12检测目标
        results = self.model(resized_frame, conf=self.conf_threshold, imgsz=self.img_size, device=self.device)

        detections = []
        # 首先使用YOLOv12的基本检测方法
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # 获取边界框坐标
                    coords = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())

                    # 恢复原始图像坐标
                    if scale < 1:
                        coords = coords / scale

                    # 获取类别名称
                    if hasattr(result, 'names') and class_id < len(result.names):
                        class_name = result.names[class_id]
                    else:
                        class_name = self.target_classes[class_id] if class_id < len(self.target_classes) else f"class_{class_id}"

                    # 只处理指定的类别（如"person"）
                    if class_name not in self.target_classes:
                        continue

                    # 检查是否在检测区域内
                    if self.is_in_detection_region(coords):
                        # 添加对象ID（使用自定义分配的ID）
                        object_id = self.assign_object_id(coords, class_name)
                        detections.append(list(coords) + [confidence, class_name, object_id])

        # 如果启用了ByteTrack增强功能并且跟踪器可用，则使用ByteTrack进行更精确的跟踪
        if self.use_bytetrack and hasattr(self, 'tracker'):
            # 清空基础检测结果，使用ByteTrack的结果
            detections = []
            # 使用ByteTrack进行跟踪
            self.frame_id += 1

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    # 准备ByteTrack输入
                    dets = []
                    for i, box in enumerate(boxes):
                        coords = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())

                        # 恢复原始图像坐标
                        if scale < 1:
                            coords = coords / scale

                        # 获取类别名称
                        if hasattr(result, 'names') and class_id < len(result.names):
                            class_name = result.names[class_id]
                        else:
                            class_name = self.target_classes[class_id] if class_id < len(self.target_classes) else f"class_{class_id}"

                        # 只处理指定的类别（如"person"）
                        if class_name not in self.target_classes:
                            continue

                        # 检查是否在检测区域内
                        if self.is_in_detection_region(coords):
                            # 构造ByteTrack需要的检测框格式 [x1, y1, x2, y2, score]
                            det = [coords[0], coords[1], coords[2], coords[3], confidence]
                            dets.append(det)
                            # 先添加检测结果，稍后添加跟踪ID
                            detections.append(list(coords) + [confidence, class_name])

            if dets:
                # 使用ByteTrack跟踪
                try:
                    online_targets = self.tracker.update(np.array(dets), [h, w], [h, w])

                    # 结合跟踪ID更新检测结果
                    for t in online_targets:
                        tlwh = t.tlwh
                        tid = t.track_id
                        vertical = tlwh[2] / tlwh[3] > 1.6
                        if tlwh[2] * tlwh[3] > 10 and not vertical:
                            x1, y1 = int(tlwh[0]), int(tlwh[1])
                            x2, y2 = int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])
                            # 添加跟踪ID到检测结果
                            for i, det in enumerate(detections):
                                det_coords = det[:4]
                                if (abs(det_coords[0] - x1) < 5 and abs(det_coords[1] - y1) < 5 and
                                    abs(det_coords[2] - x2) < 5 and abs(det_coords[3] - y2) < 5):
                                    detections[i].append(tid)
                except Exception as e:
                    print(f"Error in ByteTrack update: {e}")

        # 确保所有检测结果都有ID
        for i, detection in enumerate(detections):
            if len(detection) <= 6:  # 如果没有ID，则分配一个
                coords = detection[:4]
                class_name = detection[5]
                object_id = self.assign_object_id(coords, class_name)
                detections[i] = list(detection) + [object_id]

        # 更新跟踪对象
        self.update_tracked_objects(detections, frame_time)

        return detections, self.loitering_alarms

    def get_class_color(self, class_name):
        """
        获取类别的颜色

        Args:
            class_name: 类别名称

        Returns:
            color: BGR颜色值
        """
        return self.class_colors.get(class_name, (128, 128, 128))  # 默认灰色
