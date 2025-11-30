"""
摄像头服务层
处理摄像头相关的业务逻辑
"""

import os
import json
from typing import List, Dict, Any, Optional
from ..config.settings import UPLOAD_DIR, PROCESSED_DIR
from ..algorithms import VideoProcessingCoordinator


class CameraService:
    """摄像头服务类"""

    def __init__(self):
        """初始化摄像头服务"""
        self.cameras_data = []
        self.camera_scene_mapping = {}
        self.camera_device_mapping = {}
        self.initialize_cameras()

    def initialize_cameras(self):
        """
        初始化摄像头数据（在真实项目中，这应该从数据库加载）
        """
        # 检查是否存在摄像头配置文件
        config_file = os.path.join(os.path.dirname(__file__), "..", "config", "cameras.json")
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.cameras_data = config.get("cameras", [])
                    self.camera_scene_mapping = config.get("camera_scenes", {})
                    self.camera_device_mapping = config.get("camera_devices", {})
            except Exception as e:
                print(f"加载摄像头配置文件失败: {e}")
                # 初始化为空列表和字典
                self.cameras_data = []
                self.camera_scene_mapping = {}
                self.camera_device_mapping = {}
        else:
            # 如果配置文件不存在，初始化为空
            self.cameras_data = []
            self.camera_scene_mapping = {}
            self.camera_device_mapping = {}

    def get_all_cameras(self) -> List[Dict[str, Any]]:
        """
        获取所有摄像头列表

        Returns:
            List[Dict[str, Any]]: 摄像头列表
        """
        return self.cameras_data

    def get_camera_scene(self, camera_id: str) -> Dict[str, Any]:
        """
        获取指定摄像头的场景类型

        Args:
            camera_id: 摄像头ID

        Returns:
            Dict[str, Any]: 摄像头场景信息
        """
        if camera_id not in self.camera_scene_mapping:
            raise ValueError("摄像头未找到或未分配场景")

        scene_type = self.camera_scene_mapping[camera_id]
        return {"camera_id": camera_id, "scene_type": scene_type}

    def assign_camera_to_scene(self, camera_id: str, scene_type: str) -> Dict[str, Any]:
        """
        将摄像头分配到指定场景

        Args:
            camera_id: 摄像头ID
            scene_type: 场景类型

        Returns:
            Dict[str, Any]: 分配结果
        """
        # 验证摄像头是否存在
        camera_exists = any(cam["id"] == camera_id for cam in self.cameras_data)
        if not camera_exists:
            raise ValueError("摄像头未找到")

        # 验证场景类型
        if scene_type not in ["loitering", "leave", "gather", "banner"]:
            raise ValueError("无效的场景类型")

        # 分配场景
        self.camera_scene_mapping[camera_id] = scene_type

        # 在真实项目中，这里应该保存到数据库
        # self.save_camera_config()

        return {"message": f"摄像头 {camera_id} 已成功分配到 {scene_type} 场景"}

    def bind_camera_to_device(self, camera_id: str, device_source: str) -> Dict[str, Any]:
        """
        将摄像头绑定到设备源

        Args:
            camera_id: 摄像头ID
            device_source: 设备源

        Returns:
            Dict[str, Any]: 绑定结果
        """
        # 验证摄像头是否存在
        camera_exists = any(cam["id"] == camera_id for cam in self.cameras_data)
        if not camera_exists:
            raise ValueError("摄像头未找到")

        # 尝试解析设备源为数字
        try:
            device_source = int(device_source)
        except ValueError:
            # 如果不是数字，则保持为字符串（如RTSP地址）
            pass

        # 绑定设备
        self.camera_device_mapping[camera_id] = device_source

        # 在真实项目中，这里应该保存到数据库
        # self.save_camera_config()

        return {"message": f"摄像头 {camera_id} 已成功绑定到设备源 {device_source}"}

    def unbind_camera_device(self, camera_id: str) -> Dict[str, Any]:
        """
        解除摄像头与设备的绑定

        Args:
            camera_id: 摄像头ID

        Returns:
            Dict[str, Any]: 解绑结果
        """
        # 验证摄像头是否存在
        camera_exists = any(cam["id"] == camera_id for cam in self.cameras_data)
        if not camera_exists:
            raise ValueError("摄像头未找到")

        # 检查是否已绑定设备
        if camera_id not in self.camera_device_mapping:
            raise ValueError("摄像头未绑定任何设备")

        # 解除绑定
        removed_device = self.camera_device_mapping.pop(camera_id)

        # 在真实项目中，这里应该保存到数据库
        # self.save_camera_config()

        return {"message": f"摄像头 {camera_id} 已解除与设备 {removed_device} 的绑定"}

    def get_camera_device(self, camera_id: str) -> Dict[str, Any]:
        """
        获取摄像头绑定的设备源

        Args:
            camera_id: 摄像头ID

        Returns:
            Dict[str, Any]: 设备信息
        """
        if camera_id not in self.camera_device_mapping:
            raise ValueError("摄像头未找到或未绑定设备")

        device_source = self.camera_device_mapping[camera_id]
        return {"camera_id": camera_id, "device_source": device_source}

    def add_camera(self, camera_id: str, name: str, location: str) -> Dict[str, Any]:
        """
        添加新的摄像头

        Args:
            camera_id: 摄像头ID
            name: 摄像头名称
            location: 摄像头位置

        Returns:
            Dict[str, Any]: 添加结果
        """
        # 检查摄像头ID是否已存在
        if any(cam["id"] == camera_id for cam in self.cameras_data):
            raise ValueError("摄像头ID已存在")

        # 添加新摄像头
        new_camera = {
            "id": camera_id,
            "name": name,
            "location": location,
            "status": "inactive"  # 初始状态为未激活
        }
        self.cameras_data.append(new_camera)

        # 在真实项目中，这里应该保存到数据库
        # self.save_camera_config()

        return {"message": f"摄像头 {camera_id} 已成功添加", "camera": new_camera}

    def remove_camera(self, camera_id: str) -> Dict[str, Any]:
        """
        删除摄像头

        Args:
            camera_id: 摄像头ID

        Returns:
            Dict[str, Any]: 删除结果
        """
        # 查找要删除的摄像头
        camera_index = None
        for i, cam in enumerate(self.cameras_data):
            if cam["id"] == camera_id:
                camera_index = i
                break

        if camera_index is None:
            raise ValueError("摄像头未找到")

        # 删除摄像头
        removed_camera = self.cameras_data.pop(camera_index)

        # 同时删除相关的场景和设备映射
        if camera_id in self.camera_scene_mapping:
            del self.camera_scene_mapping[camera_id]

        if camera_id in self.camera_device_mapping:
            del self.camera_device_mapping[camera_id]

        # 在真实项目中，这里应该保存到数据库
        # self.save_camera_config()

        return {"message": f"摄像头 {camera_id} 已成功删除", "camera": removed_camera}

    def get_camera_source(self, camera_id: str) -> Any:
        """
        获取摄像头的视频源

        Args:
            camera_id: 摄像头ID

        Returns:
            Any: 视频源
        """
        if camera_id in self.camera_device_mapping:
            return self.camera_device_mapping[camera_id]
        else:
            # 默认使用系统摄像头0
            return 0

    def process_loitering_stream(self, camera_id: str, loitering_time_threshold: int = 20):
        """
        处理摄像头徘徊检测视频流

        Args:
            camera_id: 摄像头ID
            loitering_time_threshold: 徘徊时间阈值（秒）

        Yields:
            bytes: 编码后的视频帧
        """
        from ..algorithms import VideoProcessingCoordinator
        import cv2

        # 初始化视频处理器
        processor = VideoProcessingCoordinator(camera_id=camera_id)

        # 获取摄像头源
        camera_source = self.get_camera_source(camera_id)

        # 打开摄像头
        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            raise Exception(f"无法打开摄像头 {camera_id} (源: {camera_source})")

        try:
            # 初始化检测器
            detector = processor._get_loitering_detector(loitering_time_threshold=loitering_time_threshold)

            frame_count = 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 30  # 默认FPS为30
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                frame_time = frame_count / fps

                # 执行徘徊检测
                detections, alarms = detector.detect_loitering(frame, frame_time)

                # 在帧上绘制检测结果
                annotated_frame = processor._draw_loitering_detections(frame, detections, alarms)

                # 编码帧
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        finally:
            cap.release()

    def process_leave_stream(self, camera_id: str, roi: list = None, threshold: int = None):
        """
        处理摄像头离岗检测视频流

        Args:
            camera_id: 摄像头ID
            roi: ROI区域
            threshold: 阈值

        Yields:
            bytes: 编码后的视频帧
        """
        from ..algorithms import VideoProcessingCoordinator
        import cv2

        # 初始化视频处理器
        processor = VideoProcessingCoordinator(camera_id=camera_id)

        # 获取摄像头源
        camera_source = self.get_camera_source(camera_id)

        # 打开摄像头
        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            raise Exception(f"无法打开摄像头 {camera_id} (源: {camera_source})")

        try:
            # 初始化检测器
            detector = processor._get_leave_detector()

            # 设置默认ROI区域（如果没有通过参数传递）
            if roi is None:
                # 默认ROI区域可以根据您的需要修改
                roi = [(220, 300), (700, 300), (700, 700), (200, 700)]

            # 状态变量
            absence_start_time = None
            alert_triggered = False

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 执行离岗检测
                result = detector.detect_leave(frame, roi, absence_start_time, threshold if threshold is not None else 5)
                absence_start_time = result['absence_start_time']

                # 在帧上绘制检测结果
                annotated_frame = processor._draw_leave_detections(
                    frame, roi, result['status'], result['roi_person_count'],
                    absence_start_time, threshold if threshold is not None else 5, result['alert_triggered']
                )

                # 绘制检测到的人员框
                for box in result['person_boxes']:  # 绘制所有检测到的人员框
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # 编码帧
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        finally:
            cap.release()

    def process_gather_stream(self, camera_id: str, roi: list = None, threshold: int = None):
        """
        处理摄像头聚集检测视频流

        Args:
            camera_id: 摄像头ID
            roi: ROI区域
            threshold: 阈值

        Yields:
            bytes: 编码后的视频帧
        """
        from ..algorithms import VideoProcessingCoordinator
        import cv2

        # 初始化视频处理器
        processor = VideoProcessingCoordinator(camera_id=camera_id)

        # 获取摄像头源
        camera_source = self.get_camera_source(camera_id)

        # 打开摄像头
        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            raise Exception(f"无法打开摄像头 {camera_id} (源: {camera_source})")

        try:
            # 初始化检测器
            detector = processor._get_gather_detector()

            # 设置默认ROI区域（如果没有通过参数传递）
            if roi is None:
                # 默认ROI区域可以根据您的需要修改
                roi = [(220, 300), (700, 300), (700, 700), (200, 700)]

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 执行聚集检测
                result = detector.detect_gather(frame, roi, threshold if threshold is not None else 5)

                # 在帧上绘制检测结果
                annotated_frame = processor._draw_gather_detections(
                    frame, roi, result['roi_person_count'], threshold if threshold is not None else 5, result['alert_triggered']
                )

                # 绘制检测到的人员框（仅ROI区域内的人员框）
                for box in result['roi_person_boxes']:
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # 编码帧
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        finally:
            cap.release()

    def process_banner_stream(self, camera_id: str, roi: list = None, conf_threshold: float = None, iou_threshold: float = None):
        """
        处理摄像头横幅检测视频流

        Args:
            camera_id: 摄像头ID
            roi: ROI区域
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值

        Yields:
            bytes: 编码后的视频帧
        """
        from ..algorithms import VideoProcessingCoordinator
        import cv2

        # 初始化视频处理器
        processor = VideoProcessingCoordinator(camera_id=camera_id)

        # 获取摄像头源
        camera_source = self.get_camera_source(camera_id)

        # 打开摄像头
        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            raise Exception(f"无法打开摄像头 {camera_id} (源: {camera_source})")

        try:
            # 初始化检测器
            detector = processor._get_banner_detector(
                conf_threshold=conf_threshold if conf_threshold is not None else 0.5,
                iou_threshold=iou_threshold if iou_threshold is not None else 0.45
            )

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 执行横幅检测
                results, banners = detector.detect_banner(frame)

                # 在帧上绘制检测结果
                annotated_frame = processor._draw_banner_detections(frame, banners)

                # 编码帧
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        finally:
            cap.release()
