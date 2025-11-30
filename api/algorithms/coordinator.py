"""
视频处理协调器
协调各种检测算法的执行流程
"""

from typing import Optional, List, Tuple
from .loitering.processor import process_loitering_video, draw_loitering_detections
from .loitering.detector import LoiteringDetector
from .leave.processor import process_leave_video, draw_leave_detections
from .leave.detector import LeaveDetector
from .gather.processor import process_gather_video, draw_gather_detections
from .gather.detector import GatherDetector
from .banner.processor import process_banner_video, draw_banner_detections
from .banner.detector import BannerDetector


class VideoProcessingCoordinator:
    """
    视频处理协调器
    负责协调各种视频分析算法的执行
    """

    def __init__(self, camera_id: str, model_name: str = "yolov12n.pt"):
        """
        初始化视频处理协调器

        Args:
            camera_id: 摄像头ID
            model_name: 模型文件名
        """
        self.camera_id = camera_id
        self.model_name = model_name
        self.frame_rate = 30  # 默认帧率

    def _get_loitering_detector(self, loitering_time_threshold: int = 10):
        """
        获取徘徊检测器实例

        Args:
            loitering_time_threshold: 徘徊时间阈值（秒）

        Returns:
            LoiteringDetector: 徘徊检测器实例
        """
        return LoiteringDetector(
            model_name=self.model_name,
            loitering_time_threshold=loitering_time_threshold
        )

    def _get_leave_detector(self):
        """
        获取离岗检测器实例

        Returns:
            LeaveDetector: 离岗检测器实例
        """
        return LeaveDetector(
            model_path=self.model_name
        )

    def _get_gather_detector(self):
        """
        获取聚集检测器实例

        Returns:
            GatherDetector: 聚集检测器实例
        """
        return GatherDetector(
            model_path=self.model_name
        )

    def _get_banner_detector(self, conf_threshold: float = 0.3, iou_threshold: float = 0.45):
        """
        获取横幅检测器实例

        Args:
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU阈值

        Returns:
            BannerDetector: 横幅检测器实例
        """
        # 横幅检测使用专门的banner_weight.pt权重文件
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        banner_model_path = os.path.join(project_root, "yolov12", "banner_weight.pt")
        
        # 如果banner专用权重文件存在，则使用它；否则使用默认模型
        if os.path.exists(banner_model_path):
            model_path = banner_model_path
        else:
            print(f"[Coordinator] 未找到横幅专用权重文件 {banner_model_path}，使用默认模型 {self.model_name}")
            model_path = self.model_name
        
        return BannerDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )

    def _draw_leave_detections(self, frame, roi, status, roi_person_count, absence_start_time, threshold, alert_triggered):
        """
        绘制离岗检测结果

        Args:
            frame: 视频帧
            roi: ROI区域
            status: 状态（在岗/脱岗）
            roi_person_count: ROI内人数
            absence_start_time: 脱岗开始时间
            threshold: 脱岗阈值
            alert_triggered: 是否触发警报

        Returns:
            frame: 绘制了检测结果的帧
        """
        # 如果触发了离岗警报，发送到RabbitMQ
        if alert_triggered:
            from ..services.rabbitmq_service import rabbitmq_producer
            import json
            import uuid
            from datetime import datetime
            
            # 构建告警消息
            alarm_message = {
                "code": str(uuid.uuid4()),
                "alarmType": 1,
                "subType": "异常行为识别-离岗检测",
                "alarmTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "deviceCode": "camera_001",  # 默认摄像头ID，实际应该从上下文获取
                "deviceName": "摄像头001",
                "level": "warning",
                "memo": f"检测到离岗行为，离岗时间: {threshold}秒",
                "position": "",  # 可以考虑添加ROI坐标
                "personCode": "",
                "personName": "",
                "ext1": json.dumps({
                    "absence_duration": threshold,
                    "roi_person_count": roi_person_count
                })
            }
            
            # 发送到RabbitMQ
            try:
                success = rabbitmq_producer.send_message(alarm_message)
                if success:
                    print(f"[Leave] 告警消息发送成功: {alarm_message['memo']}")
                else:
                    print(f"[Leave] 告警消息发送失败: {alarm_message['memo']}")
            except Exception as e:
                print(f"[Leave] 发送告警消息时出错: {e}")
                
        return draw_leave_detections(frame, roi, status, roi_person_count, absence_start_time, threshold, alert_triggered)

    def _draw_gather_detections(self, frame, roi, roi_person_count, gather_threshold, alert_triggered):
        """
        绘制聚集检测结果

        Args:
            frame: 视频帧
            roi: ROI区域
            roi_person_count: ROI内人数
            gather_threshold: 聚集人数阈值
            alert_triggered: 是否触发警报

        Returns:
            frame: 绘制了检测结果的帧
        """
        # 如果触发了聚集警报，发送到RabbitMQ
        if alert_triggered:
            from ..services.rabbitmq_service import rabbitmq_producer
            import json
            import uuid
            from datetime import datetime
            
            # 构建告警消息
            alarm_message = {
                "code": str(uuid.uuid4()),
                "alarmType": 1,
                "subType": "异常行为识别-聚集检测",
                "alarmTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "deviceCode": "camera_001",  # 默认摄像头ID，实际应该从上下文获取
                "deviceName": "摄像头001",
                "level": "warning",
                "memo": f"检测到人员聚集，当前人数: {roi_person_count}，阈值: {gather_threshold}",
                "position": "",  # 可以考虑添加ROI坐标
                "personCode": "",
                "personName": "",
                "ext1": json.dumps({
                    "person_count": roi_person_count,
                    "threshold": gather_threshold
                })
            }
            
            # 发送到RabbitMQ
            try:
                success = rabbitmq_producer.send_message(alarm_message)
                if success:
                    print(f"[Gather] 告警消息发送成功: {alarm_message['memo']}")
                else:
                    print(f"[Gather] 告警消息发送失败: {alarm_message['memo']}")
            except Exception as e:
                print(f"[Gather] 发送告警消息时出错: {e}")
        
        return draw_gather_detections(frame, roi, roi_person_count, gather_threshold, alert_triggered)

    def _draw_banner_detections(self, frame, banners):
        """
        绘制横幅检测结果

        Args:
            frame: 视频帧
            banners: 检测到的横幅信息

        Returns:
            frame: 绘制了检测结果的帧
        """
        # 如果检测到横幅，发送到RabbitMQ
        # 只有在真正触发告警时（由detector控制）才发送消息
        if banners:
            from ..services.rabbitmq_service import rabbitmq_producer
            import json
            import uuid
            from datetime import datetime
            
            for banner in banners:
                # 构建告警消息
                alarm_message = {
                    "code": str(uuid.uuid4()),
                    "alarmType": 1,
                    "subType": "异常行为识别-横幅检测",
                    "alarmTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "deviceCode": "camera_001",  # 默认摄像头ID，实际应该从上下文获取
                    "deviceName": "摄像头001",
                    "level": "warning",
                    "memo": f"检测到横幅或可疑标语，置信度: {banner['confidence']:.2f}",
                    "position": f"[{banner['box'][0]},{banner['box'][1]},{banner['box'][2]},{banner['box'][3]}]",
                    "personCode": "",
                    "personName": "",
                    "ext1": json.dumps({
                        "confidence": banner['confidence'],
                        "class": banner['class']
                    })
                }
                
                # 发送到RabbitMQ
                try:
                    success = rabbitmq_producer.send_message(alarm_message)
                    if success:
                        print(f"[Banner] 告警消息发送成功: {alarm_message['memo']}")
                    else:
                        print(f"[Banner] 告警消息发送失败: {alarm_message['memo']}")
                except Exception as e:
                    print(f"[Banner] 发送告警消息时出错: {e}")
        
        return draw_banner_detections(frame, banners)

    def _draw_loitering_detections(self, frame, detections, alarms):
        """
        绘制徘徊检测结果

        Args:
            frame: 视频帧
            detections: 检测结果
            alarms: 警报信息

        Returns:
            frame: 绘制了检测结果的帧
        """
        # 如果有警报，发送到RabbitMQ
        if alarms:
            from ..services.rabbitmq_service import rabbitmq_producer
            import json
            import uuid
            from datetime import datetime
            
            for obj_id, alarm in alarms.items():
                # 构建告警消息
                alarm_message = {
                    "code": str(uuid.uuid4()),
                    "alarmType": 1,
                    "subType": "异常行为识别",
                    "alarmTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "deviceCode": "camera_001",  # 默认摄像头ID，实际应该从上下文获取
                    "deviceName": "摄像头001",
                    "level": "warning",
                    "memo": f"检测到徘徊行为，持续时间: {alarm['duration']:.1f}秒",
                    "image": None,
                    "position": f"[{alarm['position'][0]},{alarm['position'][1]},{alarm['position'][2]},{alarm['position'][3]}]",
                    "personCode": "",
                    "personName": "",
                    "ext1": json.dumps({
                        "object_id": str(obj_id),
                        "duration": alarm['duration']
                    })
                }
                
                # 发送到RabbitMQ
                try:
                    success = rabbitmq_producer.send_message(alarm_message)
                    if success:
                        print(f"[Loitering] 告警消息发送成功: {alarm_message['memo']}")
                    else:
                        print(f"[Loitering] 告警消息发送失败: {alarm_message['memo']}")
                except Exception as e:
                    print(f"[Loitering] 发送告警消息时出错: {e}")
        
        return draw_loitering_detections(frame, detections, alarms)

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
        return process_loitering_video(
            self.model_name,
            video_path,
            output_path,
            loitering_time_threshold,
            device
        )

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
        return process_leave_video(
            self.model_name,
            video_path,
            output_path,
            roi,
            absence_threshold,
            device
        )

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
        return process_gather_video(
            self.model_name,
            video_path,
            output_path,
            roi,
            gather_threshold,
            device
        )

    def process_banner_video(self,
                             video_path: str,
                             output_path: str,
                             roi: Optional[List[Tuple[int, int]]] = None,
                             conf_threshold: float = 0.3,
                             iou_threshold: float = 0.45,
                             device: str = 'cuda') -> str:
        """
        处理横幅检测视频

        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
            roi: ROI区域 [(x1, y1), (x2, y2), ...]
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU阈值
            device: 运行设备

        Returns:
            str: 处理后的视频路径
        """
        return process_banner_video(
            self.model_name,
            video_path,
            output_path,
            roi,
            conf_threshold,
            iou_threshold,
            device
        )