"""
摄像头相关路由
处理实时摄像头流和摄像头管理功能
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List, Optional
import cv2
import json
from datetime import datetime
import os
from api.algorithms.video_processor import VideoProcessor

router = APIRouter()

# 摄像头信息存储（实际项目中应该使用数据库）
cameras_data = []

# 摄像头与场景的映射关系（实际项目中应该使用数据库）
camera_scene_mapping = {}

# 摄像头与设备的映射关系（实际项目中应该使用数据库）
camera_device_mapping = {}

def initialize_cameras():
    """
    初始化摄像头数据（在真实项目中，这应该从数据库加载）
    """
    global cameras_data, camera_scene_mapping, camera_device_mapping

    # 检查是否存在摄像头配置文件
    config_file = os.path.join(os.path.dirname(__file__), "..", "config", "cameras.json")
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                cameras_data = config.get("cameras", [])
                camera_scene_mapping = config.get("camera_scenes", {})
                camera_device_mapping = config.get("camera_devices", {})
        except Exception as e:
            print(f"加载摄像头配置文件失败: {e}")
            # 初始化为空列表和字典
            cameras_data = []
            camera_scene_mapping = {}
            camera_device_mapping = {}
    else:
        # 如果配置文件不存在，初始化为空
        cameras_data = []
        camera_scene_mapping = {}
        camera_device_mapping = {}

# 初始化摄像头数据
initialize_cameras()

@router.get("/cameras")
async def get_all_cameras():
    """
    获取所有摄像头列表
    """
    return JSONResponse(content=cameras_data)


@router.get("/cameras/scene/{camera_id}")
async def get_camera_scene(camera_id: str):
    """
    获取指定摄像头的场景类型
    - camera_id: 摄像头ID
    """
    if camera_id not in camera_scene_mapping:
        raise HTTPException(status_code=404, detail="摄像头未找到或未分配场景")

    scene_type = camera_scene_mapping[camera_id]
    return JSONResponse(content={"camera_id": camera_id, "scene_type": scene_type})


@router.post("/cameras/assign_scene")
async def assign_camera_to_scene(camera_id: str, scene_type: str):
    """
    将摄像头分配到指定场景
    - camera_id: 摄像头ID
    - scene_type: 场景类型 (loitering, leave, gather)
    """
    # 验证摄像头是否存在
    camera_exists = any(cam["id"] == camera_id for cam in cameras_data)
    if not camera_exists:
        raise HTTPException(status_code=404, detail="摄像头未找到")

    # 验证场景类型
    if scene_type not in ["loitering", "leave", "gather"]:
        raise HTTPException(status_code=400, detail="无效的场景类型")

    # 分配场景
    camera_scene_mapping[camera_id] = scene_type

    # 在真实项目中，这里应该保存到数据库
    # save_camera_config()

    return JSONResponse(content={"message": f"摄像头 {camera_id} 已成功分配到 {scene_type} 场景"})


@router.post("/cameras/bind_device")
async def bind_camera_to_device(camera_id: str, device_source: str):
    """
    将摄像头绑定到设备源
    - camera_id: 摄像头ID
    - device_source: 设备源 (可以是数字如0,1,2或RTSP地址如rtsp://192.168.1.100:554/stream1)
    """
    # 验证摄像头是否存在
    camera_exists = any(cam["id"] == camera_id for cam in cameras_data)
    if not camera_exists:
        raise HTTPException(status_code=404, detail="摄像头未找到")

    # 尝试解析设备源为数字
    try:
        device_source = int(device_source)
    except ValueError:
        # 如果不是数字，则保持为字符串（如RTSP地址）
        pass

    # 绑定设备
    camera_device_mapping[camera_id] = device_source

    # 在真实项目中，这里应该保存到数据库
    # save_camera_config()

    return JSONResponse(content={"message": f"摄像头 {camera_id} 已成功绑定到设备源 {device_source}"})


@router.delete("/cameras/unbind_device/{camera_id}")
async def unbind_camera_device(camera_id: str):
    """
    解除摄像头与设备的绑定
    - camera_id: 摄像头ID
    """
    # 验证摄像头是否存在
    camera_exists = any(cam["id"] == camera_id for cam in cameras_data)
    if not camera_exists:
        raise HTTPException(status_code=404, detail="摄像头未找到")

    # 检查是否已绑定设备
    if camera_id not in camera_device_mapping:
        raise HTTPException(status_code=400, detail="摄像头未绑定任何设备")

    # 解除绑定
    removed_device = camera_device_mapping.pop(camera_id)

    # 在真实项目中，这里应该保存到数据库
    # save_camera_config()

    return JSONResponse(content={"message": f"摄像头 {camera_id} 已解除与设备 {removed_device} 的绑定"})


@router.get("/cameras/device/{camera_id}")
async def get_camera_device(camera_id: str):
    """
    获取摄像头绑定的设备源
    - camera_id: 摄像头ID
    """
    if camera_id not in camera_device_mapping:
        raise HTTPException(status_code=404, detail="摄像头未找到或未绑定设备")

    device_source = camera_device_mapping[camera_id]
    return JSONResponse(content={"camera_id": camera_id, "device_source": device_source})


@router.post("/cameras")
async def add_camera(camera_id: str, name: str, location: str):
    """
    添加新的摄像头
    - camera_id: 摄像头ID
    - name: 摄像头名称
    - location: 摄像头位置
    """
    # 检查摄像头ID是否已存在
    if any(cam["id"] == camera_id for cam in cameras_data):
        raise HTTPException(status_code=400, detail="摄像头ID已存在")

    # 添加新摄像头
    new_camera = {
        "id": camera_id,
        "name": name,
        "location": location,
        "status": "inactive"  # 初始状态为未激活
    }
    cameras_data.append(new_camera)

    # 在真实项目中，这里应该保存到数据库
    # save_camera_config()

    return JSONResponse(content={"message": f"摄像头 {camera_id} 已成功添加", "camera": new_camera})


@router.delete("/cameras/{camera_id}")
async def remove_camera(camera_id: str):
    """
    删除摄像头
    - camera_id: 摄像头ID
    """
    global cameras_data

    # 查找要删除的摄像头
    camera_index = None
    for i, cam in enumerate(cameras_data):
        if cam["id"] == camera_id:
            camera_index = i
            break

    if camera_index is None:
        raise HTTPException(status_code=404, detail="摄像头未找到")

    # 删除摄像头
    removed_camera = cameras_data.pop(camera_index)

    # 同时删除相关的场景和设备映射
    if camera_id in camera_scene_mapping:
        del camera_scene_mapping[camera_id]

    if camera_id in camera_device_mapping:
        del camera_device_mapping[camera_id]

    # 在真实项目中，这里应该保存到数据库
    # save_camera_config()

    return JSONResponse(content={"message": f"摄像头 {camera_id} 已成功删除", "camera": removed_camera})


@router.post("/process_camera/")
async def process_camera(
        camera_id: str = "default",
        detection_type: str = "loitering",
        loitering_time_threshold: int = 20,
        leave_roi: Optional[str] = None,
        leave_threshold: Optional[int] = None,
        gather_roi: Optional[str] = None,
        gather_threshold: Optional[int] = None
):
    """
    实时处理摄像头视频流
    """
    # 检查摄像头是否已分配场景
    if camera_id != "default" and camera_id in camera_scene_mapping:
        expected_scene = camera_scene_mapping[camera_id]
        if detection_type != expected_scene:
            # 可以选择是否允许用户覆盖默认场景分配
            pass  # 这里我们允许用户指定不同的检测类型

    # 解析ROI参数
    parsed_leave_roi = None
    parsed_gather_roi = None

    if leave_roi:
        try:
            # 解析格式如: "[(600,100),(1000,100),(1000,700),(600,700)]"
            parsed_leave_roi = eval(leave_roi)
        except:
            pass

    if gather_roi:
        try:
            # 解析格式如: "[(220,300),(700,300),(700,700),(200,700)]"
            parsed_gather_roi = eval(gather_roi)
        except:
            pass

    # 根据检测类型选择不同的处理函数
    if detection_type == "leave":
        # 离岗检测
        return StreamingResponse(
            process_camera_leave_stream(camera_id, parsed_leave_roi, leave_threshold),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    elif detection_type == "gather":
        # 聚集检测
        return StreamingResponse(
            process_camera_gather_stream(camera_id, parsed_gather_roi, gather_threshold),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    else:
        # 默认为徘徊检测
        return StreamingResponse(
            process_camera_loitering_stream(camera_id, loitering_time_threshold),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )


def get_camera_source(camera_id: str):
    """
    获取摄像头的视频源
    """
    if camera_id in camera_device_mapping:
        return camera_device_mapping[camera_id]
    else:
        # 默认使用系统摄像头0
        return 0


def process_camera_loitering_stream(camera_id: str, loitering_time_threshold: int = 20):
    """处理摄像头徘徊检测视频流"""
    # 初始化视频处理器
    processor = VideoProcessor()

    # 获取摄像头源
    camera_source = get_camera_source(camera_id)

    # 打开摄像头
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail=f"无法打开摄像头 {camera_id} (源: {camera_source})")

    try:
        # 初始化检测器
        detector = processor._get_loitering_detector(loitering_time_threshold=loitering_time_threshold)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 执行徘徊检测
            detections, alarms = detector.detect_loitering(frame, 0)  # 时间戳暂时设为0

            # 在帧上绘制检测结果
            annotated_frame = processor._draw_loitering_detections(frame, detections, alarms)

            # 编码帧
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()


def process_camera_leave_stream(camera_id: str, roi: Optional[list] = None, threshold: Optional[int] = None):
    """处理摄像头离岗检测视频流"""
    # 初始化视频处理器
    processor = VideoProcessor()

    # 获取摄像头源
    camera_source = get_camera_source(camera_id)

    # 打开摄像头
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail=f"无法打开摄像头 {camera_id} (源: {camera_source})")

    try:
        # 初始化检测器
        detector = processor._get_leave_detector()

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
            for box in result['person_boxes']:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 编码帧
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()


def process_camera_gather_stream(camera_id: str, roi: Optional[list] = None, threshold: Optional[int] = None):
    """处理摄像头聚集检测视频流"""
    # 初始化视频处理器
    processor = VideoProcessor()

    # 获取摄像头源
    camera_source = get_camera_source(camera_id)

    # 打开摄像头
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail=f"无法打开摄像头 {camera_id} (源: {camera_source})")

    try:
        # 初始化检测器
        detector = processor._get_gather_detector()

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

            # 绘制检测到的人员框
            for box in result['person_boxes']:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # 编码帧
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()
