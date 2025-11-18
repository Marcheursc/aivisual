import requests
import json
import time
from typing import List, Optional

# API服务器地址
BASE_URL = "http://localhost:8000"


def upload_file(file_path):
    """上传文件"""
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(f"{BASE_URL}/upload/", files=files)
        return response.json()


def process_video(
    file_id,
    detection_type="loitering",
    loitering_time_threshold=20,
    leave_roi=None,
    leave_threshold=None,
    gather_roi=None,
    gather_threshold=None,
    camera_id="default"
):
    """处理视频文件"""
    data = {
        "file_id": file_id,
        "detection_type": detection_type,
        "loitering_time_threshold": loitering_time_threshold,
        "camera_id": camera_id
    }

    # 添加可选参数
    if leave_roi is not None:
        data["leave_roi"] = leave_roi
    if leave_threshold is not None:
        data["leave_threshold"] = leave_threshold
    if gather_roi is not None:
        data["gather_roi"] = gather_roi
    if gather_threshold is not None:
        data["gather_threshold"] = gather_threshold

    response = requests.post(f"{BASE_URL}/process_video/", data=data)
    return response.json()


def get_task_status(task_id):
    """获取任务状态"""
    response = requests.get(f"{BASE_URL}/task_status/{task_id}")
    return response.json()


def download_processed_video(task_id, output_path):
    """下载处理后的视频"""
    response = requests.get(f"{BASE_URL}/download_processed/{task_id}")
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        return {"message": f"视频已保存到 {output_path}"}
    else:
        return {"error": "下载失败"}


# 摄像头管理相关函数
def get_all_cameras():
    """获取所有摄像头列表"""
    response = requests.get(f"{BASE_URL}/cameras")
    return response.json()


def get_camera_scene(camera_id):
    """获取指定摄像头的场景类型"""
    response = requests.get(f"{BASE_URL}/cameras/scene/{camera_id}")
    return response.json()


def assign_camera_to_scene(camera_id, scene_type):
    """将摄像头分配到指定场景"""
    data = {
        "camera_id": camera_id,
        "scene_type": scene_type
    }
    response = requests.post(f"{BASE_URL}/cameras/assign_scene", data=data)
    return response.json()


def bind_camera_to_device(camera_id, device_source):
    """将摄像头绑定到设备源"""
    data = {
        "camera_id": camera_id,
        "device_source": device_source
    }
    response = requests.post(f"{BASE_URL}/cameras/bind_device", data=data)
    return response.json()


def get_camera_device(camera_id):
    """获取摄像头绑定的设备源"""
    response = requests.get(f"{BASE_URL}/cameras/device/{camera_id}")
    return response.json()


def add_camera(camera_id, name, location):
    """添加新的摄像头"""
    data = {
        "camera_id": camera_id,
        "name": name,
        "location": location
    }
    response = requests.post(f"{BASE_URL}/cameras", data=data)
    return response.json()


def remove_camera(camera_id):
    """删除摄像头"""
    response = requests.delete(f"{BASE_URL}/cameras/{camera_id}")
    return response.json()


# 报警查询相关函数
def query_alerts(
    camera_ids: List[str],
    scene_type: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
):
    """查询报警数据"""
    params = {
        "camera_ids": camera_ids,
        "scene_type": scene_type
    }

    if start_time:
        params["start_time"] = start_time
    if end_time:
        params["end_time"] = end_time

    response = requests.get(f"{BASE_URL}/alerts", params=params)
    return response.json()


def yolov12_detect(image_path, model_id="yolov12n.pt", image_size=640, conf_threshold=0.25):
    """使用YOLOv12进行对象检测"""
    with open(image_path, "rb") as f:
        files = {"file": f}
        data = {
            "model_id": model_id,
            "image_size": image_size,
            "conf_threshold": conf_threshold
        }
        response = requests.post(f"{BASE_URL}/yolov12/detect/", files=files, data=data)
        return response.json()


def yolov12_process_video(file_id, model_id="yolov12n.pt", image_size=640, conf_threshold=0.25):
    """使用YOLOv12处理视频"""
    params = {
        "file_id": file_id,
        "model_id": model_id,
        "image_size": image_size,
        "conf_threshold": conf_threshold
    }
    response = requests.post(f"{BASE_URL}/yolov12/process_video/", params=params)
    return response.json()


if __name__ == "__main__":
    print("请确保API服务器正在运行:")
    print("1. 运行 'python cv_api.py' 启动服务器")
    print("2. 使用客户端函数与API进行交互")
