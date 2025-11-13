import requests
import json
import time

# API服务器地址
BASE_URL = "http://localhost:8000"


def upload_file(file_path):
    """上传文件"""
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(f"{BASE_URL}/upload/", files=files)
        return response.json()


def detect_objects(image_path):
    """对图像进行对象检测"""
    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(f"{BASE_URL}/detect/", files=files)
        return response.json()


def process_video(file_id, detect_loitering=True, loitering_time_threshold=20):
    """处理视频文件"""
    params = {
        "file_id": file_id,
        "detect_loitering": detect_loitering,
        "loitering_time_threshold": loitering_time_threshold
    }
    response = requests.post(f"{BASE_URL}/process_video/", params=params)
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
