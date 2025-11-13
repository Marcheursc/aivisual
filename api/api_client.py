import requests
import json

# FastAPI 服务器地址
BASE_URL = "http://localhost:8000"


def upload_video(file_path):
    """上传视频文件"""
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(f"{BASE_URL}/upload_video/", files=files)
        return response.json()


def detect_objects(image_path):
    """对图像进行对象检测"""
    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(f"{BASE_URL}/detect_objects/", files=files)
        return response.json()


def track_objects(video_id):
    """开始视频对象跟踪"""
    response = requests.post(f"{BASE_URL}/track_objects/?video_id={video_id}")
    return response.json()


def get_task_status(task_id):
    """获取任务状态"""
    response = requests.get(f"{BASE_URL}/task_status/{task_id}")
    return response.json()


def get_video_analytics(video_id):
    """获取视频分析结果"""
    response = requests.get(f"{BASE_URL}/analytics/{video_id}")
    return response.json()


# 使用示例
if __name__ == "__main__":
    print("API客户端示例")

    # 1. 上传视频
    # 注意：你需要有一个实际的视频文件路径
    # upload_result = upload_video("path/to/your/video.mp4")
    # print("上传结果:", upload_result)
    # video_id = upload_result["video_id"]

    # 2. 对图像进行对象检测
    # 注意：你需要有一个实际的图像文件路径
    # detection_result = detect_objects("path/to/your/image.jpg")
    # print("检测结果:", detection_result)

    # 3. 开始视频跟踪
    # track_result = track_objects(video_id)
    # print("跟踪任务:", track_result)
    # task_id = track_result["task_id"]

    # 4. 查询任务状态
    # status = get_task_status(task_id)
    # print("任务状态:", status)

    # 5. 获取分析结果
    # analytics = get_video_analytics(video_id)
    # print("分析结果:", analytics)

    print("请确保FastAPI服务器正在运行:")
    print("1. 运行 'python api_server.py' 启动服务器")
    print("2. 将上面的注释代码取消注释并替换为实际文件路径")
