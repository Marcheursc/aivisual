from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
import cv2
import numpy as np
import os
import uuid
import tempfile
from typing import List, Dict, Optional
import asyncio
import io
from PIL import Image
import torch
import sys

# 添加项目路径
project_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(project_path)
sys.path.insert(0, parent_path)
sys.path.insert(0, project_path)

# 导入检测器
from detector.core import LoiteringDetector
from detector.video_source import VideoSourceManager
from detector.visualizer import Visualizer

# 初始化 FastAPI 应用
app = FastAPI(title="计算机视觉API",
              description="基于YOLOv12和ByteTrack的计算机视觉服务，提供人员检测、跟踪和异常行为检测功能")

# 存储目录
UPLOAD_DIR = os.path.join(project_path, "uploads")
PROCESSED_DIR = os.path.join(project_path, "processed_videos")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

processing_tasks = {}


class DetectionResult:
    def __init__(self, class_name: str, confidence: float, bbox: List[float], object_id: Optional[int] = None):
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox
        self.object_id = object_id


class LoiteringAlarm:
    def __init__(self, object_id: int, duration: float, position: List[float]):
        self.object_id = object_id
        self.duration = duration
        self.position = position


@app.get("/")
async def root():
    return {"message": "欢迎使用计算机视觉API", "version": "1.0.0"}


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """上传文件（图像或视频）"""
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

    # 保存文件
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    return {"file_id": file_id, "filename": file.filename, "saved_path": file_path}


@app.post("/process_video/")
async def process_video(
        file_id: str,
        background_tasks: BackgroundTasks,
        detect_loitering: bool = True,
        loitering_time_threshold: int = 20,
        detection_type: str = "loitering",  # 新增参数：检测类型
        # 离岗和聚集检测的额外参数
        leave_roi: Optional[str] = None,
        leave_threshold: Optional[int] = None,
        gather_roi: Optional[str] = None,
        gather_threshold: Optional[int] = None
):
    """处理视频文件"""
    # 查找文件
    file_path = None
    for filename in os.listdir(UPLOAD_DIR):
        if filename.startswith(file_id):
            file_path = os.path.join(UPLOAD_DIR, filename)
            break

    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件未找到")

    # 添加后台任务处理视频
    task_id = str(uuid.uuid4())
    processing_tasks[task_id] = {
        "status": "processing",
        "progress": 0,
        "result_path": None
    }

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
        background_tasks.add_task(
            process_leave_detection_task,
            file_path,
            task_id,
            parsed_leave_roi,
            leave_threshold
        )
    elif detection_type == "gather":
        # 聚集检测
        background_tasks.add_task(
            process_gather_detection_task,
            file_path,
            task_id,
            parsed_gather_roi,
            gather_threshold
        )
    else:
        # 默认为徘徊检测
        background_tasks.add_task(
            process_video_task,
            file_path,
            task_id,
            detect_loitering,
            loitering_time_threshold
        )

    return {"task_id": task_id, "message": f"{detection_type}视频处理已启动"}


async def process_video_task(
        video_path: str,
        task_id: str,
        detect_loitering: bool = True,
        loitering_time_threshold: int = 20
):
    """后台处理视频任务"""
    try:
        # 初始化视频源管理器
        video_manager = VideoSourceManager([video_path])
        cap, source_type = video_manager.open_video_source()

        if cap is None:
            raise Exception("无法打开视频文件")

        # 获取视频属性
        fps, width, height = video_manager.get_video_properties()

        # 初始化检测器
        detector = LoiteringDetector(loitering_time_threshold=loitering_time_threshold)
        visualizer = Visualizer(detector)

        # 设置输出视频路径
        output_filename = f"processed_{uuid.uuid4()}.mp4"
        output_path = os.path.join(PROCESSED_DIR, output_filename)

        # 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 处理帧
            if detect_loitering:
                detections, alarms = detector.detect_loitering(frame, frame_count / fps)
                annotated_frame = visualizer.draw_detections(frame, detections, alarms)
            else:
                # 只进行基本检测
                detections, _ = detector.detect_loitering(frame, frame_count / fps)
                alarms = {}
                annotated_frame = visualizer.draw_detections(frame, detections, alarms)

            # 写入处理后的帧
            out.write(annotated_frame)

            # 更新进度
            if total_frames > 0:
                progress = int((frame_count / total_frames) * 100)
                processing_tasks[task_id]["progress"] = progress

        # 释放资源
        cap.release()
        out.release()
        video_manager.release()

        # 标记为完成
        processing_tasks[task_id]["status"] = "completed"
        processing_tasks[task_id]["result_path"] = output_path
        processing_tasks[task_id]["frame_count"] = frame_count

    except Exception as e:
        processing_tasks[task_id]["status"] = "failed"
        processing_tasks[task_id]["error"] = str(e)


@app.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
    """获取任务状态"""
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="任务未找到")

    return processing_tasks[task_id]


@app.get("/download_processed/{task_id}")
async def download_processed_video(task_id: str):
    """下载处理后的视频"""
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="任务未找到")

    task = processing_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="任务尚未完成")

    if not task["result_path"] or not os.path.exists(task["result_path"]):
        raise HTTPException(status_code=404, detail="处理后的视频文件未找到")

    def iterfile():
        with open(task["result_path"], mode="rb") as file_like:
            yield from file_like

    return StreamingResponse(iterfile(), media_type="video/mp4")


async def process_leave_detection_task(
        video_path: str,
        task_id: str,
        roi: Optional[list] = None,
        threshold: Optional[int] = None
):
    """离岗检测处理任务"""
    try:
        # 导入离岗检测模块
        sys.path.append(os.path.join(project_path, 'leave_gather_detection'))
        from leave_gather_detection.leave import process_leave_detection
        
        # 处理视频
        result_path = process_leave_detection(video_path, roi, threshold)
        
        # 标记为完成
        processing_tasks[task_id]["status"] = "completed"
        processing_tasks[task_id]["result_path"] = result_path
        processing_tasks[task_id]["frame_count"] = 0  # 这里需要根据实际处理情况更新

    except Exception as e:
        processing_tasks[task_id]["status"] = "failed"
        processing_tasks[task_id]["error"] = str(e)


async def process_gather_detection_task(
        video_path: str,
        task_id: str,
        roi: Optional[list] = None,
        threshold: Optional[int] = None
):
    """聚集检测处理任务"""
    try:
        # 导入聚集检测模块
        sys.path.append(os.path.join(project_path, 'leave_gather_detection'))
        from leave_gather_detection.main import process_gather_detection
        
        # 处理视频
        result_path = process_gather_detection(video_path, roi, threshold)
        
        # 标记为完成
        processing_tasks[task_id]["status"] = "completed"
        processing_tasks[task_id]["result_path"] = result_path
        processing_tasks[task_id]["frame_count"] = 0  # 这里需要根据实际处理情况更新

    except Exception as e:
        processing_tasks[task_id]["status"] = "failed"
        processing_tasks[task_id]["error"] = str(e)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)