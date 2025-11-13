from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
import cv2
import numpy as np
import os
import uuid
from typing import List, Dict
import asyncio
import io
from PIL import Image
import sys

# 添加项目路径
project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_path)

# 尝试导入yolov12模块
try:
    project_yolov12_path = os.path.join(project_path, 'yolov12')
    if project_yolov12_path not in sys.path:
        sys.path.insert(0, project_yolov12_path)
    from ultralytics import YOLO
    YOLOV12_AVAILABLE = True
except ImportError:
    YOLOV12_AVAILABLE = False
    print("YOLOv12 not available, some features will be disabled")

# 初始化 FastAPI 应用
app = FastAPI(title="视频分析API", description="基于YOLO和ByteTrack的视频分析服务")

# 模拟存储目录
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# 模拟检测结果模型
class DetectionResult:
    def __init__(self, class_name: str, confidence: float, bbox: List[float]):
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # [x, y, width, height]


# 模拟视频处理任务存储
processing_tasks = {}


@app.get("/")
async def root():
    return {"message": "欢迎使用AI视频分析API", "version": "1.0.0"}


@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    """上传视频文件"""
    # 生成唯一ID
    video_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{video_id}_{file.filename}")

    # 保存文件
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    return {"video_id": video_id, "filename": file.filename, "saved_path": file_path}


@app.post("/detect_objects/")
async def detect_objects(file: UploadFile = File(...)):
    """对上传的图像进行对象检测"""
    # 读取图像
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="无法解码图像")

    # 如果YOLOv12可用，使用实际模型进行检测
    if YOLOV12_AVAILABLE:
        try:
            # 加载模型
            model = YOLO("yolov12n.pt")
            
            # 进行预测
            results = model.predict(source=img, imgsz=640, conf=0.25)
            
            # 提取检测结果
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        coords = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # 获取类别名称
                        if hasattr(result, 'names') and class_id < len(result.names):
                            class_name = result.names[class_id]
                        else:
                            class_name = f"class_{class_id}"
                        
                        detections.append(DetectionResult(class_name, float(confidence), [float(x) for x in coords]))
            
            # 转换为可序列化的格式
            result = [
                {
                    "class_name": det.class_name,
                    "confidence": det.confidence,
                    "bbox": det.bbox
                }
                for det in detections
            ]
            
            return {"detections": result}
        except Exception as e:
            print(f"检测出错: {e}")
            # 出错时回退到模拟结果
    
    # 模拟检测结果
    detections = [
        DetectionResult("person", 0.95, [100, 120, 80, 200]),
        DetectionResult("car", 0.87, [300, 150, 120, 80])
    ]

    # 转换为可序列化的格式
    result = [
        {
            "class_name": det.class_name,
            "confidence": det.confidence,
            "bbox": det.bbox
        }
        for det in detections
    ]

    return {"detections": result}


@app.post("/track_objects/")
async def track_objects(video_id: str, background_tasks: BackgroundTasks):
    """对视频进行对象跟踪"""
    # 检查视频文件是否存在
    video_files = [f for f in os.listdir(UPLOAD_DIR) if f.startswith(video_id)]
    if not video_files:
        raise HTTPException(status_code=404, detail="视频文件未找到")

    video_path = os.path.join(UPLOAD_DIR, video_files[0])

    # 添加后台任务处理视频
    task_id = str(uuid.uuid4())
    processing_tasks[task_id] = {"status": "processing", "progress": 0}

    background_tasks.add_task(process_video_tracking, video_path, task_id)

    return {"task_id": task_id, "message": "视频处理已启动"}


async def process_video_tracking(video_path: str, task_id: str):
    """后台处理视频跟踪"""
    try:
        # 模拟视频处理过程
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # 模拟处理时间
            await asyncio.sleep(0.01)

            # 更新进度
            progress = int((i + 1) / total_frames * 100)
            processing_tasks[task_id]["progress"] = progress

        cap.release()

        # 标记为完成
        processing_tasks[task_id]["status"] = "completed"
        processing_tasks[task_id]["result"] = {
            "total_frames": total_frames,
            "objects_tracked": ["person", "car", "bike"]
        }
    except Exception as e:
        processing_tasks[task_id]["status"] = "failed"
        processing_tasks[task_id]["error"] = str(e)


@app.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
    """获取任务状态"""
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="任务未找到")

    return processing_tasks[task_id]


@app.get("/video_feed/{video_id}")
async def video_feed(video_id: str):
    """视频流接口"""
    # 查找视频文件
    video_files = [f for f in os.listdir(UPLOAD_DIR) if f.startswith(video_id)]
    if not video_files:
        raise HTTPException(status_code=404, detail="视频文件未找到")

    video_path = os.path.join(UPLOAD_DIR, video_files[0])

    def generate_frames():
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 这里可以添加实时处理逻辑
            # 例如: 对帧进行对象检测或跟踪

            # 编码帧
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        cap.release()

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/analytics/{video_id}")
async def get_video_analytics(video_id: str):
    """获取视频分析结果"""
    # 模拟分析数据
    analytics_data = {
        "video_id": video_id,
        "total_objects_detected": 127,
        "object_types": {
            "person": 45,
            "car": 32,
            "bike": 18,
            "bus": 5
        },
        "peak_activity_time": "00:05:23",
        "average_objects_per_frame": 3.2
    }

    return analytics_data


@app.post("/yolov12/detect/")
async def yolov12_detect(file: UploadFile = File(...), model_id: str = "yolov12n.pt"):
    """使用YOLOv12对上传的图像进行对象检测"""
    if not YOLOV12_AVAILABLE:
        raise HTTPException(status_code=501, detail="YOLOv12功能不可用")
    
    # 读取图像
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="无法解码图像")

    try:
        # 加载模型
        model = YOLO(model_id)
        
        # 进行预测
        results = model.predict(source=img, imgsz=640, conf=0.25)
        
        # 提取检测结果
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    coords = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # 获取类别名称
                    if hasattr(result, 'names') and class_id < len(result.names):
                        class_name = result.names[class_id]
                    else:
                        class_name = f"class_{class_id}"
                    
                    detections.append({
                        "class_name": class_name,
                        "confidence": float(confidence),
                        "bbox": [float(x) for x in coords]
                    })
        
        return {"detections": detections}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
