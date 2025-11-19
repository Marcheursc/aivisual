"""
文件处理相关路由
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import Optional
import os
import uuid
from api.config.settings import UPLOAD_DIR, PROCESSED_DIR
from api.algorithms.video_processor import VideoProcessor

router = APIRouter()

processing_tasks = {}

@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """上传文件（图像或视频）"""
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

    # 保存文件
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    return {"file_id": file_id, "filename": file.filename, "saved_path": file_path}


@router.post("/process_video/")
async def process_video(
        file_id: str,
        background_tasks: BackgroundTasks,
        detect_loitering: bool = True,
        loitering_time_threshold: int = 20,
        detection_type: str = "loitering",
        camera_id: str = "default",
        # 离岗和聚集检测的额外参数
        leave_roi: Optional[str] = None,
        leave_threshold: Optional[int] = None,
        gather_roi: Optional[str] = None,
        gather_threshold: Optional[int] = None,
        # 横幅检测的额外参数
        banner_roi: Optional[str] = None,
        banner_conf_threshold: Optional[float] = None,
        banner_iou_threshold: Optional[float] = None
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
        "result_path": None,
        "camera_id": camera_id,
        "detection_type": detection_type
    }

    # 解析ROI参数
    parsed_leave_roi = None
    parsed_gather_roi = None
    parsed_banner_roi = None

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

    if banner_roi:
        try:
            # 解析格式如: "[(0,0),(1280,0),(1280,720),(0,720)]"
            parsed_banner_roi = eval(banner_roi)
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
            leave_threshold,
            camera_id
        )
    elif detection_type == "gather":
        # 聚集检测
        background_tasks.add_task(
            process_gather_detection_task,
            file_path,
            task_id,
            parsed_gather_roi,
            gather_threshold,
            camera_id
        )
    elif detection_type == "banner":
        # 横幅检测
        background_tasks.add_task(
            process_banner_detection_task,
            file_path,
            task_id,
            parsed_banner_roi,
            banner_conf_threshold,
            banner_iou_threshold,
            camera_id
        )
    else:
        # 默认为徘徊检测
        background_tasks.add_task(
            process_video_task,
            file_path,
            task_id,
            detect_loitering,
            loitering_time_threshold,
            camera_id
        )

    return {"task_id": task_id, "message": f"{detection_type}视频处理已启动"}


async def process_video_task(
        video_path: str,
        task_id: str,
        detect_loitering: bool = True,
        loitering_time_threshold: int = 20,
        camera_id: str = "default"
):
    """后台处理视频任务"""
    try:
        # 初始化视频处理器
        processor = VideoProcessor()

        # 设置输出视频路径
        output_filename = f"processed_{uuid.uuid4()}.mp4"
        output_path = os.path.join(PROCESSED_DIR, output_filename)

        # 处理视频
        result_path = processor.process_loitering_video(
            video_path=video_path,
            output_path=output_path,
            loitering_time_threshold=loitering_time_threshold
        )

        # 标记为完成
        processing_tasks[task_id]["status"] = "completed"
        processing_tasks[task_id]["result_path"] = result_path
        processing_tasks[task_id]["camera_id"] = camera_id

        # 保存报警信息（示例）
        # 在实际应用中，这里会根据检测结果生成报警信息并保存到数据库

    except Exception as e:
        processing_tasks[task_id]["status"] = "failed"
        processing_tasks[task_id]["error"] = str(e)


async def process_leave_detection_task(
        video_path: str,
        task_id: str,
        roi: Optional[list] = None,
        threshold: Optional[int] = None,
        camera_id: str = "default"
):
    """离岗检测处理任务"""
    try:
        # 初始化视频处理器
        processor = VideoProcessor()

        # 设置输出视频路径
        output_filename = f"leave_processed_{uuid.uuid4()}.mp4"
        output_path = os.path.join(PROCESSED_DIR, output_filename)

        # 处理视频
        result_path = processor.process_leave_video(
            video_path=video_path,
            output_path=output_path,
            roi=roi,
            absence_threshold=threshold if threshold is not None else 5
        )

        # 标记为完成
        processing_tasks[task_id]["status"] = "completed"
        processing_tasks[task_id]["result_path"] = result_path
        processing_tasks[task_id]["camera_id"] = camera_id

    except Exception as e:
        processing_tasks[task_id]["status"] = "failed"
        processing_tasks[task_id]["error"] = str(e)


async def process_gather_detection_task(
        video_path: str,
        task_id: str,
        roi: Optional[list] = None,
        threshold: Optional[int] = None,
        camera_id: str = "default"
):
    """聚集检测处理任务"""
    try:
        # 初始化视频处理器
        processor = VideoProcessor()

        # 设置输出视频路径
        output_filename = f"gather_processed_{uuid.uuid4()}.mp4"
        output_path = os.path.join(PROCESSED_DIR, output_filename)

        # 处理视频
        result_path = processor.process_gather_video(
            video_path=video_path,
            output_path=output_path,
            roi=roi,
            gather_threshold=threshold if threshold is not None else 5
        )

        # 标记为完成
        processing_tasks[task_id]["status"] = "completed"
        processing_tasks[task_id]["result_path"] = result_path
        processing_tasks[task_id]["camera_id"] = camera_id

    except Exception as e:
        processing_tasks[task_id]["status"] = "failed"
        processing_tasks[task_id]["error"] = str(e)


async def process_banner_detection_task(
        video_path: str,
        task_id: str,
        roi: Optional[list] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        camera_id: str = "default"
):
    """横幅检测处理任务"""
    try:
        # 初始化视频处理器
        processor = VideoProcessor()

        # 设置输出视频路径
        output_filename = f"banner_processed_{uuid.uuid4()}.mp4"
        output_path = os.path.join(PROCESSED_DIR, output_filename)

        # 处理视频
        result_path = processor.process_banner_video(
            video_path=video_path,
            output_path=output_path,
            roi=roi,
            conf_threshold=conf_threshold if conf_threshold is not None else 0.5,
            iou_threshold=iou_threshold if iou_threshold is not None else 0.45
        )

        # 标记为完成
        processing_tasks[task_id]["status"] = "completed"
        processing_tasks[task_id]["result_path"] = result_path
        processing_tasks[task_id]["camera_id"] = camera_id

    except Exception as e:
        processing_tasks[task_id]["status"] = "failed"
        processing_tasks[task_id]["error"] = str(e)
