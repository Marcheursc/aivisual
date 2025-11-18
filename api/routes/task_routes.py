"""
任务管理相关路由
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List, Optional
import os
import json
from datetime import datetime
from api.routes.file_routes import processing_tasks
from api.config.settings import PROCESSED_DIR

router = APIRouter()

# 报警信息存储（实际项目中应该使用数据库）
alerts_data = []

@router.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
    """获取任务状态"""
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="任务未找到")

    return processing_tasks[task_id]


@router.get("/download_processed/{task_id}")
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


@router.get("/alerts")
async def query_alerts(
    camera_ids: List[str] = Query(...), 
    scene_type: str = Query(...),
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
):
    """
    查询报警数据
    - camera_ids: 摄像头ID列表
    - scene_type: 场景类型
    - start_time: 开始时间 (格式: YYYY-MM-DD HH:MM:SS)
    - end_time: 结束时间 (格式: YYYY-MM-DD HH:MM:SS)
    """
    # 过滤报警数据
    filtered_alerts = []
    for alert in alerts_data:
        # 检查摄像头ID
        if alert["camera_id"] not in camera_ids:
            continue
            
        # 检查场景类型
        if alert["scene_type"] != scene_type:
            continue
            
        # 检查时间范围
        if start_time and alert["timestamp"] < start_time:
            continue
            
        if end_time and alert["timestamp"] > end_time:
            continue
            
        filtered_alerts.append(alert)
    
    return JSONResponse(content=filtered_alerts)


@router.get("/alerts/stream")
async def stream_alerts():
    """
    实时报警推送接口
    返回服务器发送事件(SSE)流
    """
    # 这里只是一个示例实现，实际项目中应该使用WebSocket或其他实时通信技术
    def generate():
        # 模拟实时报警推送
        yield f"data: {json.dumps({'message': '连接已建立，等待报警信息...'})}\n\n"
        
        # 在实际应用中，这里会持续监听报警事件并推送
        # 例如通过消息队列或事件总线
    
    return StreamingResponse(generate(), media_type="text/event-stream")