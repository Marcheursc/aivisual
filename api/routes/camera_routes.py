"""
摄像头相关路由
处理实时摄像头流和摄像头管理功能
"""

from fastapi import APIRouter, HTTPException, Query, Form
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List, Optional
from ..services.camera_service import CameraService

router = APIRouter()

# 初始化摄像头服务
camera_service = CameraService()

@router.get("/cameras")
async def get_all_cameras():
    """
    获取所有摄像头列表
    """
    try:
        cameras = camera_service.get_all_cameras()
        return JSONResponse(content=cameras)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cameras/scene/{camera_id}")
async def get_camera_scene(camera_id: str):
    """
    获取指定摄像头的场景类型
    - camera_id: 摄像头ID
    """
    try:
        result = camera_service.get_camera_scene(camera_id)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cameras/assign_scene")
async def assign_camera_to_scene(camera_id: str, scene_type: str):
    """
    将摄像头分配到指定场景
    - camera_id: 摄像头ID
    - scene_type: 场景类型 (loitering, leave, gather)
    """
    try:
        result = camera_service.assign_camera_to_scene(camera_id, scene_type)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cameras/bind_device")
async def bind_camera_to_device(camera_id: str = Form(...), device_source: str = Form(...)):
    """
    将摄像头绑定到设备源
    - camera_id: 摄像头ID
    - device_source: 设备源 (可以是数字如0,1,2或RTSP地址如rtsp://192.168.1.100:554/stream1)
    """
    try:
        result = camera_service.bind_camera_to_device(camera_id, device_source)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cameras/unbind_device/{camera_id}")
async def unbind_camera_device(camera_id: str):
    """
    解除摄像头与设备的绑定
    - camera_id: 摄像头ID
    """
    try:
        result = camera_service.unbind_camera_device(camera_id)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cameras/device/{camera_id}")
async def get_camera_device(camera_id: str):
    """
    获取摄像头绑定的设备源
    - camera_id: 摄像头ID
    """
    try:
        result = camera_service.get_camera_device(camera_id)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cameras")
async def add_camera(camera_id: str = Form(default=""), name: str = Form(default=""), location: str = Form(default="")):
    """
    添加新的摄像头
    - camera_id: 摄像头ID
    - name: 摄像头名称
    - location: 摄像头位置
    """
    try:
        result = camera_service.add_camera(camera_id, name, location)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cameras/{camera_id}")
async def remove_camera(camera_id: str):
    """
    删除摄像头
    - camera_id: 摄像头ID
    """
    try:
        result = camera_service.remove_camera(camera_id)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cameras/process_camera/")
async def process_camera(
        camera_id: str = "default",
        detection_type: str = "loitering",
        loitering_time_threshold: int = 20,
        leave_roi: Optional[str] = None,
        leave_threshold: Optional[int] = None,
        gather_roi: Optional[str] = None,
        gather_threshold: Optional[int] = None,
        banner_roi: Optional[str] = None,
        banner_conf_threshold: Optional[float] = None,
        banner_iou_threshold: Optional[float] = None
):
    """
    实时处理摄像头视频流
    """
    # 检查摄像头是否已分配场景
    try:
        camera_scene = camera_service.get_camera_scene(camera_id)
        expected_scene = camera_scene["scene_type"]
        if detection_type != expected_scene:
            # 可以选择是否允许用户覆盖默认场景分配
            pass  # 这里我们允许用户指定不同的检测类型
    except:
        # 摄像头未分配场景，使用默认处理
        pass

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
        return StreamingResponse(
            camera_service.process_leave_stream(camera_id, parsed_leave_roi, leave_threshold),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    elif detection_type == "gather":
        # 聚集检测
        # 如果没有提供ROI，则使用默认值
        if not parsed_gather_roi:
            parsed_gather_roi = [(220, 300), (700, 300), (700, 700), (200, 700)]

        return StreamingResponse(
            camera_service.process_gather_stream(camera_id, parsed_gather_roi, gather_threshold),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    elif detection_type == "banner":
        # 横幅检测
        return StreamingResponse(
            camera_service.process_banner_stream(camera_id, parsed_banner_roi, banner_conf_threshold, banner_iou_threshold),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    else:
        # 默认为徘徊检测
        return StreamingResponse(
            camera_service.process_loitering_stream(camera_id, loitering_time_threshold),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )


def get_camera_source(camera_id: str):
    """
    获取摄像头的视频源
    """
    try:
        return camera_service.get_camera_source(camera_id)
    except:
        # 默认使用系统摄像头0
        return 0
