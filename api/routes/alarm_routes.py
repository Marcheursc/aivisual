"""
警报API路由
用于接收告警信息并发送到RabbitMQ队列
"""

from fastapi import APIRouter, HTTPException, Body, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
from pydantic import BaseModel, Field
from ..services.rabbitmq_service import rabbitmq_producer
from ..utils.security import require_bearer_token


class Alarm(BaseModel):
    """告警信息模型"""
    code: Optional[str] = Field(None, description="记录唯一标识")
    alarmTime: str = Field(..., description="告警时间，格式：yyyy-MM-dd HH:mm:ss")
    deviceCode: Optional[str] = Field(None, description="设备唯一标识")
    deviceName: Optional[str] = Field(None, description="设备名称")
    level: Optional[str] = Field(None, description="告警级别")
    memo: Optional[str] = Field(None, description="告警内容")
    image: Optional[str] = Field(None, description="告警照片，url")
    position: Optional[str] = Field(None, description="位置")
    personCode: Optional[str] = Field(None, description="人员唯一标识")
    personName: Optional[str] = Field(None, description="人员姓名")
    ext1: Optional[str] = Field(None, description="扩展字段json")


class BatchAlarmRequest(BaseModel):
    """批量告警请求模型"""
    alarms: List[Alarm] = Field(..., description="告警信息列表")


router = APIRouter(dependencies=[Depends(require_bearer_token)])


@router.post("/send_alarm")
async def send_alarm(
        code: Optional[str] = Body(None, description="记录唯一标识，不提供则自动生成"),
        alarmTime: datetime = Body(..., description="告警时间，格式：yyyy-MM-dd HH:mm:ss"),
        deviceCode: Optional[str] = Body(None, description="设备唯一标识"),
        deviceName: Optional[str] = Body(None, description="设备名称"),
        level: Optional[str] = Body(None, description="告警级别"),
        memo: Optional[str] = Body(None, description="告警内容"),
        image: Optional[str] = Body(None, description="告警照片，url"),
        position: Optional[str] = Body(None, description="位置"),
        personCode: Optional[str] = Body(None, description="人员唯一标识"),
        personName: Optional[str] = Body(None, description="人员姓名"),
        ext1: Optional[str] = Body(None, description="扩展字段json")
):
    """
    发送告警信息到RabbitMQ队列

    参数说明：
    - code: 记录唯一标识，不提供则自动生成
    - alarmTime: 告警时间，格式：yyyy-MM-dd HH:mm:ss（必填）
    - deviceCode: 设备唯一标识
    - deviceName: 设备名称
    - level: 告警级别
    - memo: 告警内容
    - image: 告警照片，url
    - position: 位置
    - personCode: 人员唯一标识
    - personName: 人员姓名
    - ext1: 扩展字段json

    返回结果：
    - status: 状态，success或error
    - message: 提示信息
    - data: 发送的告警数据
    """
    try:
        # 生成唯一标识（如果未提供）
        if not code:
            code = str(uuid.uuid4())

        # 构建告警消息
        alarm_message = {
            "code": code,
            "alarmType": 1,  # 告警类型固定为1（摄像头告警）
            "subType": "异常行为识别",  # 告警子类型固定为"异常行为识别"
            "alarmTime": alarmTime.strftime("%Y-%m-%d %H:%M:%S"),  # 格式化时间
            "deviceCode": deviceCode,
            "deviceName": deviceName,
            "level": level,
            "memo": memo,
            "image": image,
            "position": position,
            "personCode": personCode,
            "personName": personName,
            "ext1": ext1
        }

        # 发送消息到RabbitMQ
        success = rabbitmq_producer.send_message(alarm_message)

        if success:
            return JSONResponse(content={
                "status": "success",
                "message": "告警信息已成功发送到消息队列",
                "data": alarm_message
            })
        else:
            raise HTTPException(
                status_code=500,
                detail="告警信息发送失败，请稍后重试"
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"发送告警信息时发生错误: {str(e)}"
        )


@router.post("/batch_send_alarm")
async def batch_send_alarm(
        request: BatchAlarmRequest = Body(..., description="批量告警请求")
):
    """
    批量发送告警信息到RabbitMQ队列

    参数说明：
    - alarms: 告警信息列表，每个告警信息的字段与send_alarm接口相同

    返回结果：
    - status: 状态，success或error
    - message: 提示信息
    - data: 发送结果统计
    """
    try:
        alarms = request.alarms
        if not alarms:
            raise HTTPException(
                status_code=400,
                detail="告警列表不能为空"
            )

        success_count = 0
        failed_count = 0
        failed_alarms = []

        for alarm in alarms:
            try:
                # 生成唯一标识（如果未提供）
                alarm_code = alarm.code or str(uuid.uuid4())

                # 构建完整的告警消息
                full_alarm = {
                    "code": alarm_code,
                    "alarmType": 1,  # 告警类型固定为1（摄像头告警）
                    "subType": "异常行为识别",  # 告警子类型固定为"异常行为识别"
                    "alarmTime": alarm.alarmTime,  # 直接使用字符串格式的时间
                    "deviceCode": alarm.deviceCode,
                    "deviceName": alarm.deviceName,
                    "level": alarm.level,
                    "memo": alarm.memo,
                    "image": alarm.image,
                    "position": alarm.position,
                    "personCode": alarm.personCode,
                    "personName": alarm.personName,
                    "ext1": alarm.ext1
                }

                # 发送消息到RabbitMQ
                if rabbitmq_producer.send_message(full_alarm):
                    success_count += 1
                else:
                    failed_count += 1
                    failed_alarms.append(alarm_code)

            except Exception as e:
                failed_count += 1
                failed_alarms.append(alarm.code or "unknown")

        return JSONResponse(content={
            "status": "success",
            "message": f"批量发送完成，成功{success_count}条，失败{failed_count}条",
            "data": {
                "total": len(alarms),
                "success_count": success_count,
                "failed_count": failed_count,
                "failed_alarms": failed_alarms
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"批量发送告警信息时发生错误: {str(e)}"
        )
