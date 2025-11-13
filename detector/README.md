# 徘徊检测模块

这个目录包含了所有与徘徊检测相关的文件。

## 文件说明

### 1. core.py
- 核心徘徊检测实现
- 使用 YOLOv12 进行目标检测
- 使用 ByteTrack 进行目标跟踪
- 通过分析目标在特定区域内的移动距离和时间来判断是否徘徊
- 移动距离阈值：50像素
- 支持配置检测区域、时间阈值等参数

### 2. video_source.py
- 视频源管理器
- 支持从视频文件或摄像头读取视频流
- 提供统一的视频属性获取接口

### 3. visualizer.py
- 可视化模块
- 在视频帧上绘制检测框、跟踪ID和警报信息
- 显示FPS和处理时间信息
- 支持绘制检测区域

## 使用方法

徘徊检测功能主要通过 [main.py](file:///E:/PyCharmevent/aivisual/main.py) 文件调用，但也可以集成到其他应用中。

### 核心类使用示例

```python
from detector.core import LoiteringDetector

# 初始化检测器
detector = LoiteringDetector(
    model_path="yolov12/yolov12n.pt",
    loitering_time_threshold=20,
    target_classes=["person"],
    conf_threshold=0.5
)

# 处理视频帧
detections, alarms = detector.detect_loitering(frame, frame_time)
```

## 配置参数

1. **model_path** - YOLOv12模型路径
2. **loitering_time_threshold** - 徘徊时间阈值（秒）
3. **target_classes** - 要检测的目标类别列表
4. **conf_threshold** - 置信度阈值
5. **img_size** - 图像处理尺寸
6. **device** - 运行设备 ('cuda' 或 'cpu')
7. **detection_region** - 检测区域 (x, y, width, height) 或 None 表示全图检测
8. **use_bytetrack** - 是否使用ByteTrack跟踪器

## 输出结果

1. **检测框** - 在视频帧上绘制目标边界框
2. **跟踪ID** - 显示每个目标的唯一ID
3. **警报信息** - 当检测到徘徊行为时显示警报
4. **处理统计** - 显示FPS、处理时间等信息