# 计算机视觉API服务

本项目提供基于YOLOv12和ByteTrack的计算机视觉功能的RESTful API接口。

## 功能特性

1. **对象检测** - 使用YOLOv12进行实时对象检测
2. **视频处理** - 对视频进行对象检测和跟踪
3. **徘徊检测** - 检测视频中人员的异常徘徊行为
4. **离岗/聚集检测** - 检测特定区域的人员离岗或聚集情况

## API端点

### 基础端点

- `GET /` - API根路径，返回欢迎信息

### 文件上传

- `POST /upload/` - 上传图像或视频文件
  - 参数：`file` (multipart/form-data)
  - 返回：文件ID和保存路径

### 对象检测

- `POST /detect/` - 对上传的图像进行对象检测
  - 参数：`file` (multipart/form-data)
  - 返回：检测到的对象列表（类别、置信度、边界框）

- `POST /yolov12/detect/` - 使用YOLOv12进行高级对象检测
  - 参数：
    - `file` (multipart/form-data)
    - `model_id` (可选，默认为"yolov12n.pt")
    - `image_size` (可选，默认为640)
    - `conf_threshold` (可选，默认为0.25)
  - 返回：检测到的对象列表

### 视频处理

- `POST /process_video/` - 处理视频文件（包括徘徊检测）
  - 参数：
    - `file_id` (查询参数)
    - `detect_loitering` (布尔值，是否检测徘徊行为)
    - `loitering_time_threshold` (整数，徘徊时间阈值，单位秒)
  - 返回：任务ID

- `POST /yolov12/process_video/` - 使用YOLOv12处理视频
  - 参数：
    - `file_id` (查询参数)
    - `model_id` (可选，默认为"yolov12n.pt")
    - `image_size` (可选，默认为640)
    - `conf_threshold` (可选，默认为0.25)
  - 返回：任务ID

### 任务管理

- `GET /task_status/{task_id}` - 获取任务状态
  - 返回：任务状态（处理中/已完成/失败）和进度

- `GET /download_processed/{task_id}` - 下载处理后的视频
  - 返回：处理后的视频文件

## 离岗/聚集检测API

项目中的[yolov12/whole.py](file:///E:/PyCharmevent/aivisual/yolov12/whole.py)文件提供了完整的离岗和聚集检测功能，可通过以下方式运行：

```bash
python yolov12/whole.py
```

该脚本会同时监控两个区域：
1. 离岗检测区（默认为画面左侧区域）
2. 聚集检测区（默认为画面右侧区域）

当检测到异常行为时，会自动保存警报截图。

## 安装和运行

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行API服务器：
```bash
python cv_api.py
```

或者运行原有的API服务器：
```bash
python api_server.py
```

3. 服务器将在`http://localhost:8000`启动

## 客户端使用示例

查看[api_client_example.py](file:///E:/PyCharmevent/aivisual/api_client_example.py)和[cv_api_client.py](file:///E:/PyCharmevent/aivisual/cv_api_client.py)了解如何使用API。

## 项目结构

```
.
├── api_server.py          # 原始API服务器
├── cv_api.py              # 新的计算机视觉API服务器
├── api_client_example.py  # 原始API客户端示例
├── cv_api_client.py       # 新的API客户端示例
├── detector/              # 检测器模块
│   ├── core.py            # 核心检测逻辑
│   ├── video_source.py    # 视频源管理
│   └── visualizer.py      # 可视化模块
├── yolov12/               # YOLOv12模块
│   ├── whole.py           # 离岗/聚集检测主程序
│   ├── main.py            # 人员聚集检测
│   ├── leave.py           # 离岗检测
│   └── app.py             # Gradio应用
└── uploads/               # 上传文件目录
```