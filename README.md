# AI-Visual Detection System

## 简介

这是一个基于YOLOv12的AI视频智能检测系统，具备目标检测、跟踪和多种行为分析功能，采用React + FastAPI的现代化前后端分离架构。

## 功能特性

- 基于YOLOv12的目标检测
- 多目标跟踪（ByteTrack）
- 徘徊行为检测与报警
- 离岗检测与报警
- 人员聚集检测与报警
- 支持多种目标类别检测（人、车等）
- 可视化显示检测结果和报警信息
- 前后端分离架构，现代化用户界面

## 项目结构

```
project/
├── api/                      # FastAPI 后端服务
│   ├── algorithms/           # 核心算法模块
│   ├── config/              # 配置文件
│   ├── models/              # 模型管理
│   ├── routes/              # API路由
│   ├── core/                # 核心业务逻辑
│   ├── utils/               # 工具函数
│   ├── uploads/             # 上传文件目录
│   ├── processed_videos/    # 处理后视频目录
│   └── cv_api.py            # FastAPI 主服务
├── frontend/                # React 前端应用
│   ├── public/              # 静态资源
│   ├── src/                 # 源代码
│   │   ├── components/      # 公共组件（可复用）
│   │   ├── pages/           # 页面组件
│   │   │   ├── Home.js      # 首页
│   │   │   ├── Upload/      # 上传页面
│   │   │   │   ├── index.js
│   │   │   │   └── VideoUpload.js
│   │   │   ├── Detect/      # 检测页面
│   │   │   │   ├── index.js
│   │   │   │   └── DetectionControl.js
│   │   │   ├── Status/      # 状态页面
│   │   │   │   ├── index.js
│   │   │   │   └── TaskStatus.js
│   │   ├── routes/          # 前端路由配置
│   │   │   └── index.js     # 路由定义
│   │   ├── App.js           # 主应用组件
│   │   ├── App.css          # 样式
│   │   └── index.js         # 入口文件
│   └── package.json         # 前端依赖配置
├── yolov12/                 # YOLOv12 模型文件
├── Dockerfile.backend       # 后端 Docker 配置
├── Dockerfile.frontend      # 前端 Docker 配置
├── docker-compose.yml       # Docker 容器编排配置
└── requirements.txt         # Python 依赖
```

## 安装依赖

```bash
# 安装前端依赖
cd frontend
npm install

# 返回项目根目录
cd ..

# 安装Python依赖
pip install -r requirements.txt

# 安装lap库（ByteTrack需要）
pip install lap>=0.5.12

# 安装PyTorch相关库（根据你的环境选择合适的版本）
# CUDA 12.1版本
pip install torch==2.8.0+cu121 torchvision==0.13.0+cu121 torchaudio==2.0.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# CPU版本
pip install torch==2.8.0+cpu torchvision==0.13.0+cpu torchaudio==2.0.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
```

## 运行系统

系统采用前后端分离架构，需要分别启动前端和后端服务：

```bash
# 启动后端API服务
python api/cv_api.py

# 在另一个终端启动前端应用
cd frontend
npm start
```

或者使用Docker容器化部署：

```bash
# 使用docker-compose一键部署
docker-compose up --build
```

访问地址：
- 前端界面: http://localhost:3000
- 后端API文档: http://localhost:8000/docs

## 部署说明

本项目采用 React + FastAPI 的前后端分离架构，并支持 Docker 容器化部署。

### 本地开发部署

#### 环境要求

- Python 3.8+
- Node.js 14+
- Docker & Docker Compose (可选，用于容器化部署)

#### 前端依赖安装

```bash
cd frontend
npm install
```

#### 后端依赖安装

```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 安装lap库（ByteTrack需要）
pip install lap>=0.5.12

# 安装PyTorch相关库（根据你的环境选择合适的版本）
# CUDA 12.1版本
pip install torch==2.8.0+cu121 torchvision==0.13.0+cu121 torchaudio==2.0.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# CPU版本
pip install torch==2.8.0+cpu torchvision==0.13.0+cpu torchaudio==2.0.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
```

#### 后端服务启动

```bash
# 启动 FastAPI 服务
uvicorn api.cv_api:app --host 0.0.0.0 --port 8000

# 或者使用 Python 直接运行
python api/cv_api.py
```

访问 `http://localhost:8000` 查看 API 文档

#### 前端应用启动

```bash
cd frontend
npm start
```

访问 `http://localhost:3000` 使用前端应用

### Docker 容器化部署

#### 构建和运行

使用 docker-compose 一键部署：

```bash
docker-compose up --build
```

#### 访问应用

- 前端界面: http://localhost:3000
- 后端 API: http://localhost:8000

#### 分别构建和运行

分别构建前后端镜像：

```bash
# 构建后端镜像
docker build -t cv-backend -f Dockerfile.backend .

# 构建前端镜像
docker build -t cv-frontend -f Dockerfile.frontend .
```

分别运行容器：

```bash
# 运行后端服务
docker run -d -p 8000:8000 --name backend cv-backend

# 运行前端应用
docker run -d -p 3000:3000 --name frontend cv-frontend
```

## Git工作流规范

本项目采用标准的Git工作流进行版本控制：

### 分支策略

1. **main分支** - 生产环境稳定版本
2. **develop分支** - 开发环境最新版本
3. **feature分支** - 功能开发分支，命名规范：`feature/功能名称`
4. **hotfix分支** - 紧急修复分支，命名规范：`hotfix/问题描述`
5. **release分支** - 发布准备分支，命名规范：`release/版本号`

### 提交规范

提交信息遵循以下格式：
```
<type>(<scope>): <subject>

<body>

<footer>
```

常用type类型：
- feat: 新功能
- fix: 修复bug
- docs: 文档更新
- style: 代码格式调整
- refactor: 代码重构
- test: 测试相关
- chore: 构建过程或辅助工具的变动

示例：
```
feat(api): 添加离岗检测功能

实现离岗检测核心算法，支持ROI区域设置和时间阈值配置

Closes #123
```

### 工作流程

1. 从develop分支创建feature分支
```bash
git checkout develop
git pull origin develop
git checkout -b feature/新功能名称
```

2. 开发并提交代码
```bash
git add .
git commit -m "feat: 实现新功能"
```

3. 推送分支并创建Pull Request
```bash
git push origin feature/新功能名称
```

4. 代码审查通过后合并到develop分支

## 前端路由

前端采用React Router进行路由管理，包含以下页面：

1. **首页** - `/` 
   - 系统介绍和导航入口

2. **视频上传** - `/upload`
   - 视频文件选择和上传功能

3. **行为检测** - `/detect`
   - 检测参数设置和任务启动

4. **任务状态** - `/status`
   - 任务进度查看和结果下载

路由配置文件位于 `frontend/src/routes/index.js`，与主应用组件分离以提高可维护性。

## 后端路由

API路由已按功能模块分离到独立文件中：

1. **文件处理路由** - 位于 `api/routes/file_routes.py`
   - `POST /upload/` - 上传文件
   - `POST /process_video/` - 处理视频

2. **任务管理路由** - 位于 `api/routes/task_routes.py`
   - `GET /task_status/{task_id}` - 获取任务状态
   - `GET /download_processed/{task_id}` - 下载处理后的视频

## 模块说明

系统功能已按用途分类组织：

1. **核心算法模块** - 位于 `api/algorithms/` 目录
   - 徘徊检测: `loitering_detection.py`
   - 离岗检测: `leave_detection.py`
   - 聚集检测: `gather_detection.py`
   - 视频处理: `video_processor.py`

2. **模型管理模块** - 位于 `api/models/` 目录
   - YOLO模型管理: `yolo_models.py`

3. **配置管理模块** - 位于 `api/config/` 目录
   - 系统配置: `settings.py`

## 配置说明

### CORS 配置

后端已在 FastAPI 中配置 CORS 支持，允许所有来源访问。在生产环境中，建议修改 `api/cv_api.py` 中的 CORS 配置，指定具体的域名：

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # 指定具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 环境变量

前端通过环境变量配置后端 API 地址，在 Docker 环境中已配置为 `http://localhost:8000`。在本地开发环境中，可以通过修改前端代码中的 API 地址来配置。

## 核心算法模块说明

所有核心算法已整合到 `api/algorithms` 目录中：

- `loitering_detection.py`: 徘徊检测算法，基于 YOLOv12 和 ByteTrack
- `leave_detection.py`: 离岗检测算法
- `gather_detection.py`: 聚集检测算法
- `video_processor.py`: 统一视频处理接口，整合所有检测功能

所有算法和模型都在后端运行，前端只负责用户界面和与后端 API 交互。

## 故障排除

1. **numpy兼容性问题 (AttributeError: module 'numpy' has no attribute 'object')**:
   - 系统已自动处理此问题，在detector/core.py和test.py中添加了兼容性代码
   - 无需额外操作

2. **ByteTrack跟踪器不可用**:
   - 确保已安装lap库: `pip install lap>=0.5.12`
   - 系统会自动检测并使用ByteTrack，如果不可用会回退到基础跟踪方法

3. **CUDA相关问题**:
   - 如果没有GPU或CUDA不可用，系统会自动回退到CPU运行
   - 确保安装了正确版本的PyTorch和CUDA驱动

## 应用场景

- 安全监控
- 入侵检测
- 公共空间监控
- 智慧城市应用