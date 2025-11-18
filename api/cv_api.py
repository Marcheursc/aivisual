from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import sys

# 添加项目路径
project_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(project_path)
sys.path.insert(0, parent_path)
sys.path.insert(0, project_path)

# 导入路由
from api.routes.file_routes import router as file_router
from api.routes.task_routes import router as task_router
from api.routes.camera_routes import router as camera_router

# 初始化 FastAPI 应用
app = FastAPI(title="计算机视觉API",
              description="基于YOLOv12和ByteTrack的计算机视觉服务，提供人员检测、跟踪和异常行为检测功能")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该指定具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(file_router)
app.include_router(task_router)
app.include_router(camera_router)

@app.get("/")
async def root():
    return {"message": "欢迎使用计算机视觉API", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
