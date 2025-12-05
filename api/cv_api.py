from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 只导入需要的路由
from .routes.camera_routes import router as camera_router
from .routes.alarm_routes import router as alarm_router
from .utils.ascii import ascii_art as draw

# 初始化 FastAPI 应用
app = FastAPI(title="检测引擎API")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该指定具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 只注册实时流分析相关的路由
app.include_router(camera_router)
app.include_router(alarm_router)


@app.get("/")
async def root():
    return {"message": "欢迎使用计算机视觉API", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    print(draw)
    uvicorn.run(app, host="0.0.0.0", port=8000)