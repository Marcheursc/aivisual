#!/usr/bin/env python3
"""
检查项目依赖是否正确安装的脚本
"""
def check_dependencies():
    """检查项目依赖"""
    dependencies = {
        "torch": "PyTorch深度学习框架",
        "torchvision": "PyTorch计算机视觉库",
        "torchaudio": "PyTorch音频处理库",
        "cv2": "OpenCV计算机视觉库",
        "numpy": "NumPy科学计算库",
        "ultralytics": "Ultralytics YOLO实现",
    }

    optional_deps = {
        "flash_attn": "Flash Attention优化库（可选，用于提升性能）",
        "timm": "PyTorch图像模型库",
    }

    print("检查必需依赖...")
    all_good = True

    for dep, description in dependencies.items():
        try:
            if dep == "cv2":
                import cv2
                print(f"✓ {dep} (OpenCV): {cv2.__version__}")
            elif dep == "numpy":
                import numpy as np
                print(f"✓ {dep}: {np.__version__}")
            elif dep == "torch":
                import torch
                print(f"✓ {dep}: {torch.__version__}")
            elif dep == "torchvision":
                import torchvision
                print(f"✓ {dep}: {torchvision.__version__}")
            elif dep == "torchaudio":
                import torchaudio
                print(f"✓ {dep}: {torchaudio.__version__}")
            else:
                module = __import__(dep)
                if hasattr(module, "__version__"):
                    print(f"✓ {dep}: {module.__version__}")
                else:
                    print(f"✓ {dep}: 已安装")
        except ImportError:
            print(f"✗ {dep}: 未安装 - {description}")
            all_good = False

    print("\n检查可选依赖...")
    for dep, description in optional_deps.items():
        try:
            module = __import__(dep)
            if hasattr(module, "__version__"):
                print(f"✓ {dep}: {module.__version__} - {description}")
            else:
                print(f"✓ {dep}: 已安装 - {description}")
        except ImportError:
            print(f"⚠ {dep}: 未找到 - {description}")

    return all_good

def check_cuda():
    """检查CUDA支持"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\nCUDA 可用:")
            print(f"  CUDA 版本: {torch.version.cuda}")
            print(f"  GPU 数量: {torch.cuda.device_count()}")
            print(f"  当前 GPU: {torch.cuda.get_device_name()}")
            # 检查CUDA版本是否匹配要求
            if "12.8" in str(torch.version.cuda):
                print("  CUDA 12.8 版本匹配: ✓")
            else:
                print("  CUDA 版本警告: 与要求的 12.8 不匹配")
        else:
            print(f"\nCUDA 不可用，将使用CPU运行")
    except ImportError:
        print("\n无法检查CUDA支持，因为PyTorch未安装")

def check_model_files():
    """检查模型文件是否存在"""
    import os
    model_files = [
        ("yolov12/yolov12n.pt", "YOLOv12 Nano模型"),
        ("yolov12/yolov12s.pt", "YOLOv12 Small模型"),
        ("yolov12/yolov12m.pt", "YOLOv12 Medium模型"),
        ("yolov12/yolov12l.pt", "YOLOv12 Large模型"),
        ("yolov12/yolov12x.pt", "YOLOv12 XL模型"),
    ]

    print("\n检查模型文件...")
    for model_path, description in model_files:
        if os.path.exists(model_path):
            size = os.path.getsize(model_path) / (1024*1024)  # MB
            print(f"✓ {model_path}: {size:.1f} MB - {description}")
        else:
            print(f"⚠ {model_path}: 未找到 - {description}")

def main():
    print("YOLOv12 项目依赖检查")
    print("=" * 40)

    deps_ok = check_dependencies()
    check_cuda()
    check_model_files()

    print("\n" + "=" * 40)
    if deps_ok:
        print("✓ 所有必需依赖已正确安装")
        print("现在可以运行项目了:")
        print("  python main.py")
    else:
        print("✗ 部分必需依赖未安装")
        print("请运行以下命令安装依赖:")
        print("  pip install -r requirements.txt")
        print("或")
        print("  python install_deps.py")

if __name__ == "__main__":
    main()
