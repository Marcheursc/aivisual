#!/usr/bin/env python3
"""
安装项目依赖的脚本
"""

import subprocess
import sys
import os

def install_requirements():
    """安装requirements.txt中的依赖"""
    print("正在安装项目依赖...")

    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(current_dir, "requirements.txt")

    try:
        # 尝试使用国内镜像源安装
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", requirements_path,
            "-i", "https://pypi.tuna.tsinghua.edu.cn/simple/"
        ])
        print("依赖安装成功！")
        return True
    except subprocess.CalledProcessError:
        try:
            # 如果镜像源失败，尝试使用默认源
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", requirements_path
            ])
            print("依赖安装成功！")
            return True
        except subprocess.CalledProcessError as e:
            print(f"依赖安装失败: {e}")
            return False

def install_pytorch_cuda():
    """安装PyTorch 2.8.0 + CUDA 12.8"""
    print("正在安装 PyTorch 2.8.0 + CUDA 12.8...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch==2.8.0", "torchvision==0.13.0", "torchaudio==2.0.0",
            "--index-url", "https://download.pytorch.org/whl/cu128"
        ])
        print("PyTorch 2.8.0 + CUDA 12.8 安装成功！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"PyTorch 安装失败: {e}")
        return False

def install_lap():
    """安装lap库"""
    print("正在安装lap库...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "lap>=0.5.12"
        ])
        print("lap库安装成功！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"lap库安装失败: {e}")
        return False

def check_pytorch_cuda():
    """检查PyTorch CUDA支持"""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA available: 是")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            print(f"Current GPU: {torch.cuda.get_device_name()}")
            # 检查CUDA版本是否匹配
            if "12.8" in str(torch.version.cuda):
                print("CUDA 12.8 版本匹配")
            else:
                print(f"警告: CUDA版本 {torch.version.cuda} 与要求的 12.8 不匹配")
        else:
            print("CUDA available: 否")
        return True
    except ImportError:
        print("PyTorch 未安装")
        return False

def check_flash_attn():
    """检查flash_attn是否安装"""
    try:
        import flash_attn
        print("flash_attn 已安装")
        return True
    except ImportError:
        print("注意: flash_attn 未安装，可能需要手动安装以获得最佳性能")
        print("请参考 YOLOv12 文档进行安装")
        return False

def main():
    print("YOLOv12 项目依赖安装脚本")
    print("=" * 40)

    # 安装PyTorch
    if install_pytorch_cuda():
        # 安装lap库
        if install_lap():
            # 安装其他依赖
            if install_requirements():
                # 检查关键依赖
                check_pytorch_cuda()
                check_flash_attn()

                print("\n所有依赖处理完成！")
                print("现在可以运行项目了:")
                print("  python main.py")
            else:
                print("\n依赖安装失败，请手动检查并安装依赖")
                sys.exit(1)
        else:
            print("\nlap库安装失败，请手动安装")
            sys.exit(1)
    else:
        print("\nPyTorch 安装失败，请手动安装")
        sys.exit(1)

if __name__ == "__main__":
    main()
