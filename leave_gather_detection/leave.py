from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
import os


def process_leave_detection(video_path, roi=None, threshold=None):
    """处理离岗检测视频任务"""
    import cv2
    import numpy as np
    import os
    from datetime import datetime
    from ultralytics import YOLO

    # 创建截图保存文件夹
    screenshot_dir = "alert_screenshots"
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir)

    # ROI区域和阈值设置
    ROI = roi if roi is not None else [(600, 100), (1000, 100), (1000, 700), (600, 700)]
    离岗阈值 = threshold if threshold is not None else 5  # 离岗判定时间(秒)

    # Linux中文字体配置（优化中文显示）
    def cv2_puttext(img, text, org, font_scale=1, color=(255, 0, 0), thickness=2):
        from PIL import Image, ImageDraw, ImageFont
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        # Linux系统中文字体路径
        font_paths = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
        ]
        font = None
        for path in font_paths:
            if os.path.exists(path):
                try:
                    font = ImageFont.truetype(path, int(font_scale * 20))
                    break
                except:
                    continue
        if not font:
            font = ImageFont.load_default()
        draw.text(org, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # 点是否在ROI内（射线法）
    def point_in_roi(point, roi):
        x, y = point
        n = len(roi)
        inside = False
        for i in range(n):
            j = (i + 1) % n
            xi, yi = roi[i]
            xj, yj = roi[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
        return inside

    # 读取视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"无法打开视频 {video_path}")

    # 获取视频参数
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 设置输出视频路径到api/processed_videos目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(project_root, "api", "processed_videos")
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    output_video_path = os.path.join(processed_dir, f"leave_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise Exception("无法创建输出视频文件")

    # 加载YOLOv12模型（使用yolov12目录下的模型文件）
    try:
        model_path = os.path.join(project_root, "yolov12", "yolov12n.pt")
        model = YOLO(model_path)
    except Exception as e:
        raise Exception(f"模型加载失败：{e}")

    # 离岗检测状态变量（初始状态设为脱岗）
    status = "脱岗"
    absence_start_time = datetime.now()
    alert_triggered = False

    current_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame += 1

        # 检测行人
        results = model(frame, classes=[0], verbose=False)
        person_boxes = []
        for box in results[0].boxes:
            cls = int(box.cls[0])
            if cls == 0:
                person_boxes.append(box.xyxy.cpu().numpy()[0])

        # 统计ROI内人数
        roi_person_count = 0
        for box in person_boxes:
            x1, y1, x2, y2 = box.astype(int)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            if point_in_roi(center, ROI):
                roi_person_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 更新状态检测逻辑
        current_time = datetime.now()
        if roi_person_count > 0:
            status = "在岗"
            absence_start_time = None
            alert_triggered = False
        else:
            status = "脱岗"
            if absence_start_time is None:
                absence_start_time = current_time

            absence_duration = (current_time - absence_start_time).total_seconds()

            if absence_duration >= 离岗阈值 and not alert_triggered:
                frame = cv2_puttext(frame, "⚠️ 警告：人员脱岗！", (width // 2 - 150, 50), 1.5, (0, 0, 255), 3)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = os.path.join(screenshot_dir, f"absence_alert_{timestamp}.jpg")
                cv2.imwrite(screenshot_path, frame)
                alert_triggered = True

        # 绘制ROI
        roi_np = np.array(ROI, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [roi_np], True, (0, 255, 0), 2)
        frame = cv2_puttext(frame, "监控区域", (ROI[0][0] + 10, ROI[0][1] - 15), 1, (0, 255, 0), 2)

        # 显示状态信息
        if status == "在岗":
            frame = cv2_puttext(frame, f"状态: {status}", (30, 50), 1, (0, 255, 0), 2)
        else:
            frame = cv2_puttext(frame, f"状态: {status}", (30, 50), 1, (0, 0, 255), 2)
            if absence_start_time:
                absence_duration = (current_time - absence_start_time).total_seconds()
                frame = cv2_puttext(frame, f"脱岗时长: {absence_duration:.1f}秒", (30, 100), 1, (0, 0, 255), 2)

        # 写入处理后的帧
        out.write(frame)

    # 释放资源
    cap.release()
    out.release()

    return output_video_path


# 仅在直接运行此脚本时执行以下代码
if __name__ == "__main__":
    # 1. 创建截图保存文件夹
    screenshot_dir = "alert_screenshots"
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir)
        print(f"已创建截图文件夹：{screenshot_dir}")

    # 2. Linux中文字体配置（优化中文显示）
    def cv2_puttext(img, text, org, font_scale=1, color=(255, 0, 0), thickness=2):
        from PIL import Image, ImageDraw, ImageFont
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        # Linux系统中文字体路径
        font_paths = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
        ]
        font = None
        for path in font_paths:
            if os.path.exists(path):
                try:
                    font = ImageFont.truetype(path, int(font_scale * 20))
                    break
                except:
                    continue
        if not font:
            font = ImageFont.load_default()
        draw.text(org, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # 3. 核心参数配置（修改ROI为右移的细长状态）
    video_path = "/media/zhang/data2/peopledetect/l2.mp4"
    # 新ROI：右移（x坐标整体增加），细长型（宽度窄、高度高）
    ROI = [(600, 100), (1000, 100), (1000, 700), (600, 700)]  # 右移至x=900-1000，高度200-700
    离岗阈值 = 5  # 离岗判定时间(秒)
    save_result = True
    output_video_path = "processed_video.mp4"

    # 4. 点是否在ROI内（射线法）
    def point_in_roi(point, roi):
        x, y = point
        n = len(roi)
        inside = False
        for i in range(n):
            j = (i + 1) % n
            xi, yi = roi[i]
            xj, yj = roi[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
        return inside

    # 5. 读取视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频 {video_path}")
        exit()

    # 获取视频参数
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 6. 初始化视频写入器
    out = None
    if save_result:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if not out.isOpened():
            print(f"警告：无法保存输出视频，已关闭保存功能")
            save_result = False

    # 7. 加载YOLOv12模型（使用yolov12目录下的模型文件）
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, "yolov12", "yolov12n.pt")
        model = YOLO(model_path)
        if model.names.get(0) != "person":
            print("警告：模型类别0不是'person'，可能无法正确检测行人")
        else:
            print("模型已确认：仅检测行人（类别0）")
    except Exception as e:
        print(f"模型加载失败：{e}")
        exit()

    # 8. 离岗检测状态变量（初始状态设为脱岗）
    status = "脱岗"  # 初始状态改为脱岗
    absence_start_time = datetime.now()  # 初始即开始计时无人时间
    alert_triggered = False  # 预警是否已触发

    # 9. 实时处理与显示
    print("开始处理，窗口将显示实时画面：")
    print("  - 按 'p' 暂停，按任意键继续")
    print("  - 按 'q' 退出程序")

    current_frame = 0
    cv2.namedWindow("离岗检测（实时）", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n视频处理完成！")
            break

        current_frame += 1
        if current_frame % 50 == 0:
            progress = int(current_frame / total_frames * 100)
            print(f"进度：{current_frame}/{total_frames} 帧 ({progress}%)", end="\r")

        # 检测行人
        results = model(frame, classes=[0], verbose=False)
        person_boxes = []
        for box in results[0].boxes:
            cls = int(box.cls[0])
            if cls == 0:
                person_boxes.append(box.xyxy.cpu().numpy()[0])

        # 统计ROI内人数
        roi_person_count = 0
        for box in person_boxes:
            x1, y1, x2, y2 = box.astype(int)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            if point_in_roi(center, ROI):
                roi_person_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框标记在岗人员

        # 更新状态检测逻辑
        current_time = datetime.now()
        if roi_person_count > 0:
            # 区域内有人，切换为在岗状态
            status = "在岗"
            absence_start_time = None
            alert_triggered = False
        else:
            # 区域内无人，维持/切换为脱岗状态
            status = "脱岗"
            if absence_start_time is None:
                absence_start_time = current_time

            # 计算脱岗持续时间
            absence_duration = (current_time - absence_start_time).total_seconds()

            # 脱岗超时触发预警
            if absence_duration >= 离岗阈值 and not alert_triggered:
                # 中文屏幕警告
                frame = cv2_puttext(frame, "⚠️ 警告：人员脱岗！", (width // 2 - 150, 50), 1.5, (0, 0, 255), 3)
                # 保存预警截图
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = os.path.join(screenshot_dir, f"absence_alert_{timestamp}.jpg")
                cv2.imwrite(screenshot_path, frame)
                print(f"\n[{timestamp}] 检测到脱岗（已超过{离岗阈值}秒），截图保存至：{screenshot_path}")
                alert_triggered = True

        # 绘制右移的细长ROI
        roi_np = np.array(ROI, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [roi_np], True, (0, 255, 0), 2)
        frame = cv2_puttext(frame, "监控区域", (ROI[0][0] + 10, ROI[0][1] - 15), 1, (0, 255, 0), 2)

        # 显示状态信息
        if status == "在岗":
            frame = cv2_puttext(frame, f"状态: {status}", (30, 50), 1, (0, 255, 0), 2)
        else:
            frame = cv2_puttext(frame, f"状态: {status}", (30, 50), 1, (0, 0, 255), 2)
            # 显示脱岗持续时间
            if absence_start_time:
                absence_duration = (current_time - absence_start_time).total_seconds()
                frame = cv2_puttext(frame, f"脱岗时长: {absence_duration:.1f}秒", (30, 100), 1, (0, 0, 255), 2)

        # 实时显示
        cv2.imshow("离岗检测（实时）", frame)

        # 保存处理后视频
        if save_result and out.isOpened():
            out.write(frame)

        # 按键控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n用户已退出程序")
            break
        elif key == ord('p'):
            print("\n已暂停，按任意键继续...")
            cv2.waitKey(0)

    # 释放资源
    cap.release()
    if save_result and out is not None:
        out.release()
    cv2.destroyAllWindows()

    # 结果总结
    print(f"\n处理总结：")
    print(f"  - 总处理帧数：{current_frame}")
    print(f"  - 预警截图目录：{os.path.abspath(screenshot_dir)}")
    if save_result:
        print(f"  - 处理后视频：{os.path.abspath(output_video_path)}")
    print(f"  - 监控区域：右移细长型 {ROI[0]} 至 {ROI[2]}")