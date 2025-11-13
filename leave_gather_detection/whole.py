from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
import os

# 1. 创建截图保存文件夹
def create_screenshot_dir():
    screenshot_dir = "alert_screenshots"
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir)
        print(f"已创建截图文件夹：{screenshot_dir}")
    return screenshot_dir

# 2. 中文显示函数
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

# 3. 点是否在ROI内（射线法）
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

# 4. 主函数
def main():
    # 配置参数
    video_path = "/media/zhang/data2/peopledetect/test2.mp4"  # 视频路径
    save_result = True  # 是否保存处理后视频
    output_video_path = "processed_video.mp4"  # 输出视频路径
    
    # 两个ROI区域（可根据实际场景调整）
    # 聚集检测区（示例：画面右侧区域）
    gather_roi = [(600, 100), (1000, 100), (1000, 700), (600, 700)]
    gather_threshold = 5  # 聚集人数阈值
    
    # 离岗检测区（示例：画面左侧区域）
    leave_roi = [(100, 100), (500, 100), (500, 700), (100, 700)]
    leave_threshold = 10  # 离岗时间阈值（秒）

    # 初始化截图目录
    screenshot_dir = create_screenshot_dir()

    # 读取视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频 {video_path}")
        exit()

    # 获取视频参数
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 初始化视频写入器
    out = None
    if save_result:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if not out.isOpened():
            print(f"警告：无法保存输出视频，已关闭保存功能")
            save_result = False

    # 加载YOLO模型（仅检测行人）
    try:
        model = YOLO("yolov12n.pt")
        if model.names.get(0) != "person":
            print("警告：模型类别0不是'person'，可能无法正确检测行人")
        else:
            print("模型已加载：同时检测人员聚集和离岗")
    except Exception as e:
        print(f"模型加载失败：{e}")
        exit()

    # 初始化状态变量
    # 离岗检测状态
    leave_absence_start = None  # 离岗开始时间
    leave_alert_triggered = False  # 是否已触发离岗预警
    
    # 聚集检测状态
    gather_alert_triggered = False  # 是否已触发聚集预警（避免重复截图）

    # 实时处理
    print("开始处理，窗口将显示实时画面：")
    print("  - 按 'p' 暂停，按任意键继续")
    print("  - 按 'q' 退出程序")

    current_frame = 0
    cv2.namedWindow("人员聚集+离岗检测（实时）", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n视频处理完成！")
            break

        current_frame += 1
        # 打印进度
        if current_frame % 50 == 0:
            progress = int(current_frame / total_frames * 100)
            print(f"进度：{current_frame}/{total_frames} 帧 ({progress}%)", end="\r")

        # 检测行人（仅检测类别0：person）
        results = model(frame, classes=[0], verbose=False)
        person_boxes = []
        for box in results[0].boxes:
            if int(box.cls[0]) == 0:  # 确认是行人
                person_boxes.append(box.xyxy.cpu().numpy()[0])

        # --------------- 1. 离岗检测逻辑 ---------------
        # 统计离岗检测区内的人数
        leave_person_count = 0
        for box in person_boxes:
            x1, y1, x2, y2 = box.astype(int)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            if point_in_roi(center, leave_roi):
                leave_person_count += 1
                # 离岗区内的人用绿色框标记
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 离岗状态判断
        current_time = datetime.now()
        leave_status = "在岗" if leave_person_count > 0 else "脱岗"
        
        # 处理离岗计时
        if leave_status == "在岗":
            leave_absence_start = None
            leave_alert_triggered = False  # 重置预警状态
        else:
            if leave_absence_start is None:
                leave_absence_start = current_time  # 记录离岗开始时间
            else:
                # 计算离岗持续时间
                leave_duration = (current_time - leave_absence_start).total_seconds()
                # 超时且未触发过预警时，保存截图
                if leave_duration >= leave_threshold and not leave_alert_triggered:
                    frame = cv2_puttext(frame, "⚠️ 警告：人员离岗超时！", 
                                       (50, 100), 1.2, (0, 0, 255), 3)
                    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
                    screenshot_path = os.path.join(screenshot_dir, f"leave_alert_{timestamp}.jpg")
                    cv2.imwrite(screenshot_path, frame)
                    print(f"\n[{timestamp}] 离岗超时（{leave_duration:.1f}秒），截图保存至：{screenshot_path}")
                    leave_alert_triggered = True

        # --------------- 2. 聚集检测逻辑 ---------------
        # 统计聚集检测区内的人数
        gather_person_count = 0
        for box in person_boxes:
            x1, y1, x2, y2 = box.astype(int)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            if point_in_roi(center, gather_roi):
                gather_person_count += 1
                # 聚集区内的人用红色框标记
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 聚集状态判断
        gather_status = "正常" if gather_person_count < gather_threshold else "聚集"
        # 聚集且未触发过预警时，保存截图
        if gather_status == "聚集" and not gather_alert_triggered:
            frame = cv2_puttext(frame, "⚠️ 警告：人员聚集！", 
                               (width//2 + 50, 100), 1.2, (0, 0, 255), 3)
            timestamp = current_time.strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(screenshot_dir, f"gather_alert_{timestamp}.jpg")
            cv2.imwrite(screenshot_path, frame)
            print(f"\n[{timestamp}] 人员聚集（{gather_person_count}人），截图保存至：{screenshot_path}")
            gather_alert_triggered = True
        elif gather_status == "正常":
            gather_alert_triggered = False  # 重置预警状态

        # --------------- 绘制区域和信息 ---------------
        # 绘制离岗检测区（绿色多边形）
        leave_roi_np = np.array(leave_roi, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [leave_roi_np], True, (0, 255, 0), 2)
        frame = cv2_puttext(frame, "离岗检测区", 
                           (leave_roi[0][0] + 10, leave_roi[0][1] - 15), 
                           0.8, (0, 255, 0), 2)

        # 绘制聚集检测区（蓝色多边形）
        gather_roi_np = np.array(gather_roi, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [gather_roi_np], True, (255, 0, 0), 2)
        frame = cv2_puttext(frame, "聚集检测区", 
                           (gather_roi[0][0] + 10, gather_roi[0][1] - 15), 
                           0.8, (255, 0, 0), 2)

        # 显示离岗检测信息
        frame = cv2_puttext(frame, f"离岗区人数: {leave_person_count}", 
                           (50, 50), 0.9, (0, 255, 0), 2)
        if leave_status == "脱岗" and leave_absence_start:
            leave_duration = (current_time - leave_absence_start).total_seconds()
            frame = cv2_puttext(frame, f"脱岗时长: {leave_duration:.1f}秒", 
                               (50, 80), 0.9, (0, 0, 255), 2)

        # 显示聚集检测信息
        frame = cv2_puttext(frame, f"聚集区人数: {gather_person_count}", 
                           (width//2 + 50, 50), 0.9, (255, 0, 0), 2)
        frame = cv2_puttext(frame, f"聚集阈值: {gather_threshold}人", 
                           (width//2 + 50, 80), 0.9, (255, 0, 0), 2)

        # --------------- 显示与保存 ---------------
        cv2.imshow("人员聚集+离岗检测（实时）", frame)
        
        # 保存视频
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
    print(f"  - 离岗检测区阈值：{leave_threshold}秒")
    print(f"  - 聚集检测区阈值：{gather_threshold}人")

if __name__ == "__main__":
    main()
