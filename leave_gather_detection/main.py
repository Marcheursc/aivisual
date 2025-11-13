from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
import os

def process_gather_detection(video_path):
    """处理聚集检测视频任务"""
    import cv2
    import numpy as np
    import os
    from datetime import datetime
    from ultralytics import YOLO
    
    # 创建截图保存文件夹
    screenshot_dir = "alert_screenshots"
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir)

    # Linux中文字体配置（使用PIL绘制中文，解决OpenCV中文乱码问题）
    def cv2_puttext(img, text, org, font_scale=1, color=(255, 0, 0), thickness=2):
        from PIL import Image, ImageDraw, ImageFont
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        # Linux系统中文字体路径（自动查找可用字体）
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

    # 核心参数配置
    ROI = [(220, 300), (700, 300), (700, 700), (200, 700)]
    聚集阈值 = 5

    # 点是否在ROI内（射线法判断）
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

    # 设置输出视频路径
    output_video_path = f"gather_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    
    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise Exception("无法创建输出视频文件")

    # 加载YOLOv12模型
    try:
        model = YOLO("yolov12n.pt")
    except Exception as e:
        raise Exception(f"模型加载失败：{e}")

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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 绘制ROI区域
        roi_np = np.array(ROI, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [roi_np], True, (0, 255, 0), 2)

        # 显示ROI内人数
        frame = cv2_puttext(frame, f"ROI内人数: {roi_person_count}", (30, 50), font_scale=1, color=(255, 0, 0), thickness=2)

        # 聚集预警
        if roi_person_count >= 聚集阈值:
            frame = cv2_puttext(frame, "警告：人员聚集！", (30, 100), font_scale=1.2, color=(0, 0, 255), thickness=3)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(screenshot_dir, f"alert_{timestamp}.jpg")
            cv2.imwrite(screenshot_path, frame)

        # 写入处理后的帧
        out.write(frame)

    # 释放资源
    cap.release()
    out.release()
    
    return output_video_path

# 1. 创建截图保存文件夹
screenshot_dir = "alert_screenshots"
if not os.path.exists(screenshot_dir):
    os.makedirs(screenshot_dir)
    print(f"已创建截图文件夹：{screenshot_dir}")


# 2. Linux中文字体配置（使用PIL绘制中文，解决OpenCV中文乱码问题）
def cv2_puttext(img, text, org, font_scale=1, color=(255, 0, 0), thickness=2):
    from PIL import Image, ImageDraw, ImageFont
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # BGR转RGB（PIL使用RGB）
    draw = ImageDraw.Draw(img_pil)
    # Linux系统中文字体路径（自动查找可用字体）
    font_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
    ]
    font = None
    for path in font_paths:
        if os.path.exists(path):
            try:
                # 字体大小与font_scale关联（20为基础值，可调整）
                font = ImageFont.truetype(path, int(font_scale * 20))
                break
            except:
                continue
    if not font:
        font = ImageFont.load_default()  # 若无中文字体，使用默认字体
    draw.text(org, text, font=font, fill=color)  # 绘制文字
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)  # RGB转回BGR（OpenCV使用BGR）


# 3. 核心参数配置
video_path = "/media/zhang/data2/peopledetect/test2.mp4"  # 视频路径
ROI = [(220, 300), (700, 300), (700, 700), (200, 700)]  # 感兴趣区域（多边形顶点）
聚集阈值 = 5  # 触发聚集预警的人数阈值
save_result = True  # 是否保存处理后的视频
output_video_path = "processed_video.mp4"  # 输出视频路径


# 4. 点是否在ROI内（射线法判断）
def point_in_roi(point, roi):
    x, y = point
    n = len(roi)
    inside = False
    for i in range(n):
        j = (i + 1) % n
        xi, yi = roi[i]
        xj, yj = roi[j]
        # 判断点是否在多边形内部
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
    return inside


# 5. 读取视频
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"错误：无法打开视频 {video_path}")
    exit()

# 获取视频参数（用于保存输出视频）
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

# 7. 加载YOLOv12模型，明确只检测行人（类别0）
try:
    model = YOLO("yolov12n.pt")  # 加载模型（可替换为yolov12s.pt等其他版本）
    # 确认模型类别0为"person"（COCO数据集标准）
    if model.names.get(0) != "person":
        print("警告：模型类别0不是'person'，可能无法正确检测行人")
    else:
        print("模型已确认：仅检测行人（类别0）")
except Exception as e:
    print(f"模型加载失败：{e}")
    exit()

# 8. 实时处理与显示
print("开始处理，窗口将显示实时画面：")
print("  - 按 'p' 暂停，按任意键继续")
print("  - 按 'q' 退出程序")

current_frame = 0
cv2.namedWindow("人员聚集检测（实时）", cv2.WINDOW_NORMAL)  # 创建可调整大小的窗口

while True:
    ret, frame = cap.read()
    if not ret:
        print("\n视频处理完成！")
        break

    current_frame += 1
    # 每50帧打印一次进度
    if current_frame % 50 == 0:
        progress = int(current_frame / total_frames * 100)
        print(f"进度：{current_frame}/{total_frames} 帧 ({progress}%)", end="\r")

    # 检测行人：强制只输出类别0（person）
    results = model(frame, classes=[0], verbose=False)  # 仅检测行人，关闭详细日志
    # 提取检测结果中的边界框（双重过滤：只保留类别0）
    person_boxes = []
    for box in results[0].boxes:
        cls = int(box.cls[0])  # 获取类别ID
        if cls == 0:  # 仅保留行人
            person_boxes.append(box.xyxy.cpu().numpy()[0])  # 提取边界框（x1,y1,x2,y2）

    # 统计ROI内人数
    roi_person_count = 0
    for box in person_boxes:
        x1, y1, x2, y2 = box.astype(int)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)  # 计算目标中心坐标
        if point_in_roi(center, ROI):  # 判断中心是否在ROI内
            roi_person_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 绘制边界框（红色）

    # 绘制ROI区域（绿色多边形）
    roi_np = np.array(ROI, np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [roi_np], True, (0, 255, 0), 2)

    # 显示ROI内人数（使用自定义中文绘制函数）
    frame = cv2_puttext(frame, f"ROI内人数: {roi_person_count}",
                        (30, 50), font_scale=1, color=(255, 0, 0), thickness=2)

    # 聚集预警（超过阈值时）
    if roi_person_count >= 聚集阈值:
        # 显示警告文字（红色）
        frame = cv2_puttext(frame, "警告：人员聚集！",
                            (30, 100), font_scale=1.2, color=(0, 0, 255), thickness=3)
        # 保存预警截图
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = os.path.join(screenshot_dir, f"alert_{timestamp}.jpg")
        cv2.imwrite(screenshot_path, frame)
        print(f"\n[{timestamp}] 检测到聚集（{roi_person_count}人），截图保存至：{screenshot_path}")

    # 实时显示画面
    cv2.imshow("人员聚集检测（实时）", frame)

    # 保存处理后的视频
    if save_result and out.isOpened():
        out.write(frame)

    # 按键控制
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # 按'q'退出
        print("\n用户已退出程序")
        break
    elif key == ord('p'):  # 按'p'暂停
        print("\n已暂停，按任意键继续...")
        cv2.waitKey(0)

# 释放资源
cap.release()
if save_result and out is not None:
    out.release()
cv2.destroyAllWindows()

# 处理结果总结
print(f"\n处理总结：")
print(f"  - 总处理帧数：{current_frame}")
print(f"  - 预警截图目录：{os.path.abspath(screenshot_dir)}")
if save_result:
    print(f"  - 处理后视频：{os.path.abspath(output_video_path)}")
