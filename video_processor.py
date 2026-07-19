"""视频帧提取与处理模块 - 动态步长 + 下采样 + 并发提示词"""
import os
import gc
import re
import cv2
import torch
import numpy as np
from PIL import Image


SUPPORTED_VIDEO_FORMATS = {".mp4", ".mkv", ".avi", ".mov"}
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_FORMATS = SUPPORTED_VIDEO_FORMATS | SUPPORTED_IMAGE_FORMATS
TARGET_WIDTH = 448  # 模型输入缩放目标宽度


def safe_filename(text):
    """将文本转为安全的文件名（只保留字母数字下划线短横线）"""
    text = text.replace(" ", "_")
    text = re.sub(r'[^\w\-.]', '_', text)
    text = re.sub(r'_+', '_', text)
    return text.strip('_')[:50]


def draw_boxes_on_frame(frame, boxes, prompt, w, h):
    """在帧上绘制检测框和标签"""
    annotated = frame.copy()
    for box in boxes:
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{prompt}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.8, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(annotated, label, (x1, y1 - 5), font, 0.8, (0, 0, 0), 2)
    return annotated


def find_videos(folder):
    videos = []
    for entry in os.scandir(folder):
        if entry.is_file() and os.path.splitext(entry.name)[1].lower() in SUPPORTED_VIDEO_FORMATS:
            videos.append(os.path.abspath(entry.path))
    return sorted(videos)


def find_images(folder):
    images = []
    for entry in os.scandir(folder):
        if entry.is_file() and os.path.splitext(entry.name)[1].lower() in SUPPORTED_IMAGE_FORMATS:
            images.append(os.path.abspath(entry.path))
    return sorted(images)


def box_confidence_score(box, w, h):
    bw = (box["x2"] - box["x1"]) / w
    bh = (box["y2"] - box["y1"]) / h
    area_ratio = bw * bh
    # 面积分数：只惩罚极小框（<0.5%），大框不惩罚（全屏目标也是合法的）
    if area_ratio < 0.005:
        area_score = 0.0
    elif area_ratio < 0.03:
        area_score = area_ratio / 0.03 * 0.5
    else:
        area_score = 0.5 + 0.5 * min(1.0, area_ratio / 0.15)
    aspect = max(bw, bh) / max(min(bw, bh), 0.001)
    aspect_score = 1.0 if 0.2 <= aspect <= 5.0 else max(0.0, 1.0 - (aspect - 5.0) / 10.0)
    return area_score * 0.7 + aspect_score * 0.3


def is_dark_frame(frame, threshold=15):
    """检测是否为黑屏/暗帧（平均亮度低于阈值），跳过不送模型"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_val = gray.mean()
    return mean_val < threshold


def resize_frame(frame, target_width=TARGET_WIDTH):
    """等比例缩放帧到目标宽度"""
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / w
    new_h = int(h * scale)
    return cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_LINEAR)


def predict_multi_prompt(model_worker, pil_image, prompts):
    """合并多个提示词为一次推理（用 </c> 分隔），返回每个提示词的原始回答"""
    # LocateAnything 支持 </c> 分隔多类别
    combined = "</c>".join(prompts)
    prompt_text = f"Locate all the instances that match the following description: {combined}."
    result = model_worker.predict(pil_image, prompt_text, max_new_tokens=2048)
    answer = result["answer"]
    # 每个提示词分别解析：按 </c> 对应的检测块拆分
    # 简单策略：用整个 answer 为每个提示词做 parse_boxes（模型会输出所有类别的框）
    return {p: answer for p in prompts}


def process_video(video_path, model_worker, prompts, fps=1.0,
                  confidence_threshold=0.65, min_consecutive=3,
                  scan_step=5, output_dir=None, log_callback=None):
    """处理单个视频 - 动态步长 + 下采样 + 合并推理

    双阶段探索：
    A. 快速盲搜：scan_step 秒抽1帧
    B. 精准锁定：发现目标后切回每秒1帧追踪，目标消失后再切回盲搜
    """
    if log_callback is None:
        log_callback = lambda msg: None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log_callback(f"[错误] 无法打开视频: {os.path.basename(video_path)}")
        return {p: [] for p in prompts}

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0
    total_interval = max(1, int(video_fps / fps))
    scan_interval = total_interval * scan_step  # 盲搜步长（帧数）

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    log_callback(
        f"开始处理视频: {video_name} "
        f"(时长: {duration:.1f}s, 帧率: {video_fps:.1f}fps, "
        f"盲搜步长: {scan_step}s, 精准步长: 1s)"
    )

    results = {p: set() for p in prompts}
    frame_idx = 0
    processed_frames = 0

    # 动态状态：追踪模式 / 盲搜模式
    tracking = False
    tracking_end_sec = 0  # 目标消失后继续精准追踪的截止秒

    try:
        while True:
            # 决定当前帧是否需要处理
            current_sec = frame_idx / video_fps
            current_sec_int = int(current_sec)

            if tracking:
                # 精准模式：每秒抽1帧
                need_process = (frame_idx % total_interval == 0)
                # 目标消失后继续追踪直到 tracking_end_sec
                if current_sec_int > tracking_end_sec:
                    tracking = False
                    log_callback(f"  [{video_name}] {current_sec_int}s 目标消失，切回盲搜模式(步长{scan_step}s)")
                    need_process = (frame_idx % scan_interval == 0)
            else:
                # 盲搜模式：按 scan_step 抽帧
                need_process = (frame_idx % scan_interval == 0)

            if need_process:
                ret, frame = cap.read()
                if not ret:
                    break

                # 黑屏/转场帧跳过（不送模型，避免黑屏误检）
                if is_dark_frame(frame):
                    del frame
                    frame_idx += 1
                    continue

                h_orig, w_orig = frame.shape[:2]
                # 下采样到 448px 宽度
                frame_small = resize_frame(frame, TARGET_WIDTH)
                h_small, w_small = frame_small.shape[:2]

                # BGR → RGB → PIL
                frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                del frame_small

                # 合并多提示词一次推理
                answers = predict_multi_prompt(model_worker, pil_image, prompts)
                del pil_image

                frame_has_target = False
                for prompt in prompts:
                    answer = answers[prompt]
                    # 解析框坐标（需要映射回原始分辨率）
                    boxes = model_worker.parse_boxes(answer, w_small, h_small)
                    # 坐标缩放回原始分辨率
                    scale_x = w_orig / w_small
                    scale_y = h_orig / h_small
                    for b in boxes:
                        b["x1"] *= scale_x
                        b["y1"] *= scale_y
                        b["x2"] *= scale_x
                        b["y2"] *= scale_y

                    if "<box>none</box>" in answer:
                        boxes = []

                    # 置信度过滤
                    scored = [(b, box_confidence_score(b, w_orig, h_orig)) for b in boxes]
                    valid = [(b, s) for b, s in scored if s >= confidence_threshold]

                    if valid:
                        frame_has_target = True
                        results[prompt].add(current_sec_int)
                        best_score = valid[0][1]
                        log_callback(
                            f"  [{video_name}] {current_sec_int}s 发现[{prompt}] "
                            f"(置信度: {best_score:.2f})"
                        )
                        if output_dir:
                            all_boxes = [b for b, _ in valid]
                            annotated = draw_boxes_on_frame(frame, all_boxes, prompt, w_orig, h_orig)
                            ts = f"{current_sec_int // 3600:02d}_{(current_sec_int % 3600) // 60:02d}_{current_sec_int % 60:02d}"
                            safe_name = safe_filename(f"{video_name}_{prompt}")
                            # 用 PIL 保存（Windows 中文路径兼容性更好）
                            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                            pil_img = Image.fromarray(rgb)
                            pil_img.save(os.path.join(output_dir, f"{safe_name}_{ts}.jpg"), quality=95)

                del frame

                # 动态切换：发现目标 → 进入精准追踪
                if frame_has_target and not tracking:
                    tracking = True
                    tracking_end_sec = current_sec_int + scan_step + 5
                    log_callback(f"  [{video_name}] {current_sec_int}s 发现目标！切回精准追踪模式")
                elif frame_has_target and tracking:
                    tracking_end_sec = current_sec_int + scan_step + 5  # 延长追踪

                processed_frames += 1
                if processed_frames % 20 == 0:
                    progress = min(100, int(frame_idx / total_frames * 100))
                    log_callback(f"  [{video_name}] 进度 {progress}% (已处理{processed_frames}帧)")
            else:
                ret = cap.grab()
                if not ret:
                    break

            frame_idx += 1

    finally:
        cap.release()
        gc.collect()

    log_callback(f"视频处理完成: {video_name} (共处理 {processed_frames} 帧)")

    # 时序连续性校验
    filtered_results = {}
    for prompt in prompts:
        raw_secs = sorted(results[prompt])
        if not raw_secs:
            filtered_results[prompt] = []
            continue
        good_secs = []
        seg = [raw_secs[0]]
        for i in range(1, len(raw_secs)):
            if raw_secs[i] - raw_secs[i - 1] <= 1:
                seg.append(raw_secs[i])
            else:
                if len(seg) >= min_consecutive:
                    good_secs.extend(seg)
                seg = [raw_secs[i]]
        if len(seg) >= min_consecutive:
            good_secs.extend(seg)
        filtered = sorted(set(good_secs))
        dropped = len(raw_secs) - len(filtered)
        if dropped > 0:
            log_callback(f"  [{video_name}] 时序过滤: {len(raw_secs)}帧→{len(filtered)}帧 (丢弃{dropped}帧)")
        filtered_results[prompt] = filtered

    return filtered_results


def cleanup_video_memory():
    gc.collect()
    torch.cuda.empty_cache()


def process_image(image_path, model_worker, prompts, confidence_threshold=0.65, output_dir=None, log_callback=None):
    """处理单张图片，检测目标并保存带框截图

    Args:
        image_path: 图片文件路径
        model_worker: ModelWorker 实例
        prompts: 提示词列表
        confidence_threshold: 置信度阈值
        output_dir: 输出目录
        log_callback: 日志回调

    Returns:
        dict: {prompt: [box_list]} 每个提示词的检测结果
    """
    if log_callback is None:
        log_callback = lambda msg: None

    img_name = os.path.splitext(os.path.basename(image_path))[0]
    log_callback(f"处理图片: {img_name}")

    # 用 PIL 读取图片（支持中文路径）
    try:
        pil_orig = Image.open(image_path).convert("RGB")
    except Exception as e:
        log_callback(f"[错误] 无法读取图片: {image_path} ({e})")
        return {}

    # PIL → OpenCV 格式
    img = cv2.cvtColor(np.array(pil_orig), cv2.COLOR_RGB2BGR)
    h_orig, w_orig = img.shape[:2]

    # 下采样
    img_small = resize_frame(img, TARGET_WIDTH)
    h_small, w_small = img_small.shape[:2]

    # BGR → RGB → PIL
    frame_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    del img_small

    # 合并多提示词一次推理
    answers = predict_multi_prompt(model_worker, pil_image, prompts)
    del pil_image

    results = {}
    for prompt in prompts:
        answer = answers[prompt]
        boxes = model_worker.parse_boxes(answer, w_small, h_small)

        # 坐标缩放回原始分辨率
        scale_x = w_orig / w_small
        scale_y = h_orig / h_small
        for b in boxes:
            b["x1"] *= scale_x
            b["y1"] *= scale_y
            b["x2"] *= scale_x
            b["y2"] *= scale_y

        if "<box>none</box>" in answer:
            boxes = []

        # 置信度过滤
        scored = [(b, box_confidence_score(b, w_orig, h_orig)) for b in boxes]
        valid = [(b, s) for b, s in scored if s >= confidence_threshold]

        results[prompt] = valid

        if valid:
            log_callback(
                f"  [{img_name}] 发现[{prompt}] ({len(valid)}个, "
                f"置信度: {valid[0][1]:.2f})"
            )
            # 画框保存
            if output_dir:
                all_boxes = [b for b, _ in valid]
                annotated = draw_boxes_on_frame(img, all_boxes, prompt, w_orig, h_orig)
                safe_name = safe_filename(f"{img_name}_{prompt}")
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                pil_img.save(os.path.join(output_dir, f"{safe_name}.jpg"), quality=95)
        else:
            log_callback(f"  [{img_name}] 提示词[{prompt}] 未检测到目标")

    del img
    return results
