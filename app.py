"""LocateAnything-VideoClipper 主入口 - Gradio UI + 三阶段过滤"""
import os
import sys
import threading
import queue
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*HTTP_422.*")
warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*")

import gradio as gr

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "model_weights")
FFMPEG_PATH = os.path.join(PROJECT_ROOT, "ffmpeg.exe")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model_worker import ModelWorker
from video_processor import find_videos, find_images, process_video, process_image, cleanup_video_memory
from timeline_merger import merge_timeline, format_timestamp
from ffmpeg_cutter import cut_video

# 全局日志
log_lines = []
log_queue = queue.Queue()
processing_lock = threading.Lock()


def log_callback(msg):
    log_lines.append(msg)
    log_queue.put(msg)


def read_logs():
    return "\n".join(log_lines)


def clear_logs():
    log_lines.clear()
    while not log_queue.empty():
        try:
            log_queue.get_nowait()
        except queue.Empty:
            break


def run_pipeline(input_folder, prompts_str, output_folder,
                 confidence_threshold, min_consecutive, min_clip_duration, scan_step):
    """视频处理主流程（三阶段过滤 + 动态步长）"""
    with processing_lock:
        try:
            prompts = [p.strip() for p in prompts_str.split(",") if p.strip()]
            if not prompts:
                log_callback("[错误] 请至少输入一个提示词")
                return

            input_folder = input_folder.strip()
            output_folder = output_folder.strip()
            if not input_folder or not os.path.isdir(input_folder):
                log_callback(f"[错误] 输入文件夹不存在: {input_folder}")
                return
            if not output_folder:
                output_folder = os.path.join(PROJECT_ROOT, "output_clips")
            os.makedirs(output_folder, exist_ok=True)

            videos = find_videos(input_folder)
            images = find_images(input_folder)

            if not videos and not images:
                log_callback("[错误] 输入文件夹中未找到支持的视频或图片文件")
                return

            log_callback(f"找到 {len(videos)} 个视频, {len(images)} 张图片")
            log_callback(f"提示词: {', '.join(prompts)}")
            log_callback(f"输出目录: {output_folder}")
            log_callback(f"过滤参数: 置信度≥{confidence_threshold}, 连续帧≥{min_consecutive}, 最短片段≥{min_clip_duration}s, 盲搜步长={scan_step}s")
            log_callback("=" * 50)

            log_callback("正在加载模型，请稍候...")
            worker = ModelWorker(MODEL_PATH)
            worker.load()
            log_callback("模型加载完成！")
            log_callback("=" * 50)

            total_clips = 0
            for idx, video_path in enumerate(videos):
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                log_callback(f"\n[{idx + 1}/{len(videos)}] 处理视频: {video_name}")

                # 第一阶段+第二阶段：在 process_video 内完成置信度过滤 + 时序连续性校验
                results = process_video(
                    video_path, worker, prompts, fps=1.0,
                    confidence_threshold=confidence_threshold,
                    min_consecutive=min_consecutive,
                    scan_step=scan_step,
                    output_dir=output_folder, log_callback=log_callback
                )

                # 第三阶段：时间轴融合 + 最小片段过滤
                for prompt in prompts:
                    detected = results.get(prompt, [])
                    if not detected:
                        log_callback(f"  提示词[{prompt}] 未检测到有效目标")
                        continue

                    segments = merge_timeline(
                        detected, merge_gap=3, buffer_time=scan_step,
                        min_clip_duration=min_clip_duration
                    )
                    if not segments:
                        log_callback(f"  提示词[{prompt}] 合并后无足够长的片段（最短{min_clip_duration}s）")
                        continue

                    log_callback(
                        f"  提示词[{prompt}] 有效帧{len(detected)}个, "
                        f"合并为{len(segments)}个片段"
                    )

                    for seg_idx, (start, end) in enumerate(segments):
                        ts_start = format_timestamp(start)
                        ts_end = format_timestamp(end)
                        output_name = f"{video_name}_{prompt}_{ts_start}-{ts_end}_片段{seg_idx+1}.mp4"
                        output_path = os.path.join(output_folder, output_name)

                        success = cut_video(FFMPEG_PATH, video_path, start, end, output_path, log_callback=log_callback)
                        if success:
                            total_clips += 1

                log_callback(f"  清理显存缓存...")
                cleanup_video_memory()

            # 处理图片
            for idx, image_path in enumerate(images):
                img_name = os.path.splitext(os.path.basename(image_path))[0]
                log_callback(f"\n[{idx + 1}/{len(images)}] 处理图片: {img_name}")
                process_image(
                    image_path, worker, prompts,
                    confidence_threshold=confidence_threshold,
                    output_dir=output_folder, log_callback=log_callback
                )
                cleanup_video_memory()

            log_callback("=" * 50)
            log_callback(f"全部完成！共生成 {total_clips} 个视频片段, 处理 {len(images)} 张图片")
            log_callback(f"输出目录: {output_folder}")

        except Exception as e:
            log_callback(f"[致命错误] {e}")
            import traceback
            log_callback(traceback.format_exc())


def start_processing(input_folder, prompts, output_folder,
                     confidence_threshold, min_consecutive, min_clip_duration, scan_step):
    if not processing_lock.locked():
        clear_logs()
        t = threading.Thread(
            target=run_pipeline,
            args=(input_folder, prompts, output_folder,
                  confidence_threshold, min_consecutive, min_clip_duration, scan_step),
            daemon=True,
        )
        t.start()
        return "处理已启动，请查看日志..."
    else:
        return "上一个任务仍在处理中，请等待完成..."


def update_logs():
    return read_logs()


# 构建 Gradio UI
with gr.Blocks(title="LocateAnything-VideoClipper") as demo:
    gr.Markdown("# LocateAnything-VideoClipper")
    gr.Markdown("基于 LocateAnything-3B 模型的视频/图片智能切片工具（三阶段抗误检）")

    with gr.Row():
        with gr.Column(scale=1):
            input_folder = gr.Textbox(
                label="输入文件夹路径",
                placeholder="例如: C:\\Videos\\input",
                value=os.path.join(PROJECT_ROOT, "input_videos"),
            )
            prompts_input = gr.Textbox(
                label="提示词（逗号分隔）",
                placeholder="例如: 黑猫, 穿红衣服的人, 汽车",
            )
            output_folder = gr.Textbox(
                label="输出文件夹路径",
                value=os.path.join(PROJECT_ROOT, "output_clips"),
            )

            gr.Markdown("### 抗误检参数")
            confidence_slider = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.65, step=0.05,
                label="置信度阈值（越高越严格，过滤越多）",
            )
            consecutive_slider = gr.Slider(
                minimum=1, maximum=10, value=3, step=1,
                label="最小连续帧数（目标必须连续出现N帧才保留）",
            )
            min_duration_slider = gr.Slider(
                minimum=0, maximum=30, value=1, step=1,
                label="最短输出片段（秒，设为0则全部保留）",
            )
            scan_step_slider = gr.Slider(
                minimum=1, maximum=15, value=5, step=1,
                label="盲搜步长（秒，无目标时每N秒抽1帧，越大数据越快）",
            )

            start_btn = gr.Button("开始处理", variant="primary", size="lg")
            status_text = gr.Textbox(label="状态", interactive=False)

        with gr.Column(scale=2):
            log_box = gr.Textbox(
                label="处理日志",
                lines=25,
                interactive=False,
            )

    start_btn.click(
        fn=start_processing,
        inputs=[input_folder, prompts_input, output_folder,
                confidence_slider, consecutive_slider, min_duration_slider, scan_step_slider],
        outputs=[status_text],
    )

    timer = gr.Timer(2)
    timer.tick(fn=update_logs, outputs=[log_box])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
