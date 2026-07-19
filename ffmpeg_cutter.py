"""FFmpeg 无损视频裁剪模块"""
import os
import subprocess


def get_ffmpeg_path(project_root):
    """获取 ffmpeg.exe 的绝对路径"""
    return os.path.join(project_root, "ffmpeg.exe")


def cut_video(ffmpeg_path, input_path, start_sec, end_sec, output_path, log_callback=None):
    """使用 FFmpeg 进行无损流复制裁剪

    Args:
        ffmpeg_path: ffmpeg.exe 绝对路径
        input_path: 输入视频绝对路径
        start_sec: 开始时间(秒)
        end_sec: 结束时间(秒)
        output_path: 输出视频绝对路径
        log_callback: 日志回调

    Returns:
        bool: 是否成功
    """
    if log_callback is None:
        log_callback = lambda msg: None

    duration = end_sec - start_sec
    if duration <= 0:
        log_callback(f"[警告] 跳过无效区间: {start_sec}-{end_sec}")
        return False

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 构建命令 - 所有路径加双引号防空格/中文
    cmd = [
        ffmpeg_path,
        "-ss", str(start_sec),
        "-i", input_path,
        "-t", str(duration),
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        "-y",
        output_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
        if result.returncode == 0:
            log_callback(f"  裁剪成功: {os.path.basename(output_path)}")
            return True
        else:
            stderr = result.stderr.decode("utf-8", errors="replace")
            log_callback(f"  [错误] FFmpeg 裁剪失败: {stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        log_callback(f"  [错误] FFmpeg 裁剪超时")
        return False
    except Exception as e:
        log_callback(f"  [错误] FFmpeg 调用异常: {e}")
        return False
