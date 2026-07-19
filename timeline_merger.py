"""智能时间轴合并模块 - 将离散检测秒数合并为连续区间 + 最小片段过滤"""


def merge_timeline(
    detected_seconds,
    merge_gap=3,
    buffer_time=1,
    min_clip_duration=1,
):
    """将检测到的离散秒数合并为 [开始, 结束] 区间

    Args:
        detected_seconds: 检测到的秒数列表
        merge_gap: 相邻片段间隔小于此值(秒)则合并，默认3秒
        buffer_time: 区间前后各加的缓冲时间(秒)，默认1秒
        min_clip_duration: 最小输出片段长度(秒)，低于此值的片段丢弃，默认4秒

    Returns:
        list of (start_sec, end_sec) 元组
    """
    if not detected_seconds:
        return []

    sorted_secs = sorted(set(detected_seconds))

    # 按 merge_gap 合并为区间
    segments = []
    seg_start = sorted_secs[0]
    seg_end = sorted_secs[0]

    for sec in sorted_secs[1:]:
        if sec - seg_end <= merge_gap:
            seg_end = sec
        else:
            segments.append((seg_start, seg_end))
            seg_start = sec
            seg_end = sec
    segments.append((seg_start, seg_end))

    # 添加缓冲时间
    buffered = []
    for start, end in segments:
        buffered_start = max(0, start - buffer_time)
        buffered_end = end + buffer_time
        buffered.append((buffered_start, buffered_end))

    # 合并重叠区间
    if len(buffered) > 1:
        merged = [buffered[0]]
        for start, end in buffered[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
    else:
        merged = buffered

    # 最小片段长度过滤（0秒 = 全部保留）
    if min_clip_duration <= 0:
        final = merged
    else:
        final = [(s, e) for s, e in merged if (e - s) >= min_clip_duration]

    return final


def format_timestamp(total_seconds):
    """将秒数转为 HH_MM_SS 格式"""
    h = int(total_seconds) // 3600
    m = (int(total_seconds) % 3600) // 60
    s = int(total_seconds) % 60
    return f"{h:02d}_{m:02d}_{s:02d}"
