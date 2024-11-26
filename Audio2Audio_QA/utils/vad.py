import bisect  # 提供了支持维护已排序列表的函数，能够高效地查找和插入元素
import functools
import os
import warnings

from typing import List, NamedTuple, Optional

import numpy as np


# The code below is adapted from https://github.com/snakers4/silero-vad.
class VadOptions(NamedTuple):
    """VAD options.
    Attributes:
      threshold: Speech threshold. Silero VAD outputs speech probabilities for each audio chunk,
        probabilities ABOVE this value are considered as SPEECH. It is better to tune this
        parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.
      min_speech_duration_ms: Final speech chunks shorter min_speech_duration_ms are thrown out.
      max_speech_duration_s: Maximum duration of speech chunks in seconds. Chunks longer
        than max_speech_duration_s will be split at the timestamp of the last silence that
        lasts more than 100ms (if any), to prevent aggressive cutting. Otherwise, they will be
        split aggressively just before max_speech_duration_s.
      min_silence_duration_ms: In the end of each speech chunk wait for min_silence_duration_ms
        before separating it
      window_size_samples: Audio chunks of window_size_samples size are fed to the silero VAD model.
        WARNING! Silero VAD models were trained using 512, 1024, 1536 samples for 16000 sample rate.
        Values other than these may affect model performance!!
      speech_pad_ms: Final speech chunks are padded by speech_pad_ms each side
    """

    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    max_speech_duration_s: float = float("inf")
    min_silence_duration_ms: int = 2000
    window_size_samples: int = 1024
    speech_pad_ms: int = 400


def get_speech_timestamps(
    audio: np.ndarray,
    vad_options: Optional[VadOptions] = None,  # 表示 vad_options 可以是 VadOptions 类型的实例，也可以是 None。
    **kwargs,
) -> List[dict]:
    """This method is used for splitting long audios into speech chunks using silero VAD.
    Args:
      audio: One dimensional float array.
      vad_options: Options for VAD processing.
      kwargs: VAD options passed as keyword arguments for backward compatibility.
    Returns:
      List of dicts containing begin and end samples of each speech chunk.
    """
    if vad_options is None:
        vad_options = VadOptions(**kwargs)

    threshold = vad_options.threshold
    min_speech_duration_ms = vad_options.min_speech_duration_ms  # 只有当语音段的持续时间超过这个值时，才会被记录为有效语音。这可以帮助过滤掉短促的噪声。
    max_speech_duration_s = vad_options.max_speech_duration_s  
    min_silence_duration_ms = vad_options.min_silence_duration_ms
    window_size_samples = vad_options.window_size_samples  # 用于信号处理中的分帧操作，决定每个处理窗口包含的样本数量。
    speech_pad_ms = vad_options.speech_pad_ms  # 语音填充时间，以毫秒为单位。在检测到语音段后，可能需要向前或向后填充一些时间，以确保完整捕获语音内容。

    if window_size_samples not in [512, 1024, 1536]:
        warnings.warn(
            "Unusual window_size_samples! Supported window_size_samples:\n"
            " - [512, 1024, 1536] for 16000 sampling_rate"
        )

    sampling_rate = 16000
    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    # 为什么不是+，而是-?
    """
    减去窗口大小: 当我们将窗口大小从总样本数中减去时，实际上是在确保我们在处理的音频段中，不会因为最后一个窗口的存在而超出最大样本限制。
    也就是说，如果没有减去这个窗口的大小，可能会导致最后一个窗口不完整，影响处理效果。
    减去填充时间: 在计算中减去填充样本的数量是因为这些填充样本不应被包含在有效的语音样本数中。我们希望得到的是实际用于处理的样本数量，
    而不是包括上下文填充的样本。
    """
    max_speech_samples = (
        sampling_rate * max_speech_duration_s
        - window_size_samples
        - 2 * speech_pad_samples   # 2** speech_pad_samples表示需要在语音段的开始和结束都添加填充，以确保完整捕获语音信号。
    )
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sampling_rate * 98 / 1000  # 最大语音持续时间的最小静音样本数

    audio_length_samples = len(audio)

    model = get_vad_model()
    state = model.get_initial_state(batch_size=1)

    speech_probs = []
    for current_start_sample in range(0, audio_length_samples, window_size_samples):
        chunk = audio[current_start_sample : current_start_sample + window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = np.pad(chunk, (0, int(window_size_samples - len(chunk))))  # (0, N) 表示在块的开始位置不填充（0），在结束位置填充 N 个样本，这里的 N 等于 window_size_samples 减去当前 chunk 的长度。
        speech_prob, state = model(chunk, state, sampling_rate)
        speech_probs.append(speech_prob)

    triggered = False
    speeches = []
    current_speech = {}
    neg_threshold = threshold - 0.15

    # to save potential segment end (and tolerate some silence)
    temp_end = 0
    # to save potential segment limits in case of maximum segment size reached
    prev_end = next_start = 0

    for i, speech_prob in enumerate(speech_probs):
        if (speech_prob >= threshold) and temp_end:
            temp_end = 0  # 重置 temp_end 的值为 0，这通常意味着当前段落的结束状态已被处理，准备开始新的段落
            # 检查 next_start 是否小于 prev_end。prev_end 表示上一个处理块的结束位置。这种检查确保不会因为延续处理而导致样本索引的混乱。
            if next_start < prev_end:
                next_start = window_size_samples * i

        if (speech_prob >= threshold) and not triggered:
            triggered = True  # 将 triggered 设置为 True，表示已开始检测语音活动。
            current_speech["start"] = window_size_samples * i
            continue  # 进入下一次for循环迭代语音
        # 如果有语音活动且语音过长
        if (
            triggered
            and (window_size_samples * i) - current_speech["start"] > max_speech_samples
        ):  
            # 如果当前有效语音结束
            if prev_end:
                current_speech["end"] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                # previously reached silence (< neg_thres) and is still not speech (< thres)
                # 如果存在静音（下一个起始样本索引小于上一个结束样本索引）
                if next_start < prev_end:
                    triggered = False  # 说明不在语音活动中
                # 如果没有静音，则更新 current_speech["start"] 为 next_start，准备开始新的语音段。
                else:
                    current_speech["start"] = next_start
                # 重置相关状态变量，以便在下一轮处理中使用
                prev_end = next_start = temp_end = 0
            # 如果当前有效语音没有结束
            else:  
                current_speech["end"] = window_size_samples * i
                speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue
        # 静音检测
        if (speech_prob < neg_threshold) and triggered:
            # 记录静音结束时间
            if not temp_end:
                temp_end = window_size_samples * i
            # condition to avoid cutting in very short silence
            # 短静音处理:判断从 temp_end 到当前样本之间的时间是否超过最大语音静音样本数。若超过，则更新 prev_end 为 temp_end，表示更新上一个语音段的结束位置。
            if (window_size_samples * i) - temp_end > min_silence_samples_at_max_speech:
                prev_end = temp_end
            # 如果当前样本与 temp_end 之间的差值小于最小静音样本数，则继续下一次for循环，跳过后续处理。
            if (window_size_samples * i) - temp_end < min_silence_samples:
                continue
            # 结束当前语音段
            else:
                current_speech["end"] = temp_end
                if (
                    current_speech["end"] - current_speech["start"]
                ) > min_speech_samples:
                    speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue

    if (
        current_speech
        and (audio_length_samples - current_speech["start"]) > min_speech_samples
    ):
        current_speech["end"] = audio_length_samples
        speeches.append(current_speech)

    for i, speech in enumerate(speeches):
        # 调整语音起始位置
        if i == 0:
            speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
        # 处理两段语音之间静音较短的部分，即处理上段语音的结束位置和下段语音的开始位置
        if i != len(speeches) - 1:
            silence_duration = speeches[i + 1]["start"] - speech["end"]
            # 如果静音部分过短（即小于2倍填充）
            if silence_duration < 2 * speech_pad_samples:
                speech["end"] += int(silence_duration // 2)
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - silence_duration // 2)
                )
            else:
                speech["end"] = int(
                    min(audio_length_samples, speech["end"] + speech_pad_samples)
                )
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - speech_pad_samples)
                )
        # 结束位置
        else:
            speech["end"] = int(
                min(audio_length_samples, speech["end"] + speech_pad_samples)
            )

    return speeches

# 拼接音频片段
def collect_chunks(audio: np.ndarray, chunks: List[dict]) -> np.ndarray:
    if not chunks:
        return np.array([], dtype=np.float32)

    return np.concatenate([audio[chunk["start"] : chunk["end"]] for chunk in chunks])


class SpeechTimestampsMap:
    """Helper class to restore original speech timestamps."""

    def __init__(self, chunks: List[dict], sampling_rate: int, time_precision: int = 2):
        self.sampling_rate = sampling_rate
        self.time_precision = time_precision
        self.chunk_end_sample = []
        self.total_silence_before = []
        
        """
        previous_end = 0:
        初始化局部变量 previous_end，用于跟踪前一个音频片段的结束样本索引，初始值为 0。
        silent_samples = 0:
        初始化局部变量 silent_samples，用于计算或跟踪静音样本的数量，初始值为 0。
        """
        previous_end = 0
        silent_samples = 0

        for chunk in chunks:
            silent_samples += chunk["start"] - previous_end
            previous_end = chunk["end"]

            self.chunk_end_sample.append(chunk["end"] - silent_samples)
            self.total_silence_before.append(silent_samples / sampling_rate)
    # 获取原始语音时间戳
    def get_original_time(
        self,
        time: float,
        chunk_index: Optional[int] = None,
    ) -> float:
        if chunk_index is None:
            chunk_index = self.get_chunk_index(time)

        total_silence_before = self.total_silence_before[chunk_index]
        return round(total_silence_before + time, self.time_precision)

    def get_chunk_index(self, time: float) -> int:
        sample = int(time * self.sampling_rate)
        """
        使用 bisect 模块的 bisect 函数查找 sample 在 self.chunk_end_sample 列表中的插入位置。
        self.chunk_end_sample 是一个已排序的样本结束位置列表
        """
        return min(
            bisect.bisect(self.chunk_end_sample, sample), 
            len(self.chunk_end_sample) - 1,
        )

# lru_cache 是 functools 模块中的一个装饰器，用于缓存函数的返回值。
@functools.lru_cache
def get_vad_model():
    # 使用 os.path.dirname(__file__) 获取当前文件的目录，并将其与 "assets" 组合，形成资产目录的完整路径。
    asset_dir = os.path.join(os.path.dirname(__file__), "assets")
    path = os.path.join(asset_dir, "silero_vad.onnx")
    return SileroVADModel(path)  # 返回一个SileroVADModel类的实例


class SileroVADModel:
    def __init__(self, path):
        try:
            import onnxruntime
        except ImportError as e:
            raise RuntimeError(
                "Applying the VAD filter requires the onnxruntime package"
            ) from e

        opts = onnxruntime.SessionOptions()  # 初始化一个 SessionOptions 对象，用于配置推理会话的参数
        opts.inter_op_num_threads = 1  # 设置操作之间的并行线程数。在这里设置为 1，意味着操作之间不会并行执行。
        opts.intra_op_num_threads = 1  # 设置单个操作内部的并行线程数。这里也设置为 1，表示单个操作在执行时只使用一个线程。
        opts.log_severity_level = 4  # 将日志严重性级别设置为 4。ONNX Runtime 的日志级别从 0（最详细）到 5（最少）。级别 4 表示仅记录错误信息。

        self.session = onnxruntime.InferenceSession(
            path,
            providers=["CPUExecutionProvider"],  # 若使用GPU为"CUDAExecutionProvider"
            sess_options=opts,  # 这部分将之前定义的会话选项 opts 传递给推理会话。opts 包含了会话的一些配置，例如线程数和日志级别（如上述提到的设置）。
        )

    def get_initial_state(self, batch_size: int):
        """
        初始化隐藏状态（h）：
        使用 NumPy 创建一个全零数组，形状为 (2, batch_size, 64)。
        这里的 2 通常表示 RNN 的层数（例如双向 RNN），batch_size 是每批数据的样本数量，64 是隐藏状态的维度。
        dtype=np.float32 指定数据类型为 32 位浮点数。
        """
        h = np.zeros((2, batch_size, 64), dtype=np.float32)
        """
        初始化单元状态（c）：
        同样使用 NumPy 创建一个全零数组，形状和数据类型与 h 相同。
        在 LSTM 中，c 表示单元状态。
        """
        c = np.zeros((2, batch_size, 64), dtype=np.float32)
        return h, c

    def __call__(self, x, state, sr: int):
        if len(x.shape) == 1:
            x = np.expand_dims(x, 0)
        if len(x.shape) > 2:
            raise ValueError(
                f"Too many dimensions for input audio chunk {len(x.shape)}"
            )
        if sr / x.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")

        h, c = state

        ort_inputs = {
            "input": x,
            "h": h,
            "c": c,
            "sr": np.array(sr, dtype="int64"),
        }

        out, h, c = self.session.run(None, ort_inputs)
        state = (h, c)

        return out, state