# https://huggingface.co/spaces/gradio/omni-mini/blob/main/app.py

import gradio as gr
from huggingface_hub import snapshot_download
from threading import Thread
import time
import base64
import numpy as np
import requests
import traceback
from dataclasses import dataclass, field
import io
from pydub import AudioSegment
import librosa
# 防止命名冲突报错，不使用utils，在这我随意命名为utils_pxd
from utils.vad import get_speech_timestamps, collect_chunks,VadOptions
import tempfile


from server import serve    # 从server.py中调用inference.py

repo_id = "gpt-omni/mini-omni"
snapshot_download(repo_id, local_dir="./checkpoint", revision="main")

IP = "0.0.0.0"
PORT = 60808

thread = Thread(target=serve, daemon=True) # deamon = True 表示当主程序退出时，这个线程会自动结束。
thread.start()

API_URL = "http://0.0.0.0:60808/chat"

# recording parameters  
IN_CHANNELS = 1  # 表示音频输入的通道数。在这里，1 表示单声道（mono）音频。
IN_RATE = 24000  # 表示音频的采样率（sample rate）。24000 Hz 意味着每秒钟采集 24000 个样本，适用于语音处理和某些音乐应用。
IN_CHUNK = 1024  # 表示每次读取音频数据的样本数。1024 是一个常见的缓冲区大小，用于实时音频处理。
IN_SAMPLE_WIDTH = 2  # 表示每个样本的字节数。在这里，2 字节表示使用 16 位（2 字节）深度的音频样本，这是 CD 音质的标准。
VAD_STRIDE = 0.5  # 表示语音活动检测（VAD）处理的步幅。0.5 表示每次处理时将前一段的 50% 重叠，这样可以更准确地检测到语音活动。

# playing parameters
OUT_CHANNELS = 1
OUT_RATE = 24000
OUT_SAMPLE_WIDTH = 2
OUT_CHUNK = 5760

# 这里重新定义了 OUT_CHUNK 为 20 * 4096，即 81920。这可能用于确保足够的缓冲区大小，以便在输出时流畅播放音频。OUT_RATE 和 OUT_CHANNELS 保持不变。
OUT_CHUNK = 20 * 4096
OUT_RATE = 24000
OUT_CHANNELS = 1

# 语音活动检测
def run_vad(ori_audio, sr):
    _st = time.time()
    try:
        audio = ori_audio
        audio = audio.astype(np.float32) / 32768.0   # 将音频数据转为浮点数后归一化缩放到[-1, 1]的范围（16 位 PCM 数据类型，范围为 -32768 到 32767）
        sampling_rate = 16000  # 目标采样率
        if sr != sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)  # 重采样

        vad_parameters = {}
        vad_parameters = VadOptions(**vad_parameters) # 解包操作符**: 这个操作符用于将字典的键值对转换为关键字参数，这样可以将字典中的每个项作为参数传递给 VadOptions 构造函数。
        speech_chunks = get_speech_timestamps(audio, vad_parameters)
        audio = collect_chunks(audio, speech_chunks)  # 获取语音片段/样本
        duration_after_vad = audio.shape[0] / sampling_rate  # 计算VAD后的音频时长，audio.shape[0]为样本总数

        if sr != sampling_rate:
            # resample to original sampling rate   按照输入的采样率进行采样
            vad_audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=sr)
        else:
            vad_audio = audio
        vad_audio = np.round(vad_audio * 32768.0).astype(np.int16)  # 将音频数据从浮点型转为16位整数型numpy数组
        vad_audio_bytes = vad_audio.tobytes()  # 将np数组转换为字节格式保存语音文件
        
        # 返回VAD处理后的语音时长、字节格式保存的语音文件、处理所用的时长（四舍五入并精确到小数点后4位）
        return duration_after_vad, vad_audio_bytes, round(time.time() - _st, 4)
    except Exception as e:
        # msg中第一个参数为计算音频时长。这里的 sr*2 假设音频是 16 位（即每个样本占 2 字节），因此总时长计算为样本数除以每秒的样本数（即采样率乘以每个样本的字节数）。
        # 通过 traceback.format_exc() 获取当前异常的详细堆栈跟踪信息，帮助调试。
        msg = f"[asr vad error] audio_len: {len(ori_audio)/(sr*2):.3f} s, trace: {traceback.format_exc()}"
        print(msg)
        return -1, ori_audio, round(time.time() - _st, 4)

# # 音频预热/测试
# def warm_up():
#     # 创建1024帧（每帧 2 byte，代表 16 位 PCM 数据）的静音帧数据，这通常用于测试或预热目的
#     # b"\x00\x00" 这是一个字节串，包含两个字节，均为 0。在音频处理中，这通常表示静音，因为 PCM（脉冲编码调制）音频格式中的 0 值表示没有声音。
#     frames = b"\x00\x00" * 1024 * 2  
#     dur, frames, tcost = run_vad(frames, 16000)
#     print(f"warm up done, time_cost: {tcost:.3f} s")

# warm_up()

# 使用了装饰器，传入AppState类，定义应用程序的状态，便于在音频处理和对话管理中跟踪当前状态和信息。
@dataclass
class AppState:
    stream: np.ndarray | None = None
    sampling_rate: int = 0
    pause_detected: bool = False
    started_talking: bool =  False
    stopped: bool = False
    conversation: list = field(default_factory=list)  # 保存会话信息，使用default_factory初始化空列表

# 判断是否音频暂停（用户是否暂停说话）
def determine_pause(audio: np.ndarray, sampling_rate: int, state: AppState) -> bool:
    """Take in the stream, determine if a pause happened"""

    temp_audio = audio
    
    dur_vad, _, time_vad = run_vad(temp_audio, sampling_rate)
    duration = len(audio) / sampling_rate

    # 如果语音时长大于0.5s且开始说话，返回False（表示用户没有暂停说话）
    if dur_vad > 0.5 and not state.started_talking:
        print("started talking")
        state.started_talking = True
        return False

    print(f"duration_after_vad: {dur_vad:.3f} s, time_vad: {time_vad:.3f} s")
    
    # 如果总音频时长 - VAD处理音频时长 > 1s，说明用户没有说话，返回True（即表示检测到暂停）
    return (duration - dur_vad) > 1


def speaking(audio_bytes: str):

    base64_encoded = str(base64.b64encode(audio_bytes), encoding="utf-8")
    files = {"audio": base64_encoded}
    with requests.post(API_URL, json=files, stream=True) as response:
        try:
            for chunk in response.iter_content(chunk_size=OUT_CHUNK):
                if chunk:
                    # Create an audio segment from the numpy array
                    audio_segment = AudioSegment(
                        chunk,
                        frame_rate=OUT_RATE,
                        sample_width=OUT_SAMPLE_WIDTH,
                        channels=OUT_CHANNELS,
                    )

                    # Export the audio segment to MP3 bytes - use a high bitrate to maximise quality
                    mp3_io = io.BytesIO()  # 初始化字节流，用于在内存中临时存储音频二进制数据
                    audio_segment.export(mp3_io, format="mp3", bitrate="320k")  # 导出音频片段为MP3格式，每秒传输 320,000 比特的数据（高音质）

                    # Get the MP3 bytes
                    mp3_bytes = mp3_io.getvalue()
                    mp3_io.close()
                    yield mp3_bytes

        except Exception as e:
            raise gr.Error(f"Error during audio streaming: {e}")




def process_audio(audio: tuple, state: AppState):
    # 第一次收到音频流数据，音频流数据和采样率使用初始值
    if state.stream is None:
        state.stream = audio[1]
        state.sampling_rate = audio[0]
    else:
        state.stream =  np.concatenate((state.stream, audio[1]))  # 将前后的音频拼接

    pause_detected = determine_pause(state.stream, state.sampling_rate, state)
    state.pause_detected = pause_detected

    if state.pause_detected and state.started_talking:
        return gr.Audio(recording=False), state
    return None, state


def response(state: AppState):
    # 如果没有检测到暂停且没有开始说话
    if not state.pause_detected and not state.started_talking:
        return None, AppState()  # 无需处理音频数据
    
    audio_buffer = io.BytesIO()  # 初创建BytesIO对象，用于保存后续导出的音频数据

    segment = AudioSegment(
        state.stream.tobytes(),
        frame_rate=state.sampling_rate,
        sample_width=state.stream.dtype.itemsize,
        channels=(1 if len(state.stream.shape) == 1 else state.stream.shape[1]),
    )
    segment.export(audio_buffer, format="wav")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_buffer.getvalue())
    
    state.conversation.append({"role": "user",
                                "content": {"path": f.name,
                                "mime_type": "audio/wav"}})
    
    output_buffer = b""   # 初始化空字节串，用于后续保存音频的二进制文件

    for mp3_bytes in speaking(audio_buffer.getvalue()):
        output_buffer += mp3_bytes
        yield mp3_bytes, state

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(output_buffer)
    
    state.conversation.append({"role": "assistant",
                    "content": {"path": f.name,
                                "mime_type": "audio/mp3"}})
    yield None, AppState(conversation=state.conversation)




def start_recording_user(state: AppState):
    if not state.stopped:
        return gr.Audio(recording=True)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_audio = gr.Audio(
                label="Input Audio", sources="microphone", type="numpy"
            )
        with gr.Column():
            chatbot = gr.Chatbot(label="Conversation", type="messages")
            output_audio = gr.Audio(label="Output Audio", streaming=True, autoplay=True)
    state = gr.State(value=AppState())

    stream = input_audio.stream(
        process_audio,
        [input_audio, state],  # 传递给 process_audio 函数，用于处理音频数据
        [input_audio, state],  # 传递给 input_audio.stream 方法，用于控制流的行为或进行状态管理。
        stream_every=0.50,  # 每0.5s处理一次音频流
        time_limit=30,  # 最长录制时间不超过30s
    )
    # 机器人回答后停止录音
    respond = input_audio.stop_recording(
        response,
        [state],  # 传递给response函数
        [output_audio, state]
    )
    respond.then(lambda s: s.conversation, [state], [chatbot])

    restart = output_audio.stop(
        start_recording_user,
        [state],
        [input_audio]
    )
    cancel = gr.Button("Stop Conversation", variant="stop")
    cancel.click(lambda: (AppState(stopped=True), gr.Audio(recording=False)), None,
                [state, input_audio], cancels=[respond, restart])


demo.launch()