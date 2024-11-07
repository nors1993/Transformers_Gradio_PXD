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
from utils.vad import get_speech_timestamps, collect_chunks, VadOptions
import tempfile


from server import serve

repo_id = "gpt-omni/mini-omni"
snapshot_download(repo_id, local_dir="./checkpoint", revision="main")

IP = "0.0.0.0"
PORT = 60808

thread = Thread(target=serve, daemon=True)
thread.start()

API_URL = "http://0.0.0.0:60808/chat"

# recording parameters
IN_CHANNELS = 1
IN_RATE = 24000
IN_CHUNK = 1024
IN_SAMPLE_WIDTH = 2
VAD_STRIDE = 0.5

# playing parameters
OUT_CHANNELS = 1
OUT_RATE = 24000
OUT_SAMPLE_WIDTH = 2
OUT_CHUNK = 5760


OUT_CHUNK = 20 * 4096
OUT_RATE = 24000
OUT_CHANNELS = 1


def run_vad(ori_audio, sr):
    _st = time.time()
    try:
        audio = ori_audio
        audio = audio.astype(np.float32) / 32768.0
        sampling_rate = 16000
        if sr != sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)

        vad_parameters = {}
        vad_parameters = VadOptions(**vad_parameters)
        speech_chunks = get_speech_timestamps(audio, vad_parameters)
        audio = collect_chunks(audio, speech_chunks)
        duration_after_vad = audio.shape[0] / sampling_rate

        if sr != sampling_rate:
            # resample to original sampling rate
            vad_audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=sr)
        else:
            vad_audio = audio
        vad_audio = np.round(vad_audio * 32768.0).astype(np.int16)
        vad_audio_bytes = vad_audio.tobytes()

        return duration_after_vad, vad_audio_bytes, round(time.time() - _st, 4)
    except Exception as e:
        msg = f"[asr vad error] audio_len: {len(ori_audio)/(sr*2):.3f} s, trace: {traceback.format_exc()}"
        print(msg)
        return -1, ori_audio, round(time.time() - _st, 4)


def warm_up():
    frames = b"\x00\x00" * 1024 * 2  # 1024 frames of 2 bytes each
    dur, frames, tcost = run_vad(frames, 16000)
    print(f"warm up done, time_cost: {tcost:.3f} s")


warm_up()


@dataclass
class AppState:
    stream: np.ndarray | None = None
    sampling_rate: int = 0
    pause_detected: bool = False
    started_talking: bool =  False
    stopped: bool = False
    conversation: list = field(default_factory=list)


def determine_pause(audio: np.ndarray, sampling_rate: int, state: AppState) -> bool:
    """Take in the stream, determine if a pause happened"""

    temp_audio = audio
    
    dur_vad, _, time_vad = run_vad(temp_audio, sampling_rate)
    duration = len(audio) / sampling_rate

    if dur_vad > 0.5 and not state.started_talking:
        print("started talking")
        state.started_talking = True
        return False

    print(f"duration_after_vad: {dur_vad:.3f} s, time_vad: {time_vad:.3f} s")

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
                    mp3_io = io.BytesIO()
                    audio_segment.export(mp3_io, format="mp3", bitrate="320k")

                    # Get the MP3 bytes
                    mp3_bytes = mp3_io.getvalue()
                    mp3_io.close()
                    yield mp3_bytes

        except Exception as e:
            raise gr.Error(f"Error during audio streaming: {e}")




def process_audio(audio: tuple, state: AppState):
    if state.stream is None:
        state.stream = audio[1]
        state.sampling_rate = audio[0]
    else:
        state.stream =  np.concatenate((state.stream, audio[1]))

    pause_detected = determine_pause(state.stream, state.sampling_rate, state)
    state.pause_detected = pause_detected

    if state.pause_detected and state.started_talking:
        return gr.Audio(recording=False), state
    return None, state


def response(state: AppState):
    if not state.pause_detected and not state.started_talking:
        return None, AppState()
    
    audio_buffer = io.BytesIO()

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
    
    output_buffer = b""

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
        [input_audio, state],
        [input_audio, state],
        stream_every=0.50,
        time_limit=30,
    )
    respond = input_audio.stop_recording(
        response,
        [state],
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