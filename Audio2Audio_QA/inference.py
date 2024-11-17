import os
import lightning as L  # 引入 Lightning 框架，用于简化 PyTorch 的训练过程和模型管理。
import torch
import time
from snac import SNAC
from litgpt import Tokenizer
from litgpt.utils import (
    num_parameters,
)

"""
从 litgpt.generate.base 模块中导入了一系列生成函数，具体如下：

导入的函数解析
generate_AA:
用于生成音频与音频的配对(Audio-Audio)
generate_ASR:
用于自动语音识别(Automatic Speech Recognition)，将音频输入转化为文本输出。
generate_TA:
用于生成文本到音频的转换(Text-to-Audio)，即根据给定的文本生成相应的音频。
generate_TT:
用于文本到文本的生成(Text-to-Text)，例如文本的重写或扩展。
generate_AT:
用于生成音频到文本的转换(Audio-to-Text)，即将音频内容转化为文本。
generate_TA_BATCH:
用于批量处理文本到音频的生成，适用于同时处理多个输入文本。
next_token_batch:
用于生成下一个标记的函数，通常在序列生成任务中使用，如文本生成或语言模型推理。

"""
from litgpt.generate.base import (
    generate_AA,
    generate_ASR,
    generate_TA,
    generate_TT,
    generate_AT,
    generate_TA_BATCH,
    next_token_batch
)
import soundfile as sf
from litgpt.model import GPT, Config
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from utils.snac_utils import layershift, reconscruct_snac, reconstruct_tensors, get_time_str
from utils.snac_utils import get_snac, generate_audio_data
import whisper  # 导入 OpenAI 的 Whisper 模型，用于语音识别。
from tqdm import tqdm  # 导入 tqdm，用于显示进度条，方便监控长时间运行的任务。
from huggingface_hub import snapshot_download

# 设置张量的打印格式，当设置sci_mode为 False 时，PyTorch 将以常规数字格式输出张量的值，而不是使用科学计数法。这对于调试或查看张量内容时，尤其是在值较小或较大时，能提高可读性。
torch.set_printoptions(sci_mode=False)


# 文本词汇表和音频词汇表的大小
text_vocabsize = 151936
text_specialtokens = 64
audio_vocabsize = 4096
audio_specialtokens = 64

padded_text_vocabsize = text_vocabsize + text_specialtokens
padded_audio_vocabsize = audio_vocabsize + audio_specialtokens

"""
_eot: 表示“结束标记”(End Of Text)，通常用于指示文本序列的结束。
_pad_t: 表示“填充标记”，用于填充序列到固定长度。
_input_t: 表示“输入标记”，指示输入的开始。
_answer_t: 表示“答案标记”，用于指示输出答案的开始。
_asr: 表示“自动语音识别标记”，可能用于处理语音输入。
"""
_eot = text_vocabsize
_pad_t = text_vocabsize + 1
_input_t = text_vocabsize + 2
_answer_t = text_vocabsize + 3
_asr = text_vocabsize + 4

"""
_eoa: 表示“结束标记”(End Of Audio)，指示音频序列的结束。
_pad_a: 表示“填充标记”。
_input_a: 表示“输入标记”。
_answer_a: 表示“答案标记”。
_split: 表示用于音频处理的“分隔标记”。
"""

_eoa = audio_vocabsize
_pad_a = audio_vocabsize + 1
_input_a = audio_vocabsize + 2
_answer_a = audio_vocabsize + 3
_split = audio_vocabsize + 4

# 将输入文本转换为适合于文本到音频生成任务的输入 ID 格式。
def get_input_ids_TA(text, text_tokenizer):
    input_ids_item = [[] for _ in range(8)]  # 创建一个包含 8 个空列表的列表，用于存储不同格式的输入 ID。
    text_tokens = text_tokenizer.encode(text)
    for i in range(7):
        # 前7个id的每个id为“填充+回答”标记的组合，len() + 2 代表包含开始和结束标记？
        input_ids_item[i] = [layershift(_pad_a, i)] * (len(text_tokens) + 2) + [
            layershift(_answer_a, i)
        ]
        input_ids_item[i] = torch.tensor(input_ids_item[i]).unsqueeze(0)  # 每个id转化为tensor并在0维上增加一个维度
    # 最后一个id为“输入+token+结束+答案”标记的组合
    input_ids_item[-1] = [_input_t] + text_tokens.tolist() + [_eot] + [_answer_t]
    input_ids_item[-1] = torch.tensor(input_ids_item[-1]).unsqueeze(0)
    # 返回包含 8 个张量的列表，用于后续的模型输入。
    return input_ids_item    # 前7维度是音频，最后一维是文本

# 将输入文本转换为适合文本到文本生成任务的输入 ID 格式。
def get_input_ids_TT(text, text_tokenizer):
    input_ids_item = [[] for i in range(8)]
    text_tokens = text_tokenizer.encode(text).tolist()

    for i in range(7):
        input_ids_item[i] = torch.tensor(
            [layershift(_pad_a, i)] * (len(text_tokens) + 3)  # _pad_a还是_pad_t？
        ).unsqueeze(0)
    input_ids_item[-1] = [_input_t] + text_tokens + [_eot] + [_answer_t]
    input_ids_item[-1] = torch.tensor(input_ids_item[-1]).unsqueeze(0)

    return input_ids_item

"""
mel: 这是一个表示梅尔频谱图的输入，通常用于音频数据的表示，Whisper 模型利用这种表示进行语音识别。
leng: 这是输入的长度，通常用于指示梅尔频谱图的时间步数或帧数。
whispermodel: 这是 Whisper 模型的实例，负责处理输入并生成输出。
device: 指定计算设备（如 CPU 或 GPU），确保模型和数据在相同的设备上进行计算。
special_token_a: 默认值为 _answer_a，用于表示特定的音频输入标记，在模型中用于标记答案或输出的开始。
special_token_t: 默认值为 _answer_t，用于表示特定的文本输入标记，在模型中用于标记文本输出的开始。
"""
def get_input_ids_whisper(
    mel, leng, whispermodel, device, 
    special_token_a=_answer_a, special_token_t=_answer_t,
):

    """
    将梅尔频谱图转换为音频特征，并生成与 Whisper 模型兼容的输入 ID 列表。
    """
    with torch.no_grad():
        mel = mel.unsqueeze(0).to(device)
        # audio_feature = whisper.decode(whispermodel,mel, options).audio_features
        audio_feature = whispermodel.embed_audio(mel)[0][:leng]  # 将梅尔频谱图使用whisper模型转化为音频特征，去除第一个元素，截取leng的长度

    T = audio_feature.size(0)  # 获取音频特征时间步长
    input_ids = []
    for i in range(7):
        input_ids_item = []
        input_ids_item.append(layershift(_input_a, i))
        input_ids_item += [layershift(_pad_a, i)] * T
        input_ids_item += [(layershift(_eoa, i)), layershift(special_token_a, i)]  # 添加结束音频标记和特殊字符标记
        input_ids.append(torch.tensor(input_ids_item).unsqueeze(0))
    input_id_T = torch.tensor([_input_t] + [_pad_t] * T + [_eot, special_token_t])
    input_ids.append(input_id_T.unsqueeze(0))
    return audio_feature.unsqueeze(0), input_ids  # 返回音频特征并在第一维度上添加一个维度，返回输入ID列表


def get_input_ids_whisper_ATBatch(mel, leng, whispermodel, device):
    with torch.no_grad():
        mel = mel.unsqueeze(0).to(device)  # unsqueeze(0) 在 mel 的第一个维度上增加一个维度。这通常用于将单个样本的输入数据调整为批处理（batch）的形式，使得模型能够接受。
        # audio_feature = whisper.decode(whispermodel,mel, options).audio_features
        audio_feature = whispermodel.embed_audio(mel)[0][:leng]
    T = audio_feature.size(0)
    input_ids_AA = []
    for i in range(7):
        input_ids_item = []
        input_ids_item.append(layershift(_input_a, i))
        input_ids_item += [layershift(_pad_a, i)] * T
        input_ids_item += [(layershift(_eoa, i)), layershift(_answer_a, i)]
        input_ids_AA.append(torch.tensor(input_ids_item))
    input_id_T = torch.tensor([_input_t] + [_pad_t] * T + [_eot, _answer_t])
    input_ids_AA.append(input_id_T)

    input_ids_AT = []
    for i in range(7):
        input_ids_item = []
        input_ids_item.append(layershift(_input_a, i))
        input_ids_item += [layershift(_pad_a, i)] * T
        input_ids_item += [(layershift(_eoa, i)), layershift(_pad_a, i)]
        input_ids_AT.append(torch.tensor(input_ids_item))
    input_id_T = torch.tensor([_input_t] + [_pad_t] * T + [_eot, _answer_t])
    input_ids_AT.append(input_id_T)

    input_ids = [input_ids_AA, input_ids_AT]
    stacked_inputids = [[] for _ in range(8)]
    for i in range(2):
        for j in range(8):
            stacked_inputids[j].append(input_ids[i][j])
    
    # 对 stacked_inputids 中的每个列表应用 torch.stack，将其转换为张量。这将把每个层次的输入 ID 堆叠到一个新的张量中。
    stacked_inputids = [torch.stack(tensors) for tensors in stacked_inputids]  
    # 使用 torch.stack 将 audio_feature 堆叠两次，返回一个包含两个相同音频特征的张量。
    # 返回合并后的输入 ID 张量 stacked_inputids。
    return torch.stack([audio_feature, audio_feature]), stacked_inputids  


def load_audio(path):
    audio = whisper.load_audio(path)
    duration_ms = (len(audio) / 16000) * 1000
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio)
    return mel, int(duration_ms / 20) + 1    # 每 20 毫秒生成一个帧

# 音频生成文本
def A1_A2_batch(fabric, audio_feature, input_ids, leng, model, text_tokenizer, step,
                snacmodel, out_dir=None):
    with fabric.init_tensor():  # 用 fabric.init_tensor() 上下文管理器初始化张量环境
        model.set_kv_cache(batch_size=2)  # 设置键值缓存的批次大小为 2，以优化模型在处理多个样本时的性能。
    tokenlist = generate_TA_BATCH(
        model,
        audio_feature,
        input_ids,
        [leng, leng],
        ["A1A2", "A1T2"],
        max_returned_tokens=2048,
        temperature=0.9,
        top_k=1,
        eos_id_a=_eoa,
        eos_id_t=_eot,
        pad_id_t=_pad_t,
        shift=padded_text_vocabsize,
        include_prompt=True,
        generate_text=True,
    )
    text_tokenlist = tokenlist[-1]   # 获取生成文本的token列表
    if text_vocabsize in text_tokenlist:
        text_tokenlist = text_tokenlist[: text_tokenlist.index(text_vocabsize)]  # 截断0：text_vocabsize之间的文本token
    text = text_tokenizer.decode(torch.tensor(text_tokenlist)).strip()  # 将token解码为文字，strip() 方法用于移除字符串开头和结尾的空白字符（如空格、换行符等）。

    audio_tokenlist = tokenlist[:-1]
    audiolist = reconscruct_snac(audio_tokenlist)  # 将音频 token 列表转换为音频数据的内部表示（如音频片段或特征）
    audio = reconstruct_tensors(audiolist)  # 将音频列表转换为 PyTorch 张量。为了确保音频数据可以被模型处理。
    if out_dir is None:
        out_dir = "./output/default/A1-A2-batch"
    else:
        out_dir = out_dir + "/A1-A2-batch"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with torch.inference_mode():
        audio_hat = snacmodel.decode(audio)  # 调用 SNAC 模型的解码方法，将处理后的音频张量转换为可用的音频数据（如音频波形）。
    """
    udio_hat 是生成的音频张量。
    squeeze() 方法用于移除张量中维度为 1 的维度，以确保音频数据的形状适合写入文件。
    cpu() 将张量从 GPU 移动到 CPU（如果它是在 GPU 上的），因为 soundfile 不支持直接操作 GPU 张量。
    numpy() 将 PyTorch 张量转换为 NumPy 数组，以便 soundfile 可以处理。
    采样率设置为 24000 Hz（24 kHz）
    """
    sf.write(
        f"{out_dir}/{step:02d}.wav",  # 格式化输出格式，02d表示两位数如01，02，10等
        audio_hat.squeeze().cpu().numpy(), #
        24000,
    )
    model.clear_kv_cache()  # 清除模型缓存
    return text


def A1_T2(fabric, audio_feature, input_ids, leng, model, text_tokenizer, step):
    with fabric.init_tensor():
        model.set_kv_cache(batch_size=1)
    tokenlist = generate_AT(
        model,
        audio_feature,
        input_ids,
        [leng],
        ["AT"],
        max_returned_tokens=2048,
        temperature=0.9,
        top_k=1,
        eos_id_a=_eoa,
        eos_id_t=_eot,
        pad_id_t=_pad_t,
        shift=padded_text_vocabsize,
        include_prompt=True,
        generate_text=True,
    )
    return text_tokenizer.decode(torch.tensor(tokenlist)).strip()


def A1_A2(fabric, audio_feature, input_ids, leng, model, text_tokenizer, step,
          snacmodel, out_dir=None):
    with fabric.init_tensor():
        model.set_kv_cache(batch_size=1)
    tokenlist = generate_AA(
        model,
        audio_feature,
        input_ids,
        [leng],
        ["A1T2"],   
        max_returned_tokens=2048,
        temperature=0.9,
        top_k=1,
        eos_id_a=_eoa,
        eos_id_t=_eot,
        pad_id_t=_pad_t,
        shift=padded_text_vocabsize,
        include_prompt=True,
        generate_text=True,
    )
    audiolist = reconscruct_snac(tokenlist)   # 将音频 token 列表转换为音频数据的内部表示（如音频片段或特征）
    tokenlist = tokenlist[-1]
    if text_vocabsize in tokenlist:
        tokenlist = tokenlist[: tokenlist.index(text_vocabsize)]
    if out_dir is None:
        out_dir = "./output/default/A1-A2"
    else:
        out_dir = out_dir + "/A1-A2"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    audio = reconstruct_tensors(audiolist)
    with torch.inference_mode():
        audio_hat = snacmodel.decode(audio)
    sf.write(
        f"{out_dir}/{step:02d}.wav",
        audio_hat.squeeze().cpu().numpy(),
        24000,
    )
    model.clear_kv_cache()
    return text_tokenizer.decode(torch.tensor(tokenlist)).strip()


def A1_T1(fabric, audio_feature, input_ids, leng, model, text_tokenizer, step):
    with fabric.init_tensor():
        model.set_kv_cache(batch_size=1)
    tokenlist = generate_ASR(
        model,
        audio_feature,
        input_ids,
        [leng],
        ["A1T1"],
        max_returned_tokens=2048,
        temperature=0.9,
        top_k=1,
        eos_id_a=_eoa,
        eos_id_t=_eot,
        pad_id_t=_pad_t,
        shift=padded_text_vocabsize,
        include_prompt=True,
        generate_text=True,
    )
    model.clear_kv_cache()
    return text_tokenizer.decode(torch.tensor(tokenlist)).strip()


def T1_A2(fabric, input_ids, model, text_tokenizer, step,
          snacmodel, out_dir=None):
    with fabric.init_tensor():
        model.set_kv_cache(batch_size=1)
    tokenlist = generate_TA(
        model,
        None,
        input_ids,
        None,
        ["T1A2"],
        max_returned_tokens=2048,
        temperature=0.9,
        top_k=1,
        eos_id_a=_eoa,
        eos_id_t=_eot,
        pad_id_t=_pad_t,
        shift=padded_text_vocabsize,
        include_prompt=True,
        generate_text=True,
    )

    audiolist = reconscruct_snac(tokenlist)
    tokenlist = tokenlist[-1]

    if text_vocabsize in tokenlist:
        tokenlist = tokenlist[: tokenlist.index(text_vocabsize)]
    audio = reconstruct_tensors(audiolist)
    if out_dir is None:
        out_dir = "./output/default/T1-A2"
    else:
        out_dir = out_dir + "/T1-A2"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with torch.inference_mode():
        audio_hat = snacmodel.decode(audio)
    sf.write(
        f"{out_dir}/{step:02d}.wav",
        audio_hat.squeeze().cpu().numpy(),
        24000,
    )
    model.clear_kv_cache()
    return text_tokenizer.decode(torch.tensor(tokenlist)).strip()


def T1_T2(fabric, input_ids, model, text_tokenizer, step):

    with fabric.init_tensor():
        model.set_kv_cache(batch_size=1)
    tokenlist = generate_TT(
        model,
        None,
        input_ids,
        None,
        ["T1T2"],
        max_returned_tokens=2048,
        temperature=0.9,
        top_k=1,
        eos_id_a=_eoa,
        eos_id_t=_eot,
        pad_id_t=_pad_t,
        shift=padded_text_vocabsize,
        include_prompt=True,
        generate_text=True,
    )
    model.clear_kv_cache()
    return text_tokenizer.decode(torch.tensor(tokenlist)).strip()

    
def load_model(ckpt_dir, device):
    snacmodel = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
    whispermodel = whisper.load_model("small").to(device)
    text_tokenizer = Tokenizer(ckpt_dir)
    fabric = L.Fabric(devices=1, strategy="auto")
    config = Config.from_file(ckpt_dir + "/model_config.yaml")
    config.post_adapter = False
    
    # empty_init=False 表示在初始化模型时，应该根据配置加载参数，而不是使用空权重。设置为 False 可以确保模型的权重从指定的配置中加载。
    with fabric.init_module(empty_init=False):  
        model = GPT(config)  # 使用config中的配置进行模型初始化

    model = fabric.setup(model)
    # 加载模型权重参数文件
    state_dict = lazy_load(ckpt_dir + "/lit_model.pth")
    # strict=True 表示在加载时，要求状态字典的键必须与模型的参数名称严格匹配。如果不匹配，加载将会失败，这是确保模型完整性的一种方式。
    model.load_state_dict(state_dict, strict=True)  
    model.to(device).eval()

    return fabric, model, text_tokenizer, snacmodel, whispermodel

    
def download_model(ckpt_dir):
    repo_id = "gpt-omni/mini-omni"
    snapshot_download(repo_id, local_dir=ckpt_dir, revision="main")

    
class OmniInference:

    def __init__(self, ckpt_dir='./checkpoint', device='cuda:0'):
        self.device = device
        if not os.path.exists(ckpt_dir):
            print(f"checkpoint directory {ckpt_dir} not found, downloading from huggingface")
            download_model(ckpt_dir)
        self.fabric, self.model, self.text_tokenizer, self.snacmodel, self.whispermodel = load_model(ckpt_dir, device)

    def warm_up(self, sample='./data/samples/output1.wav'):
        for _ in self.run_AT_batch_stream(sample):
            pass
    # 装饰器用于指示 PyTorch 进入推理模式。
    @torch.inference_mode()
    def run_AT_batch_stream(self, 
                            audio_path, 
                            stream_stride=4,
                            max_returned_tokens=2048, 
                            temperature=0.9, 
                            top_k=1, 
                            top_p=1.0,
                            eos_id_a=_eoa,
                            eos_id_t=_eot,
        ):

        assert os.path.exists(audio_path), f"audio file {audio_path} not found"
        model = self.model

        with self.fabric.init_tensor():
            model.set_kv_cache(batch_size=2,device=self.device)

        mel, leng = load_audio(audio_path)
        audio_feature, input_ids = get_input_ids_whisper_ATBatch(mel, leng, self.whispermodel, self.device)
        T = input_ids[0].size(1) # 获取第一维度张量中第二维度的大小
        device = input_ids[0].device  # 获取输入数据所在设备，确保模型和数据在同一设备上运行，避免设备不一致出错

        assert max_returned_tokens > T, f"max_returned_tokens {max_returned_tokens} should be greater than audio length {T}"
        
        # 确保模型的最大序列长度大于等于模型的输出长度，减1是因为生成输出时通常需要保留一个位置用于结束标记（e.g., EOS token）。
        if model.max_seq_length < max_returned_tokens - 1:
            raise NotImplementedError(
                f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}"
            )

        input_pos = torch.tensor([T], device=device)
        list_output = [[] for i in range(8)]
        tokens_A, token_T = next_token_batch(
            model,
            audio_feature.to(torch.float32).to(model.device),
            input_ids,
            [T - 3, T - 3],
            ["A1T2", "A1T2"],
            input_pos=torch.arange(0, T, device=device),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        for i in range(7):
            list_output[i].append(tokens_A[i].tolist()[0])
        list_output[7].append(token_T.tolist()[0])

        model_input_ids = [[] for i in range(8)]
        for i in range(7):
            tokens_A[i] = tokens_A[i].clone() + padded_text_vocabsize + i * padded_audio_vocabsize
            model_input_ids[i].append(tokens_A[i].clone().to(device).to(torch.int32))
            model_input_ids[i].append(torch.tensor([layershift(4097, i)], device=device))
            model_input_ids[i] = torch.stack(model_input_ids[i])  # 将 model_input_ids[i] 中的所有张量堆叠成一个单一的张量

        model_input_ids[-1].append(token_T.clone().to(torch.int32))
        model_input_ids[-1].append(token_T.clone().to(torch.int32))
        model_input_ids[-1] = torch.stack(model_input_ids[-1])

        text_end = False
        index = 1
        nums_generate = stream_stride
        begin_generate = False
        current_index = 0
        # 使用tqdm定义进度条，用于显示循环进度
        for _ in tqdm(range(2, max_returned_tokens - T + 1)):
            tokens_A, token_T = next_token_batch(
                model,
                None,
                model_input_ids,
                None,
                None,
                input_pos=input_pos,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            if text_end:
                token_T = torch.tensor([_pad_t], device=device)

            if tokens_A[-1] == eos_id_a:
                break

            if token_T == eos_id_t:
                text_end = True

            for i in range(7):
                list_output[i].append(tokens_A[i].tolist()[0])  # 加上音频的token列表
            list_output[7].append(token_T.tolist()[0])  # 加上文本的token列表

            model_input_ids = [[] for i in range(8)]
            for i in range(7):
                tokens_A[i] = tokens_A[i].clone() +padded_text_vocabsize + i * padded_audio_vocabsize
                model_input_ids[i].append(tokens_A[i].clone().to(device).to(torch.int32))
                model_input_ids[i].append(
                    torch.tensor([layershift(4097, i)], device=device)
                )
                model_input_ids[i] = torch.stack(model_input_ids[i])

            model_input_ids[-1].append(token_T.clone().to(torch.int32))
            model_input_ids[-1].append(token_T.clone().to(torch.int32))
            model_input_ids[-1] = torch.stack(model_input_ids[-1])

            if index == 7:
                begin_generate = True

            if begin_generate:
                current_index += 1
                if current_index == nums_generate:
                    current_index = 0
                    snac = get_snac(list_output, index, nums_generate)
                    audio_stream = generate_audio_data(snac, self.snacmodel, self.device)
                    yield audio_stream

            input_pos = input_pos.add_(1)
            index += 1
        text = self.text_tokenizer.decode(torch.tensor(list_output[-1]))
        print(f"text output: {text}")
        model.clear_kv_cache()
        return list_output

# 以下供测试用
def test_infer():
    device = "cuda:0"
    out_dir = f"./output/{get_time_str()}"  # 将当前时间格式化为字符串放在最后一层目录
    ckpt_dir = f"./checkpoint"
    if not os.path.exists(ckpt_dir):
        print(f"checkpoint directory {ckpt_dir} not found, downloading from huggingface")
        download_model(ckpt_dir)

    fabric, model, text_tokenizer, snacmodel, whispermodel = load_model(ckpt_dir, device)

    task = ['A1A2', 'asr', "T1A2", "AA-BATCH", 'T1T2', 'AT']

    # 准备测试数据
    # os.listdir() 函数列出指定目录（在这里是 ./data/samples）中的所有文件和子目录。返回的结果是一个文件名列表。
    test_audio_list = sorted(os.listdir('./data/samples'))  # 按照文件名的字母顺序排序
    # 将文件名称追加到./data/samples目录后面，比如./data/samples/audio1.wav
    test_audio_list = [os.path.join('./data/samples', path) for path in test_audio_list]
    test_audio_transcripts = [
        "What is your name?",
        "what are your hobbies?",
        "Do you like beijing?",
        "How are you feeling today?",
        "what is the weather like today?",
    ]
    test_text_list = [
        "What is your name?",
        "How are you feeling today?",
        "Can you describe your surroundings?",
        "What did you do yesterday?",
        "What is your favorite book and why?",
        "How do you make a cup of tea?",
        "What is the weather like today?",
        "Can you explain the concept of time?",
        "Can you tell me a joke?",
    ]

    # 加载模型
    with torch.no_grad():
        if "A1A2" in task:
            print("===============================================================")
            print("                       testing A1A2")
            print("===============================================================")
            step = 0
            for path in test_audio_list:
                try:
                    mel, leng = load_audio(path)
                    audio_feature, input_ids = get_input_ids_whisper(mel, leng, whispermodel, device)
                    text = A1_A2(
                        fabric,
                        audio_feature,
                        input_ids,
                        leng,
                        model,
                        text_tokenizer,
                        step,
                        snacmodel,
                        out_dir=out_dir,
                    )
                    print(f"input: {test_audio_transcripts[step]}")
                    print(f"output: {text}")
                    step += 1
                    print(
                        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
                    )
                except:
                    print(f"[error] failed to process {path}")
            print("===============================================================")

        if 'asr' in task:
            print("===============================================================")
            print("                       testing asr")
            print("===============================================================")

            index = 0
            step = 0
            for path in test_audio_list:
                mel, leng = load_audio(path)
                audio_feature, input_ids = get_input_ids_whisper(mel, leng, whispermodel, device, special_token_a=_pad_a, special_token_t=_asr)
                output = A1_T1(fabric, audio_feature, input_ids ,leng, model, text_tokenizer, index).lower().replace(',','').replace('.','').replace('?','')
                print(f"audio_path: {path}")
                print(f"audio transcript: {test_audio_transcripts[index]}")
                print(f"asr output: {output}")
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                index += 1

        if "T1A2" in task:
            step = 0
            print("\n")
            print("===============================================================")
            print("                       testing T1A2")
            print("===============================================================")
            for text in test_text_list:
                input_ids = get_input_ids_TA(text, text_tokenizer)
                text_output = T1_A2(fabric, input_ids, model, text_tokenizer, step,
                                    snacmodel, out_dir=out_dir)
                print(f"input: {text}")
                print(f"output: {text_output}")
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                step += 1
            print("===============================================================")

        if "T1T2" in task:
            step = 0
            print("\n")
            print("===============================================================")
            print("                       testing T1T2")
            print("===============================================================")

            for text in test_text_list:
                input_ids = get_input_ids_TT(text, text_tokenizer)
                text_output = T1_T2(fabric, input_ids, model, text_tokenizer, step)
                print(f" Input: {text}")
                print(f"Output: {text_output}")
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("===============================================================")

        if "AT" in task:
            print("===============================================================")
            print("                       testing A1T2")
            print("===============================================================")
            step = 0
            for path in test_audio_list:
                mel, leng = load_audio(path)
                audio_feature, input_ids = get_input_ids_whisper(
                    mel, leng, whispermodel, device, 
                    special_token_a=_pad_a, special_token_t=_answer_t
                )
                text = A1_T2(
                    fabric, audio_feature, input_ids, leng, model, text_tokenizer, step
                )
                print(f"input: {test_audio_transcripts[step]}")
                print(f"output: {text}")
                step += 1
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("===============================================================")

        if "AA-BATCH" in task:
            print("===============================================================")
            print("                       testing A1A2-BATCH")
            print("===============================================================")
            step = 0
            for path in test_audio_list:
                mel, leng = load_audio(path)
                audio_feature, input_ids = get_input_ids_whisper_ATBatch(mel, leng, whispermodel, device)
                text = A1_A2_batch(
                    fabric, audio_feature, input_ids, leng, model, text_tokenizer, step,
                    snacmodel, out_dir=out_dir
                )
                print(f"input: {test_audio_transcripts[step]}")
                print(f"output: {text}")
                step += 1
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("===============================================================")

        print("*********************** test end *****************************")



if __name__ == "__main__":
    test_infer()