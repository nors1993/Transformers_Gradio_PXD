#utils/snac_utils.py

import torch
import time
import numpy as np


class SnacConfig:
    audio_vocab_size = 4096
    padded_vocab_size = 4160
    end_of_audio = 4097  # 音频结束标志，最后一个元素为4097


snac_config = SnacConfig()    


def get_time_str():
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return time_str

"""
定义一个名为 layershift 的函数，用于计算一个新的输入 ID，用于在深度学习或信号处理中的层级表示。
input_id: 一个整数，表示输入数据的 ID。
layer: 一个整数，表示当前层的索引或编号。
stride: 默认为 4160，表示层与层之间的步幅或间隔。
shift: 默认为 152000，表示一个基础的偏移量。

return input_id + shift + layer * stride:
这个表达式计算并返回新的输入 ID。具体来说：
input_id: 初始的输入 ID。
shift: 基础偏移量，增加了一个固定的值。
layer * stride: 根据层的索引和步幅计算的额外偏移，用于在不同层之间进行位置调整。
"""
def layershift(input_id, layer, stride=4160, shift=152000):
    return input_id + shift + layer * stride

    
def generate_audio_data(snac_tokens, snacmodel, device=None):
    audio = reconstruct_tensors(snac_tokens, device)  # 将音频转变为适合snac模型的张量
    with torch.inference_mode():
        audio_hat = snacmodel.decode(audio)  # 将音频张量转变为音频数据
    audio_data = audio_hat.cpu().numpy().astype(np.float64) * 32768.0  # 乘以 32768.0 将音频数据缩放到 16 位整数范围
    audio_data = audio_data.astype(np.int16)
    audio_data = audio_data.tobytes()  # 将numpy数据转变为字节串，易于保存
    return audio_data

# 从给定的输出列表中生成 SNAC token
def get_snac(list_output, index, nums_generate):

    snac = []
    start = index
    for i in range(nums_generate):
        snac.append("#")
        for j in range(7):
            snac.append(list_output[j][start - nums_generate - 5 + j + i])  # list_output为2元数组
    return snac

# 重建snac令牌输出列表
def reconscruct_snac(output_list):
    if len(output_list) == 8:
        output_list = output_list[:-1]
    output = []
    for i in range(7):
        output_list[i] = output_list[i][i + 1 :]
    for i in range(len(output_list[-1])):
        output.append("#")
        for j in range(7):
            output.append(output_list[j][i])
    return output


def reconstruct_tensors(flattened_output, device=None):
    """Reconstructs the list of tensors from the flattened output."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def count_elements_between_hashes(lst):
        try:
            # Find the index of the first '#'
            first_index = lst.index("#")
            # Find the index of the second '#' after the first
            second_index = lst.index("#", first_index + 1)
            # Count the elements between the two indices
            return second_index - first_index - 1  # 通过减去 1，排除两个 # 符号本身
        except ValueError:
            # Handle the case where there aren't enough '#' symbols
            return "List does not contain two '#' symbols"

    def remove_elements_before_hash(flattened_list):
        try:
            # Find the index of the first '#'
            first_hash_index = flattened_list.index("#")
            # Return the list starting from the first '#'
            return flattened_list[first_hash_index:]
        except ValueError:
            # Handle the case where there is no '#'
            return "List does not contain the symbol '#'"

    def list_to_torch_tensor(tensor1):
        # Convert the list to a torch tensor
        tensor = torch.tensor(tensor1)
        # Reshape the tensor to have size (1, n)
        tensor = tensor.unsqueeze(0)
        return tensor

    flattened_output = remove_elements_before_hash(flattened_output)
    codes = []
    tensor1 = []
    tensor2 = []
    tensor3 = []
    tensor4 = []

    n_tensors = count_elements_between_hashes(flattened_output)
    if n_tensors == 7:
        for i in range(0, len(flattened_output), 8):

            tensor1.append(flattened_output[i + 1])
            tensor2.append(flattened_output[i + 2])
            tensor3.append(flattened_output[i + 3])
            tensor3.append(flattened_output[i + 4])

            tensor2.append(flattened_output[i + 5])
            tensor3.append(flattened_output[i + 6])
            tensor3.append(flattened_output[i + 7])
            codes = [
                list_to_torch_tensor(tensor1).to(device),
                list_to_torch_tensor(tensor2).to(device),
                list_to_torch_tensor(tensor3).to(device),
            ]

    if n_tensors == 15:
        for i in range(0, len(flattened_output), 16):

            tensor1.append(flattened_output[i + 1])
            tensor2.append(flattened_output[i + 2])
            tensor3.append(flattened_output[i + 3])
            tensor4.append(flattened_output[i + 4])
            tensor4.append(flattened_output[i + 5])
            tensor3.append(flattened_output[i + 6])
            tensor4.append(flattened_output[i + 7])
            tensor4.append(flattened_output[i + 8])

            tensor2.append(flattened_output[i + 9])
            tensor3.append(flattened_output[i + 10])
            tensor4.append(flattened_output[i + 11])
            tensor4.append(flattened_output[i + 12])
            tensor3.append(flattened_output[i + 13])
            tensor4.append(flattened_output[i + 14])
            tensor4.append(flattened_output[i + 15])

            codes = [
                list_to_torch_tensor(tensor1).to(device),
                list_to_torch_tensor(tensor2).to(device),
                list_to_torch_tensor(tensor3).to(device),
                list_to_torch_tensor(tensor4).to(device),
            ]

    return codes