import gradio as gr
import os
import time
import gradio as gr
from typing import List, Optional, Tuple, Dict
#from FT.FT_infer_lora import infer_pt
from swift.llm import (PtEngine, RequestConfig, AdapterRequest, get_template, BaseArguments, InferRequest,
                           safe_snapshot_download)


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

History = List[Tuple[str, str]] # 每条消息应为一个包含两个元素（query，response）的元组

def infer_pt(query: Optional[str], chat_history: History):
    
    adapter_path = r"output\v20-20241230-225736\checkpoint-10"
    args = BaseArguments.from_pretrained(adapter_path)
    engine = PtEngine(args.model, adapters=[adapter_path])
    template = get_template(args.template, engine.tokenizer, args.system)
    request_config = RequestConfig(max_tokens=512, temperature=0)
    infer_request = InferRequest(messages=[{'role': 'user', 'content': query}])
    # use lora
    resp_list = engine.infer([infer_request], request_config, template=template)
    response_lora = resp_list[0].choices[0].message.content
    #print(f'lora-response: {response_lora}')
    chat_history.append((query, response_lora))    
    # for resp in chat_history[-1][1]:  # 打印最后一行聊天记录的response，逐字显示
    #     print(resp, end='', flush=True)
    #     time.sleep(0.05)
    
    yield "", chat_history
 
    

# 定义页面总体布局
css = """.my-group {max-width: 600px !important; max-height: 600 !important;}
                     .my-column {display: flex !important; justify-content: center !important; align-items: center !important};"""
with gr.Blocks(css=css) as demo:
    #gr.Markdown(value=f"<center><font size=8>PXD的助手（SFT on Qwen2.5-0.5B-Instruct)</center>")\
    gr.HTML(
        """
    <h1 style='text-align: center'
    >PXD的医疗助手
    </h1>
    """
    )
    gr.HTML(
    """
    <h3 style="text-align: center">
    <a href="https://github.com/nors1993/Transformers_Gradio_PXD" target="_blank">Made by PXD</a>
    </h3>
    """
    )
    with gr.Row():
        chatbot = gr.Chatbot(label="机器人医生")  # 创建一个聊天机器人，包含query和response（即chat_history）
    with gr.Row():
        user_input = gr.Textbox(label="请在下方输入您的问题：", placeholder="请输入您的问题")
    with gr.Row():
        with gr.Column():
            submit_button = gr.Button("🥳提交")
        with gr.Column():
            clear_button = gr.Button("🧹清空历史记录")

    # inputs=[user_input, chatbot]表示infer_pt函数的输入参数为user_input和chatbot
    # outputs=[user_input, chatbot]表示infer_pt函数的输出参数为user_input和chatbot
    submit_button.click(fn=infer_pt, inputs=[user_input, chatbot], outputs=[user_input, chatbot])  # 点击按钮提交
    user_input.submit(fn=infer_pt, inputs=[user_input, chatbot], outputs=[user_input, chatbot])  # 用户按回车键也可以提交
    clear_button.click(lambda: None, None, chatbot)

demo.launch()