import os
import gradio as gr
#from FT.FT_infer_lora import infer_pt
from swift.llm import (PtEngine, RequestConfig, AdapterRequest, get_template, BaseArguments, InferRequest,
                           safe_snapshot_download)



# import os
# from typing import Literal

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def infer_pt(prompt,chat_history):
    #adapter_path = safe_snapshot_download('swift/test_lora')
    #adapter_path = safe_snapshot_download(r"output\v20-20241230-225736\checkpoint-10")
    adapter_path = r"output\v20-20241230-225736\checkpoint-10"
    args = BaseArguments.from_pretrained(adapter_path)
    engine = PtEngine(args.model, adapters=[adapter_path])
    template = get_template(args.template, engine.tokenizer, args.system)
    request_config = RequestConfig(max_tokens=512, temperature=0)
    infer_request = InferRequest(messages=[{'role': 'user', 'content': prompt}])
    # use lora
    resp_list = engine.infer([infer_request], request_config, template=template)
    response_lora = resp_list[0].choices[0].message.content
    #print(f'lora-response: {response_lora}')
    #return response_lora
    chat_history = []
    chat_history.append((prompt, response_lora))  # 每条消息应为一个包含两个元素的元组,才能传入到gradio的chatbot中
    return chat_history   # 返回chat_history和空字符串，以便清空输入框

#定义页面总体布局
css = """.my-group {max-width: 600px !important; max-height: 600 !important;}
                     .my-column {display: flex !important; justify-content: center !important; align-items: center !important};"""
with gr.Blocks(css=css) as demo:
    gr.HTML(
        """
    <h1 style='text-align: center'
    >Medical Dialogue System (SFT on Qwen2.5-0.5B-Instruct) 
    </h1>
    """
    )
    gr.HTML(
    """
    <h3 style="text-align: center">
    <a href="https://github.com/nors1993" target="_blank">Made by PXD</a>
    </h3>
    """
    )

    with gr.Row():
        with gr.Column():
            #user_input = gr.Textbox("You are a helpful assistant", label="System Prompt")
            #chatbot = gr.Chatbot(label="PXD的助手", elem_classes="control-height")
            chatbot = gr.Chatbot(label="PXD的助手李娜")
            user_input = gr.Textbox(label="用户输入内容", placeholder="请输入您的问题")
            submit_button = gr.Button("提交")
            prompt = user_input
            submit_button.click(fn=infer_pt, inputs=[prompt, chatbot], outputs=[chatbot])  # 点击按钮提交
            prompt.submit(fn=infer_pt, inputs=[prompt], outputs=[chatbot])  # 用户按回车键也可以提交

    demo.launch()


