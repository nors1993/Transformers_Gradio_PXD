# https://modelscope.cn/studios/Qwen/Qwen2.5/summary
# app.py
import gradio as gr
import os

import gradio as gr
import modelscope_studio as mgr  
from http import HTTPStatus 
import dashscope 
from dashscope import Generation
from dashscope.api_entities.dashscope_response import Role
from typing import List, Optional, Tuple, Dict
from urllib.error import HTTPError

default_system = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'

YOUR_API_TOKEN = os.getenv('YOUR_API_TOKEN')
dashscope.api_key = YOUR_API_TOKEN

History = List[Tuple[str, str]] 
Messages = List[Dict[str, str]] 

# 这些分隔符用于解析和渲染 LaTeX 数学公式，确保它们在不同的环境中正确显示。
latex_delimiters = [{
    "left": "\\(", 
    "right": "\\)",
    "display": True
}, {
    "left": "\\begin\{equation\}",
    "right": "\\end\{equation\}",
    "display": True
}, {
    "left": "\\begin\{align\}",
    "right": "\\end\{align\}",
    "display": True
}, {
    "left": "\\begin\{alignat\}",
    "right": "\\end\{alignat\}",
    "display": True
}, {
    "left": "\\begin\{gather\}",
    "right": "\\end\{gather\}",
    "display": True
}, {
    "left": "\\begin\{CD\}",
    "right": "\\end\{CD\}",
    "display": True
}, {
    "left": "\\[",
    "right": "\\]",
    "display": True
}]

def clear_session() -> History:
    return '', []


def modify_system_session(system: str) -> str:
    if system is None or len(system) == 0:
        system = default_system
    return system, system, []


def history_to_messages(history: History, system: str) -> Messages:
    messages = [{'role': Role.SYSTEM, 'content': system}]
    for h in history:
        messages.append({'role': Role.USER, 'content': h[0].text})
        messages.append({'role': Role.ASSISTANT, 'content': h[1].text})
    return messages


def messages_to_history(messages: Messages) -> Tuple[str, History]:
    assert messages[0]['role'] == Role.SYSTEM
    system = messages[0]['content']
    history = []
    for q, r in zip(messages[1::2], messages[2::2]):
        history.append([q['content'], r['content']])
    return system, history


def model_chat(query: Optional[str], history: Optional[History], system: str, radio: str
               ) -> Tuple[str, str, History, str]:
    if query is None:
        query = ''
    if history is None:
        history = []
    messages = history_to_messages(history, system)
    messages.append({'role': Role.USER, 'content': query})
    
    label_model = f"qwen2.5-{radio.lower()}-instruct"
    
    gen = Generation.call(
        model=label_model,
        messages=messages,
        result_format='message',
        stream=True
    )
    for response in gen:
        if response.status_code == HTTPStatus.OK:
            role = response.output.choices[0].message.role
            response = response.output.choices[0].message.content
            system, history = messages_to_history(messages + [{'role': role, 'content': response}])
            yield '', history, system
        else:
            raise ValueError('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))


def chiose_radio(radio, system):
    mark_ = gr.Markdown(value=f"<center><font size=8>Qwen2.5-{radio}-instruct👾</center>")
    chatbot = mgr.Chatbot(label=f'{radio.lower()}') # 选择对应的模型
    
    if system is None or len(system) == 0:
        system = default_system
    
    return mark_, chatbot, system, system, ""


def update_other_radios(value, other_radio1, other_radio2): 
    if value == "":
        if other_radio1 != "":
            selected = other_radio1
        else:
            selected = other_radio2
        return selected, other_radio1, other_radio2
    return value, "", "" 


def main():
    # 创建两个标签
    with gr.Blocks() as demo:
        gr.Markdown("""<center><font size=8>Qwen2.5: A Party of Foundation Models!</center>""")
        with gr.Row():
            options_1 = ["72B", "32B", "14B", "7B", "3B", "1.5B", "0.5B"]
            options_math = ["Math-72B", "Math-7B", "Math-1.5B"]
            options_coder = ["Coder-7B", "Coder-1.5B"]
            with gr.Row():
                radio1 = gr.Radio(choices=options_1, label="Qwen2.5：", value=options_1[0])
            with gr.Row():
                radio2 = gr.Radio(choices=options_math, label="Qwen2.5-Math：")
            with gr.Row():
                radio3 = gr.Radio(choices=options_coder, label="Qwen2.5-Coder：")
        
        radio = gr.Radio(value=options_1[0], visible=False)
        radio1.change(fn=update_other_radios, inputs=[radio1, radio2, radio3], outputs=[radio, radio2, radio3])
        radio2.change(fn=update_other_radios, inputs=[radio2, radio1, radio3], outputs=[radio, radio1, radio3])
        radio3.change(fn=update_other_radios, inputs=[radio3, radio1, radio2], outputs=[radio, radio1, radio2])
        
        with gr.Row():
            with gr.Accordion():   # 创建一个折叠面板
                mark_ = gr.Markdown("""<center><font size=8>Qwen2.5-72B-instruct 👾</center>""")
                with gr.Row():
                    with gr.Column(scale=3):
                        system_input = gr.Textbox(value=default_system, lines=1, label='System')
                    with gr.Column(scale=1):
                        modify_system = gr.Button("🛠️ Set system prompt and clear history", scale=2)
                    system_state = gr.Textbox(value=default_system, visible=False)
                chatbot = mgr.Chatbot(label=options_1[0].lower(), latex_delimiters=latex_delimiters)
                textbox = gr.Textbox(lines=1, label='Input')
                
                with gr.Row():
                    clear_history = gr.Button("🧹 Clear history")
                    sumbit = gr.Button("🚀 Send")
                
                textbox.submit(model_chat,
                               inputs=[textbox, chatbot, system_state, radio],
                               outputs=[textbox, chatbot, system_input])
                
                sumbit.click(model_chat,
                             inputs=[textbox, chatbot, system_state, radio],
                             outputs=[textbox, chatbot, system_input],
                             concurrency_limit=5)
                clear_history.click(fn=clear_session,
                                    inputs=[],
                                    outputs=[textbox, chatbot])
                modify_system.click(fn=modify_system_session,
                                    inputs=[system_input],
                                    outputs=[system_state, system_input, chatbot])
        
        radio.change(chiose_radio,
                     inputs=[radio, system_input],
                     outputs=[mark_, chatbot, system_state, system_input, textbox])
    
    demo.queue(api_open=False,default_concurrency_limit=40)
    demo.launch(max_threads=5)


if __name__ == "__main__":
    main()