import os
os.system('pip install dashscope')
import gradio as gr
from http import HTTPStatus
import dashscope
from dashscope import Generation
from dashscope.api_entities.dashscope_response import Role
from typing import List, Optional, Tuple, Dict
from urllib.error import HTTPError
default_system = 'You are a helpful assistant.'

YOUR_API_TOKEN = os.getenv('YOUR_API_TOKEN')
dashscope.api_key = YOUR_API_TOKEN

History = List[Tuple[str, str]]
Messages = List[Dict[str, str]]

def clear_session() -> History:
    return '', []

def modify_system_session(system: str) -> str:
    if system is None or len(system) == 0:
        system = default_system
    return system, system, []

def history_to_messages(history: History, system: str) -> Messages:
    messages = [{'role': Role.SYSTEM, 'content': system}]
    for h in history:
        messages.append({'role': Role.USER, 'content': h[0]})
        messages.append({'role': Role.ASSISTANT, 'content': h[1]})
    return messages


def messages_to_history(messages: Messages) -> Tuple[str, History]:
    assert messages[0]['role'] == Role.SYSTEM
    system = messages[0]['content']
    history = []
    for q, r in zip(messages[1::2], messages[2::2]):
        history.append([q['content'], r['content']])
    return system, history


def model_chat(query: Optional[str], history: Optional[History], system: str
) -> Tuple[str, str, History]:
    if query is None:
        query = ''
    if history is None:
        history = []
    messages = history_to_messages(history, system)
    messages.append({'role': Role.USER, 'content': query})
    gen = Generation.call(
        model='qwen1.5-110b-chat',
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


def launch():
    with gr.Blocks() as demo:
        gr.Markdown("""<center><font size=8>Qwen1.5-110B-Chat Bot Preview👾</center>""")
    
        with gr.Row():
            with gr.Column(scale=3):
                system_input = gr.Textbox(value=default_system, lines=1, label='System')
            with gr.Column(scale=1):
                modify_system = gr.Button("🛠️ 设置system并清除历史对话", scale=2)
            system_state = gr.Textbox(value=default_system, visible=False)
        chatbot = gr.Chatbot(label='Qwen1.5-110B-Chat')
        textbox = gr.Textbox(lines=2, label='Input')
    
        with gr.Row():
            clear_history = gr.Button("🧹 清除历史对话")
            sumbit = gr.Button("🚀 发送")
    
        sumbit.click(model_chat,
                     inputs=[textbox, chatbot, system_state],
                     outputs=[textbox, chatbot, system_input],
                     concurrency_limit = 1)
        clear_history.click(fn=clear_session,
                            inputs=[],
                            outputs=[textbox, chatbot],
                            concurrency_limit = 1)
        modify_system.click(fn=modify_system_session,
                            inputs=[system_input],
                            outputs=[system_state, system_input, chatbot],
                            concurrency_limit = 1)
    return demo
    # demo.queue(api_open=False,default_concurrency_limit=1)
    # demo.launch(max_threads=1)
