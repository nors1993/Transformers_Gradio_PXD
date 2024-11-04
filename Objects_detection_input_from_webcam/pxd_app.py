import gradio as gr
import cv2
from huggingface_hub import hf_hub_download
from gradio_webrtc import WebRTC
from twilio.rest import Client
import os
from pxd_inference import YOLOv10

model_file = hf_hub_download(
    repo_id="onnx-community/yolov10n", filename="onnx/model.onnx"
)

model = YOLOv10(model_file)

account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
auth_token = os.environ.get("TWILIO_AUTH_TOKEN")

if account_sid and auth_token:
    client = Client(account_sid, auth_token) #创建TWILO客户端，用于实时通信
    token = client.tokens.create()  #生成新的访问令牌

    rtc_configuration = {
        "iceServers": token.ice_servers,  #获取可用的ice（Interactive Connectivity Establishment servers）服务器
        "iceTransportPolicy": "relay",  #采用中继的方式
    }
else:
    rtc_configuration = None

def detection(image, conf_threshold):
    image = cv2.resize(image, (model.input_width, model.input_height))
    new_image = model.detect_objects(image, conf_threshold)  #检测标注置信度、类别后的图片
    return cv2.resize(new_image, (800, 800))

#定义页面总体布局
css = """.my-group {max-width: 600px !important; max-height: 600 !important;}
                     .my-column {display: flex !important; justify-content: center !important; align-items: center !important};"""
with gr.Blocks(css=css) as demo:
    gr.HTML(
        """
    <h1 style='text-align: center'
    >基于网络摄像头视频流的实时目标检测
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
    # my-colum和my-group为gr中的固定参数名称，不可随意更改名称
    with gr.Column(elem_classes=["my-column"]):
        with gr.Group(elem_classes=["my-group"]):
            image = WebRTC(label="视频流", rtc_configuration=rtc_configuration)
            conf_threshold = gr.Slider(
                label="置信度阈值", #输出目标置信度阈值，即满足该阈值阈值才会输出目标类别
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.50 #置信度阈值默认值是0.50
            )
        image.stream(
            fn=detection, inputs=[image, conf_threshold], outputs=[image], time_limit= 3600
        )  #视频流时间3600s后结束输入


if __name__ == "__main__":
    demo.launch()

