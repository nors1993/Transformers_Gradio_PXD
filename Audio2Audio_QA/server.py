import flask
import base64  # 用于对数据进行 Base64 编码和解码，常用于在 HTTP 请求中传输二进制数据。
import tempfile
import traceback
from flask import Flask, Response, stream_with_context
from inference import OmniInference


class OmniChatServer(object):
    # ip='0.0.0.0': 服务器绑定的 IP 地址。0.0.0.0 表示监听所有可用的网络接口。
    def __init__(self, ip='0.0.0.0', port=60808, run_app=True,
                 ckpt_dir='./checkpoint', device='cuda:0') -> None:
        server = Flask(__name__)  # 创建一个 Flask 应用实例。__name__ 是当前模块的名称，用于 Flask 确定资源的位置。
        # CORS(server, resources=r"/*")  #  CORS（跨域资源共享）的设置，为了允许来自不同源的请求。如果需要支持跨域请求，通常会使用 flask-cors 库。
        # server.config["JSON_AS_ASCII"] = False  # 配置 Flask 的 JSON 响应，设置为 False 可以支持非 ASCII 字符的正确编码，确保中文等字符能正确显示。
        
       
        self.client = OmniInference(ckpt_dir, device)
        self.client.warm_up()
        """
        server.route("/chat", methods=["POST"]): 使用 Flask 的路由装饰器定义一个新的路由 /chat，该路由仅接受 POST 请求。
        (self.chat): 将请求发送到 self.chat 方法。self.chat 是处理 /chat 路由的视图函数，通常负责接收客户端的请求、处理输入并返回响应。
        """
        server.route("/chat", methods=["POST"])(self.chat)
        
        """
        server.run(...): 如果 run_app 为 True，调用 Flask 的 run 方法启动服务器。
        host=ip: 指定服务器监听的 IP 地址。ip 通常是 '0.0.0.0'，表示监听所有可用的网络接口。
        threaded=False: 这表示 Flask 服务器将以单线程模式运行，适用于某些情况，例如需要避免资源竞争或调试时。
        """
        if run_app:
            server.run(host=ip, port=port, threaded=False)
        else:
            self.server = server
    #这段代码是 Flask 应用中的一个视图函数 chat，用于处理音频数据并返回处理结果。
    def chat(self) -> Response:

        req_data = flask.request.get_json()  # 从客户端获取JASON数据,包括音频数据和其他参数
        try:
            data_buf = req_data["audio"].encode("utf-8")  # 从获取的数据中提取音频数据，进行UTF-8编码
            data_buf = base64.b64decode(data_buf)  # 解码音频数据，这意味着客户端发送的音频数据是经过 Base64 编码的。
            stream_stride = req_data.get("stream_stride", 4)
            max_tokens = req_data.get("max_tokens", 2048)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(data_buf)
                audio_generator = self.client.run_AT_batch_stream(f.name, stream_stride, max_tokens)
                return Response(stream_with_context(audio_generator), mimetype="audio/wav")
        except Exception as e:
            print(traceback.format_exc())  # traceback.format_exc() 是 Python 标准库 traceback 模块中的一个函数，用于获取当前异常的详细信息，并以字符串的形式返回。


# CUDA_VISIBLE_DEVICES=1 gunicorn -w 2 -b 0.0.0.0:60808 'server:create_app()'
def create_app():
    server = OmniChatServer(run_app=False)  # 创建 OmniChatServer 的实例，但不立即运行服务器。这可以用于配置或获取服务器实例，而不启动它。
    return server.server


def serve(ip='0.0.0.0', port=60808, device='cuda:0'):

    OmniChatServer(ip, port=port,run_app=True, device=device)


if __name__ == "__main__":
    import fire
    fire.Fire(serve)  # fire.Fire(...) 是 fire 库的核心功能，它会根据传入的对象（在这里是 serve）创建一个命令行接口。