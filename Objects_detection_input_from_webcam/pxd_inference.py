import time
import cv2
import numpy as np
import onnxruntime 
# 跨onnxruntime 是一个高性能的推理引擎，用于运行 ONNX（Open Neural Network Exchange）模型。ONNX 是一个开放的深度学习模型格式，允许不同的深度学习框架（如 PyTorch、TensorFlow 等）之间进行互操作，使得模型可以在不同的环境中运行。

from pxd_utils import draw_detections

class YOLOv10:
    def __init__(self, path):
        # 初始化模型
        self.initialize_model(path)
    
    # 使用__call__，表示可以直接调用类YOLOV10的实例，而无需显示调用
    def __call__(self, image):
        return self.detect_objects(image)
    
    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(
            path, providers=onnxruntime.get_available_providers()
        )
        # 获取模型信息
        self.get_input_details()
        self.get_output_details()

    def detect_objects(self, image, conf_threshold):
        input_tensor = self.prepare_input(image)
        # 推理返回检测结果
        new_image = self.inference(image, input_tensor, conf_threshold)
        
        return new_image
    
    # 输入图片预处理
    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]
        # 将输入图像的颜色空间从OpenCV的BGR格式转换成Pytorch中的RGB格式
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 修改图片尺寸
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))
        
        input_img = input_img / 255.0  #归一化像素值在[0, 1]之间，这种处理可以帮助加速模型训练，提高收敛速度，并减少数值不稳定性。
        input_img = input_img.transpose(2, 0, 1) # 交换维度，由(w, h , c)，变为(c, w, h)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32) # 增加新维度，变成(1, c, w, h),即增加一个batchsize维度

        return input_tensor
    
    # 推理
    def inference(self, image, input_tensor, conf_threshold):
        start_time = time.perf_counter()
        outputs = self.session.run(
            self.output_names, {self.input_names[0]: input_tensor}
        )

        print(f"Inference time: {(time.perf_counter() - start_time)*1000:.2f} ms")
        boxes, scores, class_ids, = self.process_output(outputs, conf_threshold)
        return self.draw_detections(image, boxes, scores, class_ids)
    
    # 获取目标边界框、分数和类别
    def process_output(self, outputs, conf_threshold):

        # predictions 的维度为(x, y, w, h, 置信度, 类别)
        predictions = np.squeeze(outputs[0]) # 删除输出结果中维度是1的维度，如output[0]形状为(1，3, 244, 244)，执行后结果为形状(3， 244， 244)
        
        # 过滤掉小于阈值的目标置信度分数
        scores = predictions[:, 4]
        predictions = predictions[scores > conf_threshold, :]
        scores = scores[scores > conf_threshold]

        if len(scores) == 0:
            return [], [], []
        
        # 获取置信度最高的类别
        class_ids = predictions[:, 5].astype(int)

        # 获取每个目标的bbox
        boxes = self.extract_boxes(predictions)

        return boxes, scores, class_ids
    
    def extract_boxes(self, predictions):
        # 获取每个预测目标的前4列信息xywh
        boxes = predictions[:, :4]
        
        # 调整边界框到原始图像维度
        boxes = self.rescale_boxes(boxes)

        return boxes
    
    def rescale_boxes(self, boxes):
        # 输入的原始图像尺寸为x1y1x2y2，即左上角和右下角坐标
        input_shape = np.array(
            [self.input_width, self.input_height, self.input_width, self.input_height]
        )
        # 将boxes中的每个边界框坐标除以input_shape的对应值，使得边界框坐标从原始输入图像的比例缩放到[0,1]的范围，并确保结果为浮点数
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        # 调整边界框大小
        boxes *= np.array(
            [self.img_width, self.img_height, self.img_width, self.img_height]
        )
        return boxes
    
    def draw_detections(self, image, boxes, scores, class_ids, draw_scores=True, mask_alpha=0.4):
        return draw_detections(image, boxes, scores, class_ids, mask_alpha)
    
    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        # shape 属性通常是一个包含输入张量维度的元组，例如 (batch_size, channels,  height, width)
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
    

if __name__ == "__main__":
    import requests
    import tempfile
    from huggingface_hub import hf_hub_download

    model_file = hf_hub_download(
        repo_id="onnx-community/yolov10s", filename="onnx/model.onnx"
    )

    yolov10_detector = YOLOv10(model_file)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:   # delete = False表示关闭文件后文件不会被删除
        f.write(
            requests.get(
                "https://live.staticflickr.com/13/19041780_d6fd803de0_3k.jpg"
            ).content   #content 属性返回响应体的字节内容
        )
        f.seek(0)  #将指针移到文件开头，以便后续读取
        img = cv2.imread(f.name)

    # 目标检测
    combined_image = yolov10_detector.detect_objects(img)


    # 绘制窗口
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL) #cv2.WINDOW_NORMAL 参数允许用户调整窗口大小，使其可以根据需要进行调整。
    cv2.imshow("Output", combined_image)
    cv2.waitKey(0) #等待用户按下任意键后退出窗口。0 表示无限期等待，直到有键被按下。






    








    



      



    


    
