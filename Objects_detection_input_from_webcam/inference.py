import time
import cv2
import numpy as np
import onnxruntime

from utils import draw_detections


class YOLOv10:
    def __init__(self, path):

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(
            path, providers=onnxruntime.get_available_providers()
        )
        # Get model info
        self.get_input_details()
        self.get_output_details()

    def detect_objects(self, image, conf_threshold=0.3):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image  推理返回检测结果，包含标注类型和置信度的图片
        new_image = self.inference(image, input_tensor, conf_threshold)

        return new_image

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)  #交换维度，由（宽，高，通道），变为（通道，宽，高），以用于Pytorch框架
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)  #增加新的维度，变成（1，通道，宽，高），即表示一个batch

        return input_tensor

    def inference(self, image, input_tensor, conf_threshold=0.3):
        start = time.perf_counter()
        outputs = self.session.run(
            self.output_names, {self.input_names[0]: input_tensor}  
        )

        print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        boxes, scores, class_ids, = self.process_output(outputs, conf_threshold)
        return self.draw_detections(image, boxes, scores, class_ids)

    def process_output(self, output, conf_threshold=0.3):
        predictions = np.squeeze(output[0]) #去除输出结果中维度是1的维度，如output[0]维度为(N,1),则结果为(N,)
        
        # predictions 的维度为(x, y, w, h, 置信度, 类别)

        # Filter out object confidence scores below threshold
        scores = predictions[:, 4]
        predictions = predictions[scores > conf_threshold, :]
        scores = scores[scores > conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = predictions[:, 5].astype(int)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        return boxes, scores, class_ids

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]  #获取每个预测目标的前4列信息即xywh

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        #boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):
        # Rescale boxes to original image dimensions
        input_shape = np.array(
            [self.input_width, self.input_height, self.input_width, self.input_height] #即xywh，xy是中心坐标
        )
        #将boxes中的每个边界框坐标除以input_shape的对应值，使得边界框坐标从原始输入图像的比例缩放到[0,1]的范围，并确保结果为浮点数
        boxes = np.divide(boxes, input_shape, dtype=np.float32) 
        #将边界框重新缩放到输入原始图像的大小
        boxes *= np.array(
            [self.img_width, self.img_height, self.img_width, self.img_height] 
        )
        return boxes

    def draw_detections(self, image, boxes, scores, class_ids, draw_scores=True, mask_alpha=0.4):
        return draw_detections(
            image, boxes, scores, class_ids, mask_alpha
        )

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        
        #shape 属性通常是一个包含输入张量维度的元组，例如 (batch_size, channels, height, width)。
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

    yolov8_detector = YOLOv10(model_file)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:   # delete = False表示关闭文件后文件不会被删除
        f.write(
            requests.get(
                "https://live.staticflickr.com/13/19041780_d6fd803de0_3k.jpg"
            ).content   #content 属性返回响应体的字节内容
        )
        f.seek(0)  #将指针移到文件开头，以便后续读取
        img = cv2.imread(f.name)

    # # Detect Objects
    combined_image = yolov8_detector.detect_objects(img)


    # Draw detections
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL) #cv2.WINDOW_NORMAL 参数允许用户调整窗口大小，使其可以根据需要进行调整。
    cv2.imshow("Output", combined_image)
    cv2.waitKey(0) #等待用户按下任意键后退出窗口。0 表示无限期等待，直到有键被按下。