import numpy as np
import cv2

# COCO数据集
class_names = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# 随机数生成器，种子为5，保证每次生产的随机数相同
rng = np.random.default_rng(10)
"""
0,255表示生成 0~255 之间的随机数,表示不同的颜色值
生成形状为(size=(len(class_names), 3)的数组,size表示obj的类别,3表示RGB颜色的三个分量

"""
colors = rng.uniform(0, 255, size=(len(class_names), 3)) # 为所有类别随机生成固定的不同颜色，用于后续绘制边界框

# NMS去除重叠边界框
def nms(boxes, scores, iou_threshold):
    # 按照得分进行排序
    sorted_indices = np.argsort(scores)[::-1] # 先升序排序，再通过切片操作降序排序，返回索引

    keep_boxes = []  # 初始化列表，用于保存符合NMS筛选得到的边界框
    while sorted_indices.size > 0:
        box_id = sorted_indices[0]  # 提取置信度分数最大的检测框
        keep_boxes.append(box_id)

        # 计算当前边界框与其他所有边界框的iou
        # 每个边界框用box[索引,坐标]表示，其中索引表示边界框索引，坐标表示每个边界框的坐标（x1,y1,x2,y2）,即左上角坐标和右下角坐标
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1, :], :])

        # 删除iou大于阈值的边界框，保留剩余边界框，返回对应边界框索引
        keep_indices = np.where(ious < iou_threshold)[0]

        # 更新索引，进行下一个循环
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes
    

# 多类别NMS
def multiclass_nms(boxes, scores, class_ids, iou_threshold):
    unique_class_ids = np.unique(class_ids)  # 去除重复类别id,生成唯一id列表，按升序排列

    keep_boxes = []
    # 遍历所有类别
    for class_id in unique_class_ids:
        class_indices = np.where(class_ids == class_id)[0]  # 获取同类目标的索引
        class_boxes = boxes[class_indices, :]  # 获取所有同类目标边界框
        class_scores = scores[class_indices]  # 获取同类别目标边界框的置信度得分情况

        class_keep_boxes = nms(class_boxes, class_scores, iou_threshold)
        keep_boxes.extend(class_indices[class_keep_boxes])
    
    return keep_boxes

# 计算IoU
# 边界框的尺寸为x1y1x2y2
def compute_iou(box, boxes):
    xmin = np.maximum(box[0], boxes[:, 0]) # 计算当前检测框的x1和所有检测框的x1的最大值
    ymin = np.maximum(box[1], boxes[:, 1]) # 计算当前检测框的y1和所有检测框的y1的最大值
    xmax = np.minimum(box[2], boxes[:, 2]) # 计算当前检测框的x2和所有检测框的x2的最小值
    ymax = np.minimum(box[3], boxes[:, 3]) # 计算当前检测框的y2和所有检测框的y2的最小值

    # 计算交集(矩形)
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
    
    # 计算并集
    box_area = (box[2] - box[0]) * (box[3] - box[1]) #当前检测框面积(x2-x1)*(y2-y1)
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) # 所有检测框面积(x2-x1)*(y2-y1)
    union_area = box_area + boxes_area - intersection_area  # 当前检测框面积 + 所有检测框面积 - 当前检测框面积与所有检测框面积的交集

    # 计算iou: 交并比
    iou = intersection_area / union_area
    
    return iou

# 尺寸转换：xywh ——>  x1y1x2y2
def xywh2xyxy(x):
    y = np.copy(x)  # 拷贝，避免在原数组上操作
    y[..., 0] = x[..., 0] - x[..., 2] / 2  #计算所有检测框（...与：是不同的用法）左上角的x1坐标：x1 = x - w/2
    y[..., 1] = x[..., 1] - x[..., 3] / 2  #计算所有检测框左上角的y1坐标：y1 = y - h/2
    y[..., 2] = x[..., 0] + x[..., 2] / 2  #计算所有检测框右下角的x2坐标：x2 = x + w/2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  #计算所有检测框右下角的y2坐标：y2 = y + h/2
    
    return y

def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3):
    det_img = image.copy()
    img_height, img_width = image.shape[:2]  # image的形状为(w,h,c) 即宽高和通道数（RGB）
    font_size = min([img_height, img_width]) * 0.0006 # 文本大小
    text_thickness = int(min([img_height, img_width]) * 0.001) # 字体粗细

    # 画边界框和标签
    for class_id, box, score in zip(class_ids, boxes, scores):
        color = colors[class_id]

        draw_box(det_img, box, color)

        label = class_names[class_id]
        caption = f"{label} {int(score * 100)}%"
        draw_text(det_img, caption, box, color, font_size, text_thickness)

    return det_img

def draw_box(
    image: np.ndarray,  # 图像为Numpy数组
    box: np.ndarray,
    color: tuple[int, int, int] = (0, 0, 255),  # 初始化为红色
    thickness: int = 2,  # 线条粗细程度
) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(int)  # 坐标类型为整数型
    return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

def draw_text(
    image: np.ndarray,
    text: str,
    box: np.ndarray,
    color: tuple[int, int, int] = (0, 0, 255),
    font_size: float = 0.001,
    text_thickness: int = 2,
) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(int)
    (tw, th), _ = cv2.getTextSize(
        text=text,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_size,
        thickness=text_thickness,
    )
    th = int(th * 1.2)

    cv2.rectangle(image, (x1, y1), (x1 + tw, y1 - th), color, -1)  # -1表示用矩形填充图像
    
    # 绘制文本
    return cv2.putText(
        image,
        text,
        (x1, y1),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,  # 抗锯齿线条类型
    )

# 绘制掩码图像
def draw_masks(
    image: np.ndarray, boxes: np.ndarray, classes: np.ndarray, mask_alpha: float = 0.3
) -> np.ndarray:
    mask_img = image.copy()

    
    for box, class_id in zip(boxes, classes):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # 使用掩码图像进行矩形填充
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)  # -1: 填充参数。如果设置为 -1，则矩形将被填充为指定的颜色；如果设置为其他正值，将绘制矩形的边框。

    # 返回原始图像和掩码图像的加权混合(掩码图像，掩码图像权重，原始图像，原始图像权重，加权混合后图像的亮度偏移量)
    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)  
