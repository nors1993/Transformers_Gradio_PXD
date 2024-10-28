import numpy as np
import cv2

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

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)   # 随机数生成器，种为3，保证每次生成的随机数相同
“”“
0,255表示生成 0~255 之间的随机数,表示不同的颜色值
生成形状为(size=(len(class_names), 3)的数组，size表示obj的类别，3表示RGB颜色的三个分量
”“”
colors = rng.uniform(0, 255, size=(len(class_names), 3))   # 随机生成size种颜色（对应不同的类别）


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]   #先升序排序，再通过切片操作降序排序，返回索引

    keep_boxes = []   #初始化列表，用于保存符合NMS算法筛选得到的检测框
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]   #提取置信度最大的检测框
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])   #计算当前边界框与其他边界框的IoU，
        # 每个边界框用box[索引,坐标]表示，其中索引表示边界框索引，坐标表示每个边界框的坐标（x1,y1,x2,y2）,即左上角坐标和右下角坐标

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]   # 删除IoU大于阈值的边界框，保留剩余边界框，返回对应边界框索引

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]  #不包含当前筛选出的检测框，所以索引还需+1

    return keep_boxes

# 多类别NMS
def multiclass_nms(boxes, scores, class_ids, iou_threshold):
    unique_class_ids = np.unique(class_ids)  #去除重复类别id,生成唯一id列表，按升序排列

    keep_boxes = []
    for class_id in unique_class_ids:   #遍历所有物体类别
        class_indices = np.where(class_ids == class_id)[0]  #获取相同类别物体的索引
        class_boxes = boxes[class_indices, :]  #同类物体的边界框
        class_scores = scores[class_indices]  # 同类别物体边界框的置信度得分情况

        class_keep_boxes = nms(class_boxes, class_scores, iou_threshold)
        keep_boxes.extend(class_indices[class_keep_boxes])

    return keep_boxes

#  计算IoU
def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])  #计算当前检测框的x1和所有检测框的x1的最大值
    ymin = np.maximum(box[1], boxes[:, 1])  #计算当前检测框的y1和所有检测框的y1的最大值
    xmax = np.minimum(box[2], boxes[:, 2])  #计算当前检测框的x2和所有检测框的x2的最小值
    ymax = np.minimum(box[3], boxes[:, 3])  #计算当前检测框的y2和所有检测框的y2的最小值

    # Compute intersection area   计算交集
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area   计算并集
    box_area = (box[2] - box[0]) * (box[3] - box[1])  #当前检测框面积(x2-x1)*(y2-y1)
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  #所有检测框面积(x2-x1)*(y2-y1)
    union_area = box_area + boxes_area - intersection_area  #  当前检测框面积 + 所有检测框面积 - 当前检测框面积与所有检测框面积的交集

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)   其中x,y表示检测框中心坐标，w,h表示x,y上的宽和高
    y = np.copy(x)  # 拷贝，避免在原数组上操作
    y[..., 0] = x[..., 0] - x[..., 2] / 2  #计算所有检测框（可用...或者：表示）左上角的x1坐标：x1 = x - w/2
    y[..., 1] = x[..., 1] - x[..., 3] / 2  #计算所有检测框（可用...或者：表示）左上角的y1坐标：y1 = y - h/2
    y[..., 2] = x[..., 0] + x[..., 2] / 2  #计算所有检测框（可用...或者：表示）右下角的x2坐标：x2 = x + w/2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  #计算所有检测框（可用...或者：表示）右下角的y2坐标：y2 = y + h/2
    return y


def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3):
    det_img = image.copy()

    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    #det_img = draw_masks(det_img, boxes, class_ids, mask_alpha)

    # Draw bounding boxes and labels of detections
    for class_id, box, score in zip(class_ids, boxes, scores):
        color = colors[class_id]

        draw_box(det_img, box, color)

        label = class_names[class_id]
        caption = f"{label} {int(score * 100)}%"
        draw_text(det_img, caption, box, color, font_size, text_thickness)

    return det_img


def draw_box(
    image: np.ndarray,
    box: np.ndarray,
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(int)
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

    cv2.rectangle(image, (x1, y1), (x1 + tw, y1 - th), color, -1)

    return cv2.putText(
        image,
        text,
        (x1, y1),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )


def draw_masks(
    image: np.ndarray, boxes: np.ndarray, classes: np.ndarray, mask_alpha: float = 0.3
) -> np.ndarray:
    mask_img = image.copy()

    # Draw bounding boxes and labels of detections
    for box, class_id in zip(boxes, classes):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # Draw fill rectangle in mask image
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)