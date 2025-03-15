import json
import numpy as np


def box_iou_batch(
        boxes_a: np.ndarray, boxes_b: np.ndarray
) -> np.ndarray:
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area_a = box_area(boxes_a.T)
    area_b = box_area(boxes_b.T)

    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[:, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[:, 2:])

    area_inter = np.prod(
        np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)

    return area_inter / (area_a[:, None] + area_b - area_inter)


def non_max_suppression(
        bboxes: np.ndarray,
        conf_scores: np.ndarray,
        iou_threshold: float = 0.5
) -> np.ndarray:
    rows, columns = bboxes.shape

    sort_index = np.flip(conf_scores.argsort())
    bboxes = bboxes[sort_index]

    boxes = bboxes[:, :4]
    ious = box_iou_batch(boxes, boxes)
    ious = ious - np.eye(rows)

    keep = np.ones(rows, dtype=bool)

    for index, iou in enumerate(ious):
        if not keep[index]:
            continue

        condition = (iou > iou_threshold)
        keep = keep & ~condition

    return np.nonzero(keep[sort_index.argsort()] == True)[0]


def calculate_iou(box1, box2):
    # Calculate the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        # If there is no overlap, return 0
        return 0.0

    # Calculate the area of intersection rectangle
    intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)

    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate the area of union
    union_area = box1_area + box2_area - intersection_area

    # Calculate the IoU
    iou = intersection_area / union_area

    return iou

def get_ocr_from_form_recognizer(ocr_result, level='line'):
    if isinstance(ocr_result, str):
        data = json.load(open(ocr_result, 'r', encoding='utf-8'))
    else:
        data = ocr_result

    boxes, words, conf_scores = [], [], []
    for element in data:
        box = element['polygon']
        x1 = min(box[0][0], box[1][0], box[2][0], box[3][0])
        y1 = min(box[0][1], box[1][1], box[2][1], box[3][1])
        x2 = max(box[0][0], box[1][0], box[2][0], box[3][0])
        y2 = max(box[0][1], box[1][1], box[2][1], box[3][1])
        boxes.append([x1, y1, x2, y2])
        words.append(element['text'])
        conf_scores.append(element['confidence'] if element.get('confidence') is not None else 1)

    return words, boxes, conf_scores
def ocr2structured_text(words, boxes, conf_scores, image_width, image_height=None):
    if image_height is None:
        image_height = max(box[3] for box in boxes) if boxes else 2000

    # Nhóm các từ theo dòng dựa trên vị trí y
    line_groups = {}
    line_height = 25  # Có thể điều chỉnh dựa trên kích cỡ phổ biến của văn bản
    
    for word, box, conf in zip(words, boxes, conf_scores):
        # Sử dụng trung tâm của box để xác định dòng
        center_y = (box[1] + box[3]) / 2
        line_idx = int(center_y / line_height)
        
        if line_idx not in line_groups:
            line_groups[line_idx] = []
            
        line_groups[line_idx].append((word, box, conf))
    
    # Sắp xếp các dòng theo thứ tự tăng dần của y
    sorted_lines = sorted(line_groups.items(), key=lambda x: x[0])
    
    # Tính toán tỉ lệ nén khoảng cách ngang
    # Giá trị nhỏ hơn sẽ nén nhiều hơn
    compression_ratio = 0.3  # Có thể điều chỉnh (0.1 - 0.5 là phạm vi tốt)
    
    # Tạo markdown với khoảng cách được tối ưu
    markdown_text = ""
    
    for i, (line_idx, line_items) in enumerate(sorted_lines):
        # Sắp xếp các từ trên mỗi dòng theo vị trí x
        line_items.sort(key=lambda item: item[1][0])
        
        line = ""
        last_end_x = 0
        
        for word, box, _ in line_items:
            # Tính khoảng cách từ cuối từ trước đến đầu từ hiện tại
            start_x = box[0]
            
            if last_end_x > 0:  # Không phải từ đầu tiên trên dòng
                # Tính số khoảng trắng cần chèn
                # Nén khoảng cách dài bằng logarit hoặc căn bậc hai
                raw_spaces = start_x - last_end_x
                
                if raw_spaces <= 10:  # Khoảng cách nhỏ
                    spaces = max(1, int(raw_spaces / 5))
                else:  # Khoảng cách lớn, áp dụng nén
                    spaces = max(1, int(compression_ratio * raw_spaces / 5))
                    
                line += " " * spaces + word
            else:
                # Từ đầu tiên trên dòng, áp dụng khoảng cách từ lề
                indent = max(0, int(compression_ratio * start_x / 10))
                line += " " * indent + word
            
            # Cập nhật vị trí cuối cùng
            # Ước tính vị trí kết thúc bằng vị trí bắt đầu + độ rộng của từ
            avg_char_width = 8  # Ước tính độ rộng trung bình của ký tự
            last_end_x = start_x + len(word) * avg_char_width
            
        markdown_text += line + "\n"
        
        # Thêm khoảng cách dọc giữa các dòng nếu cần
        if i < len(sorted_lines) - 1:  # Không phải dòng cuối cùng
            next_line_idx = sorted_lines[i + 1][0]  # Lấy chỉ số của dòng tiếp theo
            if next_line_idx - line_idx > 2:  # Nếu khoảng cách giữa 2 dòng lớn
                # Thêm dòng trống
                markdown_text += "\n"
    
    return markdown_text