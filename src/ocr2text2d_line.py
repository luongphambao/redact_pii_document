# filepath: f:\code\Ron_Wee\ocr2text2d_line.py
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


def merge_nearby_lines(
        line_boxes: List[List[float]],
        threshold: float = 0.3
) -> List[List[float]]:
    """
    Merge nearby lines based on vertical distance threshold.
    
    Args:
        line_boxes: List of line bounding boxes [x1, y1, x2, y2]
        threshold: Threshold for vertical distance relative to line height
        
    Returns:
        List of merged line bounding boxes
    """
    if not line_boxes:
        return []
    
    # Sort lines by y-coordinate (top to bottom)
    sorted_lines = sorted(line_boxes, key=lambda box: box[1])
    merged_lines = [sorted_lines[0]]
    
    for line in sorted_lines[1:]:
        prev_line = merged_lines[-1]
        
        # Calculate line heights
        current_height = line[3] - line[1]
        prev_height = prev_line[3] - prev_line[1]
        avg_height = (current_height + prev_height) / 2
        
        # Check if lines are close enough to merge
        if line[1] - prev_line[3] < threshold * avg_height:
            # Merge the lines
            merged_lines[-1] = [
                min(prev_line[0], line[0]),
                prev_line[1],
                max(prev_line[2], line[2]),
                line[3]
            ]
        else:
            merged_lines.append(line)
            
    return merged_lines


def extract_lines_from_ocr(
        boxes: List[List[float]],
        words: List[str],
        conf_scores: List[float],
        overlap_threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Extract text lines from OCR results by grouping word boxes.
    
    Args:
        boxes: Word bounding boxes [x1, y1, x2, y2]
        words: Text content for each box
        conf_scores: Confidence scores for each box
        overlap_threshold: Threshold for vertical overlap
        
    Returns:
        List of line dictionaries with boxes, text and confidence
    """
    if not boxes or not words:
        return []
    
    # Convert to numpy arrays for easier manipulation
    boxes_array = np.array(boxes)
    
    # Sort boxes by y-coordinate first (top to bottom)
    sorted_indices = np.argsort(boxes_array[:, 1])
    
    lines = []
    current_line = {
        'box': boxes[sorted_indices[0]],
        'words': [words[sorted_indices[0]]],
        'confidence': [conf_scores[sorted_indices[0]]],
        'indices': [sorted_indices[0]]
    }
    
    def vertical_overlap(box1, box2):
        # Calculate vertical overlap between two boxes
        height1 = box1[3] - box1[1]
        height2 = box2[3] - box2[1]
        overlap_height = min(box1[3], box2[3]) - max(box1[1], box2[1])
        return overlap_height / min(height1, height2)
    
    for i in sorted_indices[1:]:
        current_box = boxes[i]
        
        # Check if the current box belongs to the current line
        if vertical_overlap(current_line['box'], current_box) > overlap_threshold:
            # Update line box
            current_line['box'] = [
                min(current_line['box'][0], current_box[0]),
                min(current_line['box'][1], current_box[1]),
                max(current_line['box'][2], current_box[2]),
                max(current_line['box'][3], current_box[3])
            ]
            current_line['words'].append(words[i])
            current_line['confidence'].append(conf_scores[i])
            current_line['indices'].append(i)
        else:
            # Sort words in current line by x-coordinate
            x_sorted = sorted(zip(
                current_line['words'], 
                current_line['confidence'],
                current_line['indices'],
                [boxes[j][0] for j in current_line['indices']]
            ), key=lambda x: x[3])
            
            # Save the current line
            lines.append({
                'box': current_line['box'],
                'words': [word for word, _, _, _ in x_sorted],
                'confidence': [conf for _, conf, _, _ in x_sorted],
                'indices': [idx for _, _, idx, _ in x_sorted]
            })
            
            # Start a new line
            current_line = {
                'box': current_box,
                'words': [words[i]],
                'confidence': [conf_scores[i]],
                'indices': [i]
            }
    
    # Don't forget the last line
    if current_line['words']:
        x_sorted = sorted(zip(
            current_line['words'], 
            current_line['confidence'],
            current_line['indices'],
            [boxes[j][0] for j in current_line['indices']]
        ), key=lambda x: x[3])
        
        lines.append({
            'box': current_line['box'],
            'words': [word for word, _, _, _ in x_sorted],
            'confidence': [conf for _, conf, _, _ in x_sorted],
            'indices': [idx for _, _, idx, _ in x_sorted]
        })
    
    return lines


def get_line_text_representation(lines: List[Dict[str, Any]]) -> str:
    """
    Convert extracted lines to a formatted text representation.
    
    Args:
        lines: List of line dictionaries with boxes and text
        
    Returns:
        Formatted text with line information
    """
    result = []
    
    for i, line in enumerate(lines):
        words = ' '.join(line['words'])
        box = line['box']
        avg_conf = sum(line['confidence']) / len(line['confidence']) if line['confidence'] else 0
        
        line_info = f"Line {i+1} [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}] " \
                    f"(Conf: {avg_conf:.2f}): {words}"
        result.append(line_info)
    
    return '\n'.join(result)


def reorder_lines_based_on_layout(
        lines: List[Dict[str, Any]],
        column_threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Reorder lines based on potential multi-column layout.
    
    Args:
        lines: List of line dictionaries
        column_threshold: Threshold for determining columns
        
    Returns:
        Reordered list of lines
    """
    if not lines:
        return []
    
    # Extract boxes
    boxes = [line['box'] for line in lines]
    boxes_array = np.array(boxes)
    
    # Calculate horizontal center point for each box
    centers_x = (boxes_array[:, 0] + boxes_array[:, 2]) / 2
    
    # Try to detect columns using histogram
    hist, bin_edges = np.histogram(centers_x, bins=20)
    peaks = [i for i in range(1, len(hist)-1) if hist[i] > hist[i-1] and hist[i] > hist[i+1]]
    
    if len(peaks) > 1:
        # Multiple columns detected
        column_splits = [bin_edges[p] for p in peaks]
        column_splits = [0] + column_splits + [float('inf')]
        
        # Assign lines to columns
        columns = []
        for i in range(len(column_splits) - 1):
            start, end = column_splits[i], column_splits[i+1]
            col_lines = [
                j for j, line in enumerate(lines)
                if start <= (line['box'][0] + line['box'][2]) / 2 < end
            ]
            if col_lines:
                columns.append(col_lines)
        
        # Create a new ordering of lines by going through columns
        reordered_indices = []
        for col in columns:
            # Sort lines in this column by y-coordinate
            col_sorted = sorted(col, key=lambda i: lines[i]['box'][1])
            reordered_indices.extend(col_sorted)
        
        return [lines[i] for i in reordered_indices]
    
    # Single column - sort by y-coordinate
    return sorted(lines, key=lambda line: line['box'][1])