import os 
import json 
import re
from difflib import SequenceMatcher

def find_sensitive_data_bboxes(ocr_json, sensitive_data_file):
    """
    Find bounding boxes for sensitive data in OCR results using word-by-word matching.
    
    Args:
        ocr_json: List of OCR result objects with text and bounding boxes
        sensitive_data_file: Path to file containing sensitive data to search for
        
    Returns:
        Dictionary mapping sensitive data to their bounding boxes
    """
    # Read sensitive data
    with open(sensitive_data_file, "r") as f:
        sensitive_lines = [line.strip() for line in f.readlines() if line.strip()]
    
    results = {}
    
    # For each piece of sensitive data, find matching OCR results
    for sensitive_item in sensitive_lines:
        results[sensitive_item] = []
        
        # STEP 1: First try exact and partial matches in single blocks (fast path)
        exact_match_found = False
        for ocr_item in ocr_json:
            ocr_text = ocr_item.get("text", "")
            
            # Check for exact match
            if sensitive_item == ocr_text:
                # Exact match - use the whole bounding box
                results[sensitive_item].append({
                    "text": ocr_text,
                    "bbox": ocr_item.get("bbox"),
                    "confidence": ocr_item.get("confidence"),
                    "match_type": "exact"
                })
                exact_match_found = True
                break
            # Check for partial match
            elif sensitive_item in ocr_text:
                # Partial match - estimate the position of the sensitive data within the text
                bbox = estimate_substring_bbox(ocr_text, sensitive_item, ocr_item.get("bbox"))
                if bbox:
                    results[sensitive_item].append({
                        "text": ocr_text,
                        "bbox": bbox,
                        "confidence": ocr_item.get("confidence"),
                        "match_type": "partial"
                    })
                    exact_match_found = True
                    break
        
        # STEP 2: If no exact match found, try word-by-word matching
        if not exact_match_found:
            words = sensitive_item.split()
            if not words:
                continue  # Skip empty items
            
            word_matches = []
            
            # Find best match for each word individually
            for word in words:
                
                # Check all OCR items for the best match for this word
                for ocr_item in ocr_json:
                    ocr_text = ocr_item.get("text", "")
                    if not ocr_text or not ocr_item.get("bbox"):
                        continue
                    
                    # Try exact substring match first
                    if word in ocr_text:
                        word_bbox = estimate_substring_bbox(ocr_text, word, ocr_item.get("bbox"))
                        if word_bbox:
                            # word_matches.append({
                            #     "word": word,
                            #     "bbox": word_bbox,
                            #     "confidence": ocr_item.get("confidence", 0),
                            #     "score": 1.0
                            # })
                            results[sensitive_item].append({
                                "text": ocr_text,
                                "bbox": word_bbox,
                                "confidence": ocr_item.get("confidence"),
                                "match_type": "word_match"
                            })
                            break  # Found exact match for this word
                    
            
    
    return results

def get_fuzzy_match_score(word1, word2):
    """
    Calculate fuzzy match score between two words using sequence matcher.
    Returns a score between 0 and 1, where 1 is a perfect match.
    """
    # Normalize words before comparing
    norm_word1 = normalize_word(word1)
    norm_word2 = normalize_word(word2)
    
    # If either word is empty after normalization, return 0
    if not norm_word1 or not norm_word2:
        return 0
    
    # Calculate similarity ratio
    return SequenceMatcher(None, norm_word1, norm_word2).ratio()

def normalize_word(word):
    """
    Normalize a word by removing punctuation and converting to lowercase.
    """
    import re
    # Remove punctuation and convert to lowercase
    return re.sub(r'[^\w\s]', '', word).lower()
def estimate_substring_bbox(full_text, substring, full_bbox):
    """
    Estimate the bounding box of a substring within text with maximum accuracy.
    
    Args:
        full_text: The complete text string
        substring: The substring to locate
        full_bbox: The bounding box of the complete text [x_min, y_min, x_max, y_max]
        
    Returns:
        Estimated bounding box for the substring or None if not found
    """
    # Early validation checks
    if not full_bbox or not full_text or not substring:
        return None
    
    if substring not in full_text:
        return None
    
    # Quick path for exact matches
    if full_text == substring:
        return full_bbox
    
    # Unpack bbox
    x_min, y_min, x_max, y_max = full_bbox
    total_width = x_max - x_min
    total_height = y_max - y_min
    
    # Find substring position
    start_pos = full_text.find(substring)
    end_pos = start_pos + len(substring)
    
    # Comprehensive character width weights (based on proportional font metrics)
    char_weights = {
        # Narrow characters
        'i': 0.4, 'j': 0.4, 'l': 0.4, 'I': 0.4, '!': 0.4, '.': 0.4, ',': 0.4, "'": 0.4, 
        '(': 0.4, ')': 0.4, '[': 0.4, ']': 0.4, '{': 0.4, '}': 0.4, '|': 0.3, ':': 0.4, 
        ';': 0.4, '-': 0.5, '`': 0.4, '´': 0.4, '\'': 0.4, '"': 0.7,
        
        # Standard characters
        'a': 0.9, 'b': 1.0, 'c': 0.9, 'd': 1.0, 'e': 0.9, 'f': 0.6, 'g': 1.0, 
        'h': 1.0, 'k': 0.9, 'n': 1.0, 'o': 1.0, 'p': 1.0, 'q': 1.0, 'r': 0.7, 
        's': 0.8, 't': 0.6, 'u': 1.0, 'v': 0.9, 'x': 1.0, 'y': 0.9, 'z': 0.9,
        
        # Wide characters
        'm': 1.5, 'w': 1.5, 'W': 1.8, 'M': 1.8, '@': 1.6, '#': 1.3, '%': 1.3, '&': 1.3,
        
        # Uppercase characters (generally wider than lowercase)
        'A': 1.2, 'B': 1.1, 'C': 1.2, 'D': 1.2, 'E': 1.1, 'F': 1.0, 'G': 1.3, 
        'H': 1.2, 'J': 0.9, 'K': 1.1, 'L': 0.9, 'N': 1.2, 'O': 1.3, 'P': 1.1, 
        'Q': 1.3, 'R': 1.1, 'S': 1.1, 'T': 1.0, 'U': 1.2, 'V': 1.2, 'X': 1.2, 
        'Y': 1.2, 'Z': 1.1,
        
        # Numbers
        '0': 1.0, '1': 0.6, '2': 1.0, '3': 1.0, '4': 1.0, '5': 1.0, 
        '6': 1.0, '7': 0.9, '8': 1.0, '9': 1.0,
        
        # Whitespace
        ' ': 0.6,  # Space
        '\t': 2.4,  # Tab (usually 4 spaces)
        '\n': 0.0,  # Newline (no width)
        
        # Special/common symbols
        '*': 0.7, '+': 1.0, '=': 1.1, '/': 0.7, '\\': 0.7, '_': 1.0, 
        '~': 1.1, '<': 1.0, '>': 1.0, '?': 0.9, '$': 1.0, '€': 1.2, '£': 1.1,
        '¥': 1.2, '©': 1.5, '®': 1.5, '™': 1.3, '°': 0.7, '^': 0.7, '§': 1.0
    }
    default_weight = 1.0
    
    # Handle multiline text
    newline_positions = [pos for pos, char in enumerate(full_text) if char == '\n']
    if newline_positions:
        # Find which line contains the substring
        line_start = 0
        line_end = len(full_text)
        
        for pos in newline_positions:
            if pos < start_pos:
                line_start = pos + 1
            elif pos >= end_pos:
                line_end = pos
                break
        
        # Adjust for multi-line height (approximate)
        line_count = full_text.count('\n') + 1
        line_height = total_height / line_count
        
        # Calculate which line the substring is on (0-indexed)
        line_index = full_text[:start_pos].count('\n')
        
        # Adjust y coordinates
        y_min_adjusted = y_min + (line_index * line_height)
        y_max_adjusted = min(y_min_adjusted + line_height, y_max)
    else:
        y_min_adjusted = y_min
        y_max_adjusted = y_max
    
    # Calculate weighted lengths with fast path for specific cases
    # Single char optimization
    if len(substring) == 1:
        char_count = len(full_text)
        if char_count < 3:  # Very short text
            char_position = start_pos
            est_x_min = x_min + (total_width * (char_position / char_count))
            est_x_max = x_min + (total_width * ((char_position + 1) / char_count))
            return [est_x_min, y_min_adjusted, est_x_max, y_max_adjusted]
    
    # Calculate precise positions using character weights
    text_before = full_text[:start_pos]
    text_substr = substring
    
    # Optimize calculation of weighted lengths
    weighted_len_before = sum(char_weights.get(c, default_weight) for c in text_before)
    weighted_len_substr = sum(char_weights.get(c, default_weight) for c in text_substr)
    weighted_len_total = sum(char_weights.get(c, default_weight) for c in full_text)
    
    # Avoid division by zero
    if weighted_len_total <= 0:
        return full_bbox
    
    # Calculate proportions
    start_ratio = weighted_len_before / weighted_len_total
    substr_ratio = weighted_len_substr / weighted_len_total
    
    # Handle edge cases with leading/trailing spaces
    if text_before.endswith(' ' * min(3, len(text_before))):  # Leading spaces before substring
        # Add a small gap after spaces
        start_ratio += 0.01
    
    if text_substr.startswith(' '):  # Substring starts with space
        # Adjust for leading space in substring
        space_count = len(text_substr) - len(text_substr.lstrip(' '))
        if space_count > 0:
            space_adjustment = (space_count * char_weights.get(' ', 0.6)) / weighted_len_total
            start_ratio += space_adjustment
            substr_ratio -= space_adjustment
    
    if text_substr.endswith(' '):  # Substring ends with space
        # Adjust for trailing space in substring
        space_count = len(text_substr) - len(text_substr.rstrip(' '))
        if space_count > 0:
            space_adjustment = (space_count * char_weights.get(' ', 0.6)) / weighted_len_total
            substr_ratio -= space_adjustment
    
    # Calculate coordinates with special handling for the edges
    est_x_min = x_min + (total_width * start_ratio)
    est_x_max = est_x_min + (total_width * substr_ratio)
    
    # Boundary checks
    est_x_min = max(x_min, min(est_x_min, x_max - 1))
    est_x_max = max(est_x_min + 1, min(est_x_max, x_max))
    
    # Adaptive padding based on substring length and position
    if len(substring) < 5:  # Short substrings need more relative padding
        padding_factor = 0.03  # 3% of width for very short strings
    elif start_pos == 0 or end_pos == len(full_text):  # Start or end of text
        padding_factor = 0.015  # 1.5% padding for edge substrings
    else:
        padding_factor = 0.02  # Standard padding (2%)
    
    padding = total_width * padding_factor
    
    # Apply padding (asymmetric if needed)
    if start_pos == 0:  # If substring is at the start, add more padding to the right
        est_x_min = max(x_min, est_x_min - padding * 0.5)
        est_x_max = min(x_max, est_x_max + padding * 1.5)
    elif end_pos == len(full_text):  # If substring is at the end, add more padding to the left
        est_x_min = max(x_min, est_x_min - padding * 1.5)
        est_x_max = min(x_max, est_x_max + padding * 0.5)
    else:  # Normal case - equal padding on both sides
        est_x_min = max(x_min, est_x_min - padding)
        est_x_max = min(x_max, est_x_max + padding)
    
    return [est_x_min, y_min_adjusted, est_x_max, y_max_adjusted]
# def estimate_substring_bbox(full_text, substring, full_bbox):
#     """
#     Estimate the bounding box of a substring within a text with improved accuracy.
    
#     Args:
#         full_text: The complete text string
#         substring: The substring to locate
#         full_bbox: The bounding box of the complete text [x_min, y_min, x_max, y_max]
        
#     Returns:
#         Estimated bounding box for the substring
#     """
#     if not full_bbox or substring not in full_text:
#         return None
    
#     x_min, y_min, x_max, y_max = full_bbox
#     total_width = x_max - x_min
    
#     # Find the starting position of the substring
#     start_pos = full_text.find(substring)
#     end_pos = start_pos + len(substring)
    
#     # Define character width weights (approximate relative widths)
#     char_weights = {
#         'i': 0.4, 'j': 0.4, 'l': 0.4, 'I': 0.4, '!': 0.4, '.': 0.4, ',': 0.4, "'": 0.4, 
#         'm': 1.5, 'w': 1.5, 'W': 1.8, 'M': 1.8,
#         '@': 1.6, '#': 1.3, '%': 1.3, '&': 1.3,
#         ' ': 0.6  # Space
#     }
#     default_weight = 1.0
    
#     # Calculate weighted lengths
#     text_before = full_text[:start_pos]
#     text_substr = full_text[start_pos:end_pos]
#     total_text = full_text
    
#     weighted_len_before = sum(char_weights.get(c, default_weight) for c in text_before)
#     weighted_len_substr = sum(char_weights.get(c, default_weight) for c in text_substr)
#     weighted_len_total = sum(char_weights.get(c, default_weight) for c in total_text)
    
#     # Calculate proportions
#     if weighted_len_total > 0:
#         start_ratio = weighted_len_before / weighted_len_total
#         substr_ratio = weighted_len_substr / weighted_len_total
#     else:
#         return full_bbox  # Cannot estimate
    
#     # Calculate coordinates
#     est_x_min = x_min + (total_width * start_ratio)
#     est_x_max = est_x_min + (total_width * substr_ratio)
    
#     # Check for excessive estimates
#     est_x_max = min(est_x_max, x_max)
    
#     # Add padding for better visibility (5% of bounding box width)
#     padding = total_width * 0.02
#     est_x_min = max(x_min, est_x_min - padding)
#     est_x_max = min(x_max, est_x_max + padding)
    
#     return [est_x_min, y_min, est_x_max, y_max]
def visualize_sensitive_data(image_path, sensitive_data_bboxes, return_cv2=True):
    """
    Function to visualize sensitive data bounding boxes on an image.
    
    Args:
        image_path: Path to the original image
        sensitive_data_bboxes: Dictionary of sensitive data and their bounding boxes
        return_cv2: Whether to return a CV2 image instead of displaying with matplotlib
        
    Returns:
        CV2 image with visualized bounding boxes if return_cv2 is True
    """
    try:
        import cv2
        import numpy as np
        
        # Load the image with cv2
        img = cv2.imread(image_path)
        
        # Colors for different sensitive data (BGR format for OpenCV)
        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
        
        # Draw rectangles around each sensitive data
        color_index = 0
        for sensitive_item, bboxes in sensitive_data_bboxes.items():
            color = colors[color_index % len(colors)]
            color_index += 1
            
            for match in bboxes:
                if match["bbox"]:
                    # Bounding box format: [x_min, y_min, x_max, y_max]
                    x_min, y_min, x_max, y_max = match["bbox"]
                    
                    # Draw the rectangle
                    cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 3)
                    
                    # Add match type indicator
                    if "match_type" in match:
                        cv2.putText(img, match["match_type"], 
                                   (int(x_min), int(y_min) - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save the image for reference
        cv2.imwrite("sensitive_data_visualization.png", img)
        
        return img
        
    except ImportError:
        print("Visualization requires cv2. Please install with:")
        print("pip install opencv-python")
        return None
def mask_sensitive_data(image_path, sensitive_data_bboxes, mask_color=(0, 0, 0), 
                       border_color=(0, 0, 0), border_width=2, return_cv2=True):
    """
    Function to mask sensitive data by drawing black rectangles over them.
    
    Args:
        image_path: Path to the original image
        sensitive_data_bboxes: Dictionary of sensitive data and their bounding boxes
        mask_color: Color for the mask (default: black)
        border_color: Color for the border around the mask (default: red)
        border_width: Width of the border (default: 2)
        return_cv2: Whether to return a CV2 image
        
    Returns:
        CV2 image with masked sensitive data if return_cv2 is True
    """
    try:
        import cv2
        import numpy as np
        
        # Load the image with cv2
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            return None
            
        # Create a copy of the original image for the mask
        masked_img = img.copy()
        
        # Draw masks around each sensitive data
        for sensitive_item, bboxes in sensitive_data_bboxes.items():
            for match in bboxes:
                if match["bbox"]:
                    # Bounding box format: [x_min, y_min, x_max, y_max]
                    x_min, y_min, x_max, y_max = match["bbox"]
                    
                    # Convert coordinates to integers
                    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                    
                    # Fill the area with the mask color
                    cv2.rectangle(masked_img, (x_min, y_min), (x_max, y_max), mask_color, -1)
                    
                    # Draw border around the mask if specified
                    if border_width > 0:
                        cv2.rectangle(masked_img, (x_min, y_min), (x_max, y_max), 
                                     border_color, border_width)
        
        return masked_img
        
    except ImportError:
        print("Masking requires cv2. Please install with:")
        print("pip install opencv-python")
        return None
# Main execution
if __name__ == "__main__":
    img_path = "imgs/tender-documents-tcat-2020-11_15.png"
    ocr_result = json.load(open("imgs/tender-documents-tcat-2020-11_15_ocr.json", 'r', encoding='utf-8'))
    sensitive_data_file = "result.txt"
    
    # Find sensitive data bounding boxes
    sensitive_bboxes = find_sensitive_data_bboxes(ocr_result, sensitive_data_file)
    
    # Print findings
    print("Found sensitive data with bounding boxes:")
    for data, matches in sensitive_bboxes.items():
        print(f"\n{data}:")
        for match in matches:
            print(f"  - In text: '{match['text']}' (confidence: {match['confidence']})")
            print(f"    Bounding box: {match['bbox']}")
            print(f"    Match type: {match.get('match_type', 'unknown')}")
    
    # Visualize if the image exists
    if os.path.exists(img_path):
        visualize_sensitive_data(img_path, sensitive_bboxes)
    else:
        print(f"Image file not found: {img_path}")
        print("Visualization skipped.")