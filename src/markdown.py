import json
import numpy as np
from ocr2tex2d import ocr2structured_text, get_ocr_from_form_recognizer
import sys

#using OCR result to reconstruct to 2d markdown
"""
Input: OCR result from Surya OCR:
List of line:  {"polygon": [[1070.0, 46.0], [1996.0, 46.0], [1996.0, 91.0], [1070.0, 91.0]], "text": "Why you need a last line of defence", "confidence": 0.93359375, "bbox": [1070.0, 46.0, 1996.0, 91.0]}
Output: 2D markdown:

"""
def ocr2markdown(ocr_results):
    words, boxes, conf_scores = get_ocr_from_form_recognizer(ocr_results, level='line')
    
    # Calculate image width from the bounding boxes if not provided in the JSON
    # Use the maximum x-coordinate from all boxes
    image_width = max(box[2] for box in boxes) if boxes else 2000  # Default if no boxes
    
    plain_text = ocr2structured_text(words, boxes, conf_scores, image_width)
    return plain_text

if __name__ == "__main__":
    ocr_results = json.load(open("imgs/tender-documents-tcat-2020-11_15_ocr.json", 'r', encoding='utf-8'))
    markdown = ocr2markdown(ocr_results)
    print(markdown)
    with open("tender-documents-tcat-2020-11_15_ocr.md", "w+") as f:
        f.write(markdown)