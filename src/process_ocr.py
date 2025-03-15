import os
import json
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from pdf2image import convert_from_bytes

def load_ocr_model():
    """
    Load the OCR model.
    
    Returns:
        OCR model
    """
    recognition_model = RecognitionPredictor()
    detection_model = DetectionPredictor()
    return recognition_model, detection_model

def convert_pdf_to_images(pdf_path, output_dir="imgs"):
    """
    Convert PDF to images and save them to disk.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save the images
        
    Returns:
        List of paths to the saved images
    """
    name = os.path.basename(pdf_path).split(".")[0]
    os.makedirs(output_dir, exist_ok=True)
    
    images = convert_from_bytes(open(pdf_path, "rb").read())
    image_paths = []
    
    for i, image in enumerate(images):
        image_path = f"{output_dir}/{name}_{i}.png"
        image.save(image_path, "PNG")
        image_paths.append(image_path)
        
    return image_paths, name




def perform_ocr(image_path, languages=None, rec_model=None, det_model=None):
    """
    Perform OCR on an image using pre-loaded models.
    
    Args:
        image_path: Path to the image
        languages: List of languages to use for OCR (e.g. ["en"])
        rec_model: Recognition model (optional, uses module-level model if None)
        det_model: Detection model (optional, uses module-level model if None)
        
    Returns:
        OCR results
    """
    if languages is None:
        languages = ["en"]
    
    # Use provided models or fallback to module-level models
    recognition_predictor = rec_model 
    detection_predictor = det_model 
    if recognition_predictor is None:
        recognition_predictor = RecognitionPredictor()
    if detection_predictor is None:
        detection_predictor = DetectionPredictor()
    
    # Load image
    image = Image.open(image_path)
    
    # Perform OCR
    predictions = recognition_predictor([image], [languages], detection_predictor)
    return predictions[0]

def save_ocr_results_as_json(ocr_results, output_file="output.json"):
    """
    Save OCR results to a JSON file.
    
    Args:
        ocr_results: OCR results object
        output_file: Path to save the JSON file
        
    Returns:
        List of dictionaries containing OCR results
    """
    result_list = []
    
    for res in ocr_results.text_lines:
        result_dict = {
            "polygon": res.polygon,
            "text": res.text,
            "confidence": res.confidence,
            "bbox": res.bbox
        }
        result_list.append(result_dict)
        
    with open(output_file, "w+") as f:
        json.dump(result_list, f)
        
    return result_list


def visualize_ocr_results(image_path, ocr_results, output_dir="output", highlight_opacity=0.5):
    """
    Visualize OCR results on the image.
    
    Args:
        image_path: Path to the original image
        ocr_results: OCR results object or list of result dictionaries
        output_dir: Directory to save output images
        highlight_opacity: Opacity for the highlighted regions (0-1)
        
    Returns:
        Path to the final output image
    """
    os.makedirs(output_dir, exist_ok=True)
    name = os.path.basename(image_path).split(".")[0]
    
    # Load image with OpenCV for processing
    img_cv2 = cv2.imread(image_path)
    mask = np.zeros_like(img_cv2)
    
    # Handle both OCR result object and list of dictionaries
    results_to_process = ocr_results.text_lines if hasattr(ocr_results, 'text_lines') else ocr_results
    
    for i, result in enumerate(results_to_process):
        # Extract polygon and bbox data based on input type
        if hasattr(result, 'polygon'):
            polygon = result.polygon
            text = result.text
        else:
            polygon = result["polygon"]
            text = result["text"]
            
        # Get coordinates
        if len(polygon) >= 4 and all(len(point) >= 2 for point in polygon):
            x1, y1 = polygon[0]
            x2, y2 = polygon[1]
            x3, y3 = polygon[2]
            x4, y4 = polygon[3]
            
            xmin = min(x1, x2, x3, x4)
            ymin = min(y1, y2, y3, y4)
            xmax = max(x1, x2, x3, x4)
            ymax = max(y1, y2, y3, y4)
            
            # Draw rectangle on the image
            cv2.rectangle(img_cv2, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            
            # Create mask for highlighting
            box_mask = np.zeros_like(img_cv2)
            box_mask = cv2.fillPoly(box_mask, [np.array(polygon).astype(np.int32)], (255, 0, 0))
            
            # Apply the mask to the image with given opacity
            img_cv2 = cv2.addWeighted(img_cv2, 1, box_mask, highlight_opacity, 0)
            
            # Optionally save each step
            output_path = f"{output_dir}/{name}_{i}.png"
            cv2.imwrite(output_path, img_cv2)
    
    # Save final result
    final_output_path = f"{output_dir}/{name}_final.png"
    cv2.imwrite(final_output_path, img_cv2)
    
    return final_output_path


def highlight_sensitive_data(image_path, ocr_results_json, sensitive_data_file, output_dir="output"):
    """
    Highlight sensitive data in an image based on a list of sensitive terms.
    
    Args:
        image_path: Path to the original image
        ocr_results_json: Path to the JSON file containing OCR results or list of result dictionaries
        sensitive_data_file: Path to file containing sensitive data to highlight
        output_dir: Directory to save output images
        
    Returns:
        Path to the output image with highlighted sensitive data
    """
    # Load OCR results
    if isinstance(ocr_results_json, str):
        with open(ocr_results_json, "r") as f:
            ocr_json = json.load(f)
    else:
        ocr_json = ocr_results_json
    
    # Read sensitive data
    with open(sensitive_data_file, "r") as f:
        sensitive_lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # Load image
    img = cv2.imread(image_path)
    name = os.path.basename(image_path).split(".")[0]
    os.makedirs(output_dir, exist_ok=True)
    
    # Colors for different sensitive data (BGR format for OpenCV)
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    
    # For each sensitive data item
    for i, sensitive_item in enumerate(sensitive_lines):
        color = colors[i % len(colors)]
        
        # Find all occurrences in the OCR results
        for ocr_item in ocr_json:
            ocr_text = ocr_item.get("text", "")
            
            # Process exact or partial matches
            if sensitive_item in ocr_text:
                # Get bounding box
                if sensitive_item == ocr_text:
                    # Exact match
                    bbox = ocr_item.get("bbox")
                else:
                    # Partial match - estimate substring position
                    bbox = estimate_substring_bbox(ocr_text, sensitive_item, ocr_item.get("bbox"))
                
                if bbox:
                    x_min, y_min, x_max, y_max = bbox
                    # Draw rectangle for the sensitive data
                    cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
                    
                    # Add a semi-transparent overlay
                    overlay = img.copy()
                    cv2.rectangle(overlay, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, -1)
                    img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
    
    # Save the result
    output_path = f"{output_dir}/{name}_sensitive_data.png"
    cv2.imwrite(output_path, img)
    
    return output_path


def estimate_substring_bbox(full_text, substring, full_bbox):
    """
    Estimate the bounding box of a substring within a text.
    
    Args:
        full_text: The complete text string
        substring: The substring to locate
        full_bbox: The bounding box of the complete text [x_min, y_min, x_max, y_max]
        
    Returns:
        Estimated bounding box for the substring
    """
    if not full_bbox or substring not in full_text:
        return None
    
    x_min, y_min, x_max, y_max = full_bbox
    total_width = x_max - x_min
    
    # Find the starting position of the substring
    start_pos = full_text.find(substring)
    end_pos = start_pos + len(substring)
    
    # Calculate the proportional position
    if len(full_text) > 0:
        start_ratio = start_pos / len(full_text)
        end_ratio = end_pos / len(full_text)
    else:
        return full_bbox  # Cannot estimate, return the full bbox
    
    # Calculate the estimated coordinates
    est_x_min = x_min + (total_width * start_ratio)
    est_x_max = x_min + (total_width * end_ratio)
    
    return [est_x_min, y_min, est_x_max, y_max]


def main(pdf_path, sensitive_data_file=None):
    """
    Main function to process a PDF and highlight sensitive data.
    
    Args:
        pdf_path: Path to the PDF file
        sensitive_data_file: Path to file containing sensitive data to highlight
        
    Returns:
        List of paths to processed images
    """
    # Convert PDF to images
    image_paths, name = convert_pdf_to_images(pdf_path)
    results = []
    
    rec_model,det_model = load_ocr_model()
    # Load models once
    # rec_model = RecognitionPredictor()
    # det_model = DetectionPredictor()
    
    # Process each page
    for image_path in image_paths:
        # Perform OCR with pre-loaded models
        ocr_results = perform_ocr(image_path, rec_model=rec_model, det_model=det_model)
        
        # Rest of the processing...
        output_file = f"{os.path.splitext(image_path)[0]}_ocr.json"
        result_list = save_ocr_results_as_json(ocr_results, output_file)
        
        # Visualize OCR results
        output_path = visualize_ocr_results(image_path, ocr_results)
        results.append(output_path)
        
        # If sensitive data file is provided, highlight sensitive data
        if sensitive_data_file and os.path.exists(sensitive_data_file):
            sensitive_output = highlight_sensitive_data(image_path, result_list, sensitive_data_file)
            results.append(sensitive_output)
    
    return results

if __name__ == "__main__":
    pdf_path = "../documents/tender-documents-tcat-2020-11.pdf"
    sensitive_data_file = "output1_result.txt"  # Optional, set to None if not needed
    
    processed_images = main(pdf_path, sensitive_data_file)
    output_pdf_path = "output.pdf"
    #merge_pdf(processed_images,output_pdf_path)
    print(f"Processed images: {processed_images}")