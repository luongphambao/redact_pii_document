import json
import cv2
import os
from typing import List, Dict, Any, Set

from du_prompts import EXTRACT_INFOMATION_PROMPT, DETECT_SENSITIVE_PROMPT1
from markdown import ocr2markdown
from search import find_sensitive_data_bboxes, mask_sensitive_data
from process_ocr import convert_pdf_to_images, load_ocr_model, perform_ocr, save_ocr_results_as_json
from process_llm import call_model_litellm
from process_presidio import load_analyzer, detect
from utils import merge_pdf, parse_result, convert_str_to_json, process_llm_results

def combine_sensitive_data(llm_results: List[str], presidio_results: List[Dict[str, Any]]) -> List[str]:
    """
    Combine and deduplicate sensitive data from LLM and Presidio.
    
    Args:
        llm_results: List of sensitive strings detected by the LLM
        presidio_results: List of dictionaries with Presidio detection results
        
    Returns:
        Combined list of unique sensitive data strings
    """
    # Create a set to store unique sensitive data items
    #combined_results = set(llm_results)
    presidio_results = [ item for item in presidio_results ] # Remove PERSON entity
    print("Presidio results: ", presidio_results)
    print("LLM results: ", llm_results)
    combined_results = list(set(llm_results + presidio_results))
    #combined_results = list(set(presidio_results + llm_results))
    return llm_results
def combined_detection(text: str, analyzer, llm_model: str = 'openai', 
                      temperature: float = 0, key_information: str = None) -> List[str]:
    """
    Perform sensitive data detection using both LLM and Presidio.
    
    Args:
        text: Text to analyze
        analyzer: Presidio analyzer
        llm_model: LLM model to use
        temperature: Temperature for LLM
        key_information: Optional extracted key information for context
        
    Returns:
        Combined list of sensitive data
    """
    # Use Presidio for pattern-based detection
    presidio_results = detect(text, analyzer)
    print(key_information)
    # Use LLM for contextual detection
    detect_sensitive_prompt = DETECT_SENSITIVE_PROMPT1.format(
        document=text, 
        key_infomation=key_information if key_information else "")
    
    llm_results = call_model_litellm(
        detect_sensitive_prompt,
        model_name=llm_model,
        temparature=temperature
    )
    
    # Combine results
    llm_results = process_llm_results(llm_results)
    combined_results = combine_sensitive_data(llm_results, presidio_results)
    
    return combined_results

# Main execution
if __name__ == "__main__":
    pdf_path = "../documents/GVT(LT)24002 Part 1 Section A Instructions to Tenderers.pdf"
    
    # Initialize models and directories
    image_paths, name = convert_pdf_to_images(pdf_path)
    results = []
    pdf_final_path = "final/" + name + "_masked.pdf"
    rec_model, det_model = load_ocr_model()
    analyzer = load_analyzer()
    
    masked_dir = "masked_images"
    os.makedirs(masked_dir, exist_ok=True)
    masked_pdf_path = os.path.join(masked_dir, name)
    os.makedirs(masked_pdf_path, exist_ok=True)
    
    # Process each page
    index = 0
    list_masked_images_path = []
    
    for image_path in image_paths:
        # Perform OCR with pre-loaded models
        ocr_results = perform_ocr(image_path, rec_model=rec_model, det_model=det_model)
        
        # Save OCR results
        output_file = f"{os.path.splitext(image_path)[0]}_ocr.json"
        result_list = save_ocr_results_as_json(ocr_results, output_file)
        
        # Convert to markdown for text processing
        markdown = ocr2markdown(result_list)
        with open(os.path.join("markdowns", f"{name}_page_{index+1}.md"), 'w') as f:
            f.write(markdown)
        
        # Extract key information to provide context
        extract_infomation_prompt = EXTRACT_INFOMATION_PROMPT.format(document=markdown)
        information_extracted = call_model_litellm(
            extract_infomation_prompt,
            model_name='openai',
            temparature=0,
            json_format=True
        )
        
        # Perform combined detection using both LLM and Presidio
        result = combined_detection(
            text=markdown,
            analyzer=analyzer,
            llm_model='openai',
            temperature=0,
            key_information=information_extracted
        )
        
        # Log detection results
        print("Combined sensitive data detected:", result)
        
        # Save sensitive data to file
        result_path = f"{os.path.splitext(image_path)[0]}_result.txt"
        with open(result_path, 'w') as f:
            for item in result:
                f.write("%s\n" % item)
        
        # Find bounding boxes and mask sensitive data
        sensitive_bboxes = find_sensitive_data_bboxes(result_list, result_path)
        masked_img = mask_sensitive_data(image_path, sensitive_bboxes)
        masked_img_to_save_path = os.path.join(masked_pdf_path, f"masked_page_{index+1}.png")
        print(f"Saving masked image to: {masked_img_to_save_path}")
        cv2.imwrite(masked_img_to_save_path, masked_img)
        
        # Visualization option (commented out)
        # img_result = visualize_sensitive_data(image_path, sensitive_bboxes)
        # cv2.imwrite(f"../imgs/sensitive_data_visualization_{index}.png", img_result)
        
        index += 1
        results.append(result)
        list_masked_images_path.append(masked_img_to_save_path)
    
    # Merge all masked pages into a single PDF
    merge_pdf(list_masked_images_path, pdf_final_path)
    print(f"Final masked PDF saved to: {pdf_final_path}")