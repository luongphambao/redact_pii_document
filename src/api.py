import os
import shutil
import tempfile
import uuid
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
import uvicorn
import cv2
import time

from du_prompts import EXTRACT_INFOMATION_PROMPT, DETECT_SENSITIVE_PROMPT1,CHECK_SENSITIVE_PROMPT
from markdown import ocr2markdown
from search import find_sensitive_data_bboxes, mask_sensitive_data
from process_ocr import convert_pdf_to_images, load_ocr_model, perform_ocr, save_ocr_results_as_json
from process_llm import call_model_litellm
from process_presidio import load_analyzer, detect
from utils import merge_pdf, process_llm_results

app = FastAPI(
    title="Document Redaction API",
    description="API for detecting and redacting sensitive data in documents",
    version="1.0.0"
)

# Initialize global models at startup to avoid reloading for each request
rec_model, det_model = None, None
analyzer = None

class ProcessingStatus:
    def __init__(self):
        self.jobs = {}

status_tracker = ProcessingStatus()

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global rec_model, det_model, analyzer
    
    print("Loading OCR models...")
    rec_model, det_model = load_ocr_model()
    
    print("Loading Presidio analyzer...")
    analyzer = load_analyzer()
    
    # Create necessary directories
    os.makedirs("temp", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    print("API is ready to process documents")

def combined_detection(text: str, analyzer, llm_model: str = 'openai', 
                      temperature: float = 0, key_information: str = None) -> List[str]:
    """Perform sensitive data detection using both LLM and Presidio."""
    # Use Presidio for pattern-based detection
    presidio_results = detect(text, analyzer)
    
    # Use LLM for contextual detection
    detect_sensitive_prompt = DETECT_SENSITIVE_PROMPT1.format(
        document=text, 
        key_infomation=key_information if key_information else "")
    
    llm_results = call_model_litellm(
        detect_sensitive_prompt,
        model_name=llm_model,
        temparature=temperature
    )
    
    # Process and return results
    llm_results = process_llm_results(llm_results)
    return llm_results  # Using LLM results as primary source

def process_document(job_id: str, file_path: str, use_llm: bool = True, llm_model: str = 'openai'):
    """Process a document to detect and redact sensitive data"""
    global rec_model, det_model, analyzer
    
    try:
        # Update status
        status_tracker.jobs[job_id]["status"] = "processing"
        
        # Create temporary working directory for this job
        temp_dir = os.path.join("temp", job_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Convert PDF to images
        image_paths, name = convert_pdf_to_images(file_path)
        
        # Set up output directories
        masked_dir = os.path.join(temp_dir, "masked_images")
        os.makedirs(masked_dir, exist_ok=True)
        
        masked_pdf_path = os.path.join(masked_dir, name)
        os.makedirs(masked_pdf_path, exist_ok=True)
        
        # Process each page
        list_masked_images_path = []
        
        for index, image_path in enumerate(image_paths):
            # Perform OCR
            ocr_results = perform_ocr(image_path, rec_model=rec_model, det_model=det_model)
            
            # Save OCR results
            output_file = os.path.join(temp_dir, f"page_{index+1}_ocr.json")
            result_list = save_ocr_results_as_json(ocr_results, output_file)
            
            # Convert to markdown for text processing
            markdown = ocr2markdown(result_list)
            
            # Extract key information for context if using LLM
            information_extracted = None
            if use_llm:
                extract_information_prompt = EXTRACT_INFOMATION_PROMPT.format(document=markdown)
                information_extracted = call_model_litellm(
                    extract_information_prompt,
                    model_name=llm_model,
                    temparature=0,
                    json_format=True
                )
            
            # Detect sensitive data
            result = combined_detection(
                text=markdown,
                analyzer=analyzer,
                llm_model=llm_model if use_llm else None,
                temperature=0,
                key_information=information_extracted
            )
            
            # Save sensitive data to file
            result_path = os.path.join(temp_dir, f"page_{index+1}_result.txt")
            with open(result_path, 'w') as f:
                for item in result:
                    f.write("%s\n" % item)
            
            # Find bounding boxes and mask sensitive data
            sensitive_bboxes = find_sensitive_data_bboxes(result_list, result_path)
            masked_img = mask_sensitive_data(image_path, sensitive_bboxes)
            
            masked_img_to_save_path = os.path.join(masked_pdf_path, f"masked_page_{index+1}.png")
            cv2.imwrite(masked_img_to_save_path, masked_img)
            
            list_masked_images_path.append(masked_img_to_save_path)
        
        # Create final PDF output
        pdf_final_path = os.path.join("output", f"{job_id}_masked.pdf")
        merge_pdf(list_masked_images_path, pdf_final_path)
        
        # Update job status to completed
        status_tracker.jobs[job_id]["status"] = "completed"
        status_tracker.jobs[job_id]["output_file"] = pdf_final_path
        
        # Clean up temporary files except the final output
        # shutil.rmtree(temp_dir)
        
    except Exception as e:
        status_tracker.jobs[job_id]["status"] = "failed"
        status_tracker.jobs[job_id]["error"] = str(e)
        print(f"Error processing document: {str(e)}")

@app.post("/redact-document/")
async def redact_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    use_llm: bool = True,
    llm_model: str = 'openai'
):
    """
    Upload a document (PDF or image) to redact sensitive information
    
    - **file**: The PDF or image file to process
    - **use_llm**: Whether to use LLM for enhanced detection
    - **llm_model**: LLM model to use (if use_llm is True)
    
    Returns a job ID for checking status and retrieving results
    """
    # Generate a unique job ID
    job_id = str(uuid.uuid4())
    
    # Create directory for this job
    os.makedirs(os.path.join("temp", job_id), exist_ok=True)
    
    # Save the uploaded file
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in ['.pdf', '.jpg', '.jpeg', '.png']:
        raise HTTPException(status_code=400, detail="Only PDF and image files (JPG, JPEG, PNG) are supported")
    
    # Save the uploaded file
    file_path = os.path.join("temp", job_id, f"input{file_extension}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Initialize job status
    status_tracker.jobs[job_id] = {
        "status": "queued",
        "file_name": file.filename,
        "input_file": file_path,
        "start_time": time.time()
    }
    
    # Process the document in the background
    background_tasks.add_task(process_document, job_id, file_path, use_llm, llm_model)
    
    return {"job_id": job_id, "status": "queued"}

@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    """
    Check the status of a redaction job
    
    - **job_id**: The ID of the redaction job
    
    Returns the current status and details of the job
    """
    if job_id not in status_tracker.jobs:
        raise HTTPException(status_code=404, detail=f"Job ID {job_id} not found")
    
    job_info = status_tracker.jobs[job_id]
    
    # Calculate processing time
    if "start_time" in job_info:
        elapsed = time.time() - job_info["start_time"]
        job_info["processing_time"] = f"{elapsed:.2f} seconds"
    
    return job_info

@app.get("/download-result/{job_id}")
async def download_result(job_id: str):
    """
    Download the redacted document
    
    - **job_id**: The ID of the redaction job
    
    Returns the redacted document file
    """
    if job_id not in status_tracker.jobs:
        raise HTTPException(status_code=404, detail=f"Job ID {job_id} not found")
    
    job_info = status_tracker.jobs[job_id]
    
    if job_info["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job is not completed yet. Current status: {job_info['status']}")
    
    if "output_file" not in job_info:
        raise HTTPException(status_code=500, detail="Output file information not found")
    
    return FileResponse(
        job_info["output_file"], 
        media_type="application/pdf", 
        filename=f"redacted_{job_info['file_name']}"
    )

@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job and its associated files
    
    - **job_id**: The ID of the redaction job
    """
    if job_id not in status_tracker.jobs:
        raise HTTPException(status_code=404, detail=f"Job ID {job_id} not found")
    
    # Clean up job files
    try:
        job_temp_dir = os.path.join("temp", job_id)
        if os.path.exists(job_temp_dir):
            shutil.rmtree(job_temp_dir)
            
        output_file = status_tracker.jobs[job_id].get("output_file")
        if output_file and os.path.exists(output_file):
            os.remove(output_file)
            
        # Remove job from tracker
        del status_tracker.jobs[job_id]
        
        return {"message": f"Job {job_id} and associated files deleted"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting job: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)