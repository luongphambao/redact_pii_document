from PIL import Image
from base import client,client_google
from io import BytesIO, StringIO
from openai import APIError, RateLimitError
from fastapi import HTTPException
import base64
from du_prompts import DETECT_SENSITIVE_PROMPT1,EXTRACT_INFOMATION_PROMPT

def call_model_litellm(prompt:str,model_name: str="openai",temparature:float=0.0,json_format=False):
    """
    Call model from LLM host:
    Args:
    prompt: str: prompt to be passed to model
    model_name: str: model name to be called from LLM host
    temparature: float: temparature value
    json_format: bool: json format
    Returns:
    str: response from model
    """ 
    print("model_name", model_name)
    if model_name == "gemini-2.0-flash":
        respone = client_google.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return respone.choices[0].message.content

    try:
        if json_format:
            final_prompt = [{'role': 'system', 'content': prompt}]
            response = client.chat.completions.create(
                model=model_name,
                messages=final_prompt,
                response_format={"type": "json_object"},
                temperature=temparature
            )
            #return response.choices[0].message.content
        else:
            final_prompt = [{'role': 'system', 'content': prompt}]
            response = client.chat.completions.create(
                model=model_name,
                messages=final_prompt,
                #temperature=temparature,
                #n=1  #52s  
            )
        return response.choices[0].message.content
    except RateLimitError as e:
        raise HTTPException(status_code=429, detail='Rate limit exceeded')
    except APIError as e:
        raise HTTPException(status_code=500, detail='Internal server error')
def call_model_litellm_vision(prompt:str,image:Image,model_name: str= "openai",temparature:float=0.0,max_tokens:int=4096):
    """
    Call model vision from LLM host:
    Args:
    prompt: str: prompt to be passed to model
    """
    def encode_image(image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    base64_image = encode_image(image)
    final_prompt = [{
    "role": "user",
    "content": [
        {
        "type": "text",
        "text": prompt
        },
        {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
        }
        }
    ]
    }]
    response = client.chat.completions.create(
        model=model_name,
        messages=final_prompt,
        stream=False,
        temperature=temparature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content
if __name__=="__main__":
    #Test call_model_litellm
    import time 
    document = open("Final Sec Attendee List.md").read()
    # extract_infomation_prompt = EXTRACT_INFOMATION_PROMPT.format(document=document)
    # infomation_extracted = call_model_litellm(extract_infomation_prompt,model_name ='openai',temparature=0,json_format=True)
    # print(infomation_extracted)
    
    #convert json to str 
    import json
    infomation_extracted = ""
    pii_detect_prompt = DETECT_SENSITIVE_PROMPT1.format(document=document,key_infomation=infomation_extracted)
    with open("prompt.txt","w") as f:
        f.write(pii_detect_prompt)
    result = call_model_litellm(pii_detect_prompt,model_name ='openai',temparature=0)
    print(result)
    # detect_image_senstive_prompt = """Phân tích tài liệu hình ảnh này và xác định bất kỳ thông tin nào có thể được coi là riêng tư, nhạy cảm, hoặc có thể gây hại nếu bị tiết lộ hoặc sử dụng sai mục đích. Hãy trích xuất tất cả những thông tin này"

    # """
    
    # image = Image.open("imgs/tender-documents-tcat-2020-11_15.png")
    # result = call_model_litellm_vision(detect_image_senstive_prompt,image,model_name ='openai',temparature=0)
    # print(result)