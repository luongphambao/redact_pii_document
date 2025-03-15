from PIL import Image
import json 
def merge_pdf(images_path,output_pdf_path):
    """Merge images into a single PDF file.
    Args:
        images_path (list): List of image paths to merge.
        output_pdf_path (str): Path to save the output PDF file."""
    images = []
    for image_path in images_path:
        images.append(Image.open(image_path))
    images[0].save(output_pdf_path, save_all=True, append_images=images[1:])
    return output_pdf_path
def convert_str_to_json(json_str:str):
    """Convert JSON string to JSON object"""
    return json.loads(json_str) 
def parse_result(result:dict):
    """"Parse JSON Result to list of sensitive data
    Example JSON:
    {
    "Customer ID": {
        "data_type": "Identifier",
        "value": "SGSIN001"
    },
    "License ID": {
        "data_type": "Identifier",
        "value": "9ae122c6b22ea947f6ad2689c28efbb8efa1c425"
    },
    "License Start Date": {
        "data_type": "Date",
        "value": "01 JAN 2024"
    },
    "License End Date": {
        "data_type": "Date",
        "value": "31 DEC 2025"
    },
    "Registered Contacts": [
        {
            "Name": {
                "data_type": "Name",
                "value": "Simon Lee"
            },
            "Email": {
                "data_type": "Email",
                "value": "SimonIts@singtel.com"
            }
        },
        {
            "Name": {
                "data_type": "Name",
                "value": "Andrew Lim"
            },
            "Email": {
                "data_type": "Email",
                "value": "andrew.lim@singtel.com"
            }
        }
    ]
}"""
    sensitive_data = []
    for key in result:
        if isinstance(result[key],dict):
            sensitive_data.append(result[key]['value'] )
        elif isinstance(result[key],list):
            for item in result[key]:
                for key1 in item:
                    sensitive_data.append(item[key1]['value'])
    return sensitive_data   
def process_llm_results(llm_results:str):
    """Process LLM results and return list of sensitive data"""
    llm_results+= "\n"
    #replace " and ' with ""
    
    llm_results = llm_results.replace("```","").strip().split("\n")
    llm_results = [ item.strip() for item in llm_results]
    #replace \n with space
    llm_results = [ item.replace("\n"," ") for item in llm_results]
    llm_results = [item if item != "" else None for item in llm_results]
    llm_results = list(filter(None,llm_results))
    return llm_results
def process_llm_to_list(llm_results:str):
    """Process LLM results and return list of sensitive data
    ```SATS Food Procurement  ', '65418673  ', 'SATS INFLIGHT CATERING CENTRE 1  ', '20 AIRPORT BOULEVARD  ', 'SINGAPORE 819659'````"""
    #result_list = []
    #print("LLM Results: ",llm_results)
    llm_results = llm_results.replace("'","").replace('"','')
    llm_results = llm_results.replace("[","").replace("]","").replace("``","").strip().split(",")
    
    llm_results = [ item.strip() for item in llm_results]
    llm_results = [item if item != "" else None for item in llm_results]
    llm_results = list(filter(None,llm_results))
    return llm_results
if __name__=="__main__":
    str_llm_result = """
```'1 JAN 2024, '9ae122c6b22ea947f6ad2689c28efbb8efa1c425', 'andrew.li', 'Andrew Lim', 'SGSIN001', 'Simon Lee', 'SimonIts@singtel.com', 'singtel.com', '4600', '31 DEC 2025', 'andrew.lim@singtel.com'```
"""
    llm_results = process_llm_to_list(str_llm_result)
    print(llm_results)