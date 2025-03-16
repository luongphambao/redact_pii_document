EXTRACT_INFOMATION_PROMPT ="""You are a document analysis expert with a deep understanding of various types of texts (e.g., contracts, financial reports, medical records, job applications, etc.).

**Task:**

1.  **Identify Document Type:** Based on the content of the provided document, clearly identify what type of document this is (e.g., "Employment Contract," "Medical Record," "Q3 Financial Report"). Briefly explain the reason for your conclusion.

2.  **Extract Key Information and Explain Meaning:** Carefully read the document and extract all key information. For each piece of information, provide:
    *   **Extracted Information:** (e.g., "Name: John Doe," "Account Number: 1234567890," "Date of Birth: 01/01/1990").
3.  **Special Notes:**
    *   **Ensure Complete Extraction:** Do not omit any important information, including identifying information, contact information, financial information, medical information, legal information, etc.
    *   **Prioritize Potentially Sensitive Information:** Pay particular attention to information that could be considered sensitive or PII (Personally Identifiable Information).
    *   **Provide Structured Formatting:** Present the results in an easy-to-read and easy-to-process format (e.g., list, table, or JSON format).
4.  **Conciseness:** Sensitive data and PII will *typically* be very short, usually *less than 3 words*. Think names, phone numbers, ID numbers, etc.
5 . **Output Format:**
    return the extracted information in JSON format.
    ```json
    {{
        "document_type": "Type of Document",
        "infomation1" "value1"
        "infomation2" "value2"
        "infomation3" "value3"
        ...
    }}
    ```
**Here is the document to analyze:**\n{document}
"""

DETECT_SENSITIVE_PROMPT1 = """You are a data security expert with in-depth knowledge of various types of sensitive data and global data protection regulations (e.g., GDPR, CCPA, HIPAA).  

### **Task:**  
Analyze the provided document and identify all instances of sensitive data or Personally Identifiable Information (PII).  

### **Input:**  
- **Document:** {document}  
- **Extracted Information:** {key_infomation}  

### **Output Requirements:**  

**1. Extract ALL Sensitive Data & PII**  
- Identify every occurrence of sensitive data or PII exactly as it appears in the document.  
    Example: 
        Customer ID :  SGSIN001 => SGSIN001

 **3. Maintain Accuracy & Completeness**  
- Extract **only** existing PII/sensitive data—do not infer or generate new information.  

**4. Enforce Conciseness**  
- Typically, sensitive data is **≤3 words** (e.g., names, ID numbers, phone numbers).  
- If the data spans multiple lines, return each line separately.  
 **5. Standardized Output Format**  
- Each extracted item should be listed **line by line**.  
- If no sensitive data is found, return:  `----------------`
<output>    
John Doe
123-456-
7890
1234-5678-9012
</output>
Sensitive data includes (but is not limited to):  
- **Personal Identifiers:** Full name, date of birth, ID numbers (SSN, passport, driver's license, national ID).  
- **Contact Information:** Phone numbers, email addresses, residential addresses.  
- **Financial Data:** Credit card numbers, bank account numbers, tax IDs.  
- **Medical & Health Data:** Health conditions, insurance numbers, medical records.  
- **Sensitive Attributes:** Religious beliefs, political affiliations, criminal history, biometric data, employment records, educational background.  
"""

# DETECT_SENSITIVE_PROMPT = """You are an expert in data security and privacy, specializing in Personally Identifiable Information (PII) and sensitive data detection. Your primary role is to analyze a given document and identify any text segments that may constitute PII or other sensitive information.

# **Instructions:**

# 1.  **Analyze the document thoroughly:** Carefully examine the provided text for any potential PII or sensitive data.
# 2.  **Identify sensitive text segments:**  Pinpoint the exact portions of the text that contain sensitive information.
# 3.  **Return only the sensitive text:** Your output should consist *exclusively* of the identified sensitive text segments. Do not include any surrounding context, explanations, or introductory phrases.  If no sensitive information is found, return an empty string.
# 4. **Prioritize accuracy:** It is better to have false positives (identifying non-sensitive data as sensitive) than to miss actual sensitive information.

# **Examples of PII and Sensitive Data:**

# *   Full names, partial names with context
# *   Email addresses
# *   Phone numbers
# *   Physical addresses (street address, city, state, zip code)
# *   Social Security numbers (or similar national identification numbers)
# *   Credit card numbers
# *   Bank account numbers
# *   Medical information
# *   Login credentials (usernames, passwords)
# *   IP Addresses
# *   Geolocation data
# *   Any information that, when combined with other data, could identify an individual.

# **Here is the document to analyze:**\n{document}
# """
PROMPT2 = """
Identify and replace all entities in the following text that are either 
persons, institutions, or places with a unique identifier throughout the 
entire text. This includes replacing both the full entity name and any 
partial occurrences of the name (e.g., replacing "Micky Mouse" and "Micky" 
with the same identifier).

Steps:

    * Identify all entities in the text that are classified as persons, institutions, or places.
    * Assign a unique identifier to each entity.
    * Replace every occurrence of the full name and any partial name with the corresponding identifier in the text.
    * Print the processed text with all entities replaced by their respective identifiers. Do NOT print the original text but only the processed version.
    * Provide a JSON-formatted list of all replaced entities as pairs, where each pair consists of the entity name and its corresponding identifier.

Example:

    Original text: "Micky Mouse is a character created by Walt Disney."
    Processed text: "ENTITY_1 is a character created by ENTITY_2."
    JSON output:

[
  {"entity": "Micky Mouse", "identifier": "ENTITY_1"},
  {"entity": "Micky", "identifier": "ENTITY_1"},
  {"entity": "Walt Disney", "identifier": "ENTITY_2"},
  {"entity": "Walt", "identifier": "ENTITY_2"}
]"""
CHECK_SENSITIVE_PROMPT = """You are a cybersecurity expert with extensive knowledge of Personally Identifiable Information (PII). I have extracted sensitive data from a document, and you need to help me to recheck  PII or sensitive data. Be extremely careful and only remove information that you are absolutely certain is not sensitive. If there is any doubt about whether a piece of data could be PII, keep it.

Input format example (list of strings):

'<output>  ', 'SATS Food Procurement', '65418673', 'SATS INFLIGHT CATERING CENTRE 1', '20 AIRPORT BOULEVARD  ', 'SINGAPORE 819659', '----------------  ', '</output>'

**Output format example**:
['SATS Food Procurement', '65418673', 'SATS INFLIGHT CATERING CENTRE 1', '20 AIRPORT BOULEVARD ', 'SINGAPORE 819659']
**Important Notes**:
PII includes (but is not limited to): Name, address, phone number, email address, credit card number, social security number, medical information, etc.
Top Priority: Avoid mistakenly removing PII. It is better to keep non-sensitive information than to remove sensitive information.
Please return extractly output as output format
Here is a list of extracted sensitive data that needs to be rechecked:
{sensitive_data}"""
RECOMMEND_SENSITIVE_PROMPT = """You are a cybersecurity expert tasked with reviewing a document to identify potential sensitive data that should be redacted or protected. Your goal is to recommend the appropriate actions to secure the sensitive information in the document.
Please recommend potential infomatics that should be redacted or protected based on the following types of sensitive data:"""

DEFAULT_PII_TMPL = (
    "The current context information is provided. \n"
    "A task is also provided to mask the PII within the context. \n"
    "Return the text, with all PII masked out, and a mapping of the original PII "
    "to the masked PII. \n"
    "Return the output of the task in JSON. \n"
    "Context:\n"
    "Hello Zhang Wei, I am John. "
    "Your AnyCompany Financial Services, "
    "LLC credit card account 1111-0000-1111-0008 "
    "has a minimum payment of $24.53 that is due "
    "by July 31st. Based on your autopay settings, we will withdraw your payment. "
    "Task: Mask out the PII, replace each PII with a tag, and return the text. Return the mapping in JSON. \n"
    "Output: \n"
    "Hello [NAME1], I am [NAME2]. "
    "Your AnyCompany Financial Services, "
    "LLC credit card account [CREDIT_CARD_NUMBER] "
    "has a minimum payment of $24.53 that is due "
    "by [DATE_TIME]. Based on your autopay settings, we will withdraw your payment. "
    "Output Mapping:\n"
    '{{"NAME1": "Zhang Wei", "NAME2": "John", "CREDIT_CARD_NUMBER": "1111-0000-1111-0008", "DATE_TIME": "July 31st"}}\n'
    "Context:\n{context_str}\n"
    "Task: {query_str}\n"
    "Output: \n"
    ""
)
"""You are an expert in data extraction and analysis. I am providing you with a document (e.g., PDF, DOCX, Excel, etc.) that contains various types of information. Your task is to analyze the document and extract all sensitive data and personally identifiable information (PII) accurately and comprehensively. The extracted information must be based solely on the content within the document and presented in a clear, structured format.

Sensitive data and PII to extract include, but are not limited to:
- Full names
- Dates of birth
- Addresses (e.g., street, city, state, country)
- Phone numbers
- Email addresses
- Social security numbers or other identification numbers (e.g., passport, driver's license)
- Financial information (e.g., bank account numbers, credit card numbers)
- Employment details (e.g., job titles, employee IDs)
- Any other personal or confidential information explicitly present in the document

Instructions:
1. Carefully scan the entire document to identify and extract all instances of the above information.
2. Do not infer or generate data that is not explicitly stated in the document.
3. Organize the extracted data into a table or list with clear labels (e.g., "Name", "Phone Number", "Address").
4. If a specific type of PII is not found, indicate it as "Not present" or "N/A".
5. If the document contains ambiguous or unclear data, note it and provide the exact text as it appears.
6. Return the results in a structured format, such as a table or bullet points.

Please confirm once you have received the document and begin the extraction process. If no document is provided yet, ask me to upload one."""

prompt="""You are a data de-identification assistant. Your job is to point out all the personal information from the given text.
    Examples of personal information include:  (full name, maiden name, or any other names by which the individual is known), Contact Information (Addresses, phone numbers, email addresses, and social media profiles),
    Identification Number (Social Security numbers, passport numbers, driver's license numbers, and other government-issued identification numbers),
    Financial Information (Bank account numbers, credit card numbers, and financial records),
    Date of Birth, Health Information (Medical records, health insurance information), Employment information (job titles, employer names, and contact information), Online activities ( IP addresses, usernames, and online behavior such as website browser history),
    Education Information (School names, major), or any other information that you consider as personal. 

    You should be very careful about identifying if the information is personal. If you are not confident about a word or a phrase, then you should include it as personal information because we don't want any information to be leaked.

    You should always respond back by a list of words that you consider as personal information. DO NOT INCLUDE any other words in your response.
    Example response should be:
    ["Joe Biden", "America", "400-923-1093"]
    
    Following are the text, please provide the list of personal information as required:{text}"""