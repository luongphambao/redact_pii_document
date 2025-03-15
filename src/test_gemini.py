
from litellm import completion
import os

os.environ['GEMINI_API_KEY'] = os.getenv('GENAI_API_KEY')
response = completion(
    model="gemini/gemini-2.0-flash", 
    messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}]
)
print(response.choices[0].message.content)