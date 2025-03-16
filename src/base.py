import openai
import os
from dotenv import load_dotenv
from project_config import ENV_PATH
_ = load_dotenv(ENV_PATH)

print("ENV_PATH", ENV_PATH)

client = openai.OpenAI(
    api_key='anything',
    base_url="http://0.0.0.0:4000",
)
client_google = openai.OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
