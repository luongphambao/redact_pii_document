import os

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

ENV_PATH = os.path.join(ROOT_DIRECTORY, '.env')

# OCR config
OPENAI_MODEL_NAME = "gpt-4o"
OPENAI_GPT4_MODEL_NAME = "gpt-4o"
OPENAI_MODEL_LIST = ['gpt-3', 'gpt-4']

ENGINES = ['azure', 'google']

# AZURE OPENAI
AZURE_OPENAI_VERSION = "2023-03-15-preview"
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4o-mini"

OUTPUT_DIR = os.path.join(ROOT_DIRECTORY, 'tmp_output')

# Vision prompt 
GEMINI_VISION_PROMPT = "extract all the text into markdown format. If there is table, return markdown table format. If there is checkbox, return into checkbox format. Language: Vietnamese"
OPENAI_VISION_PROMPT = "Read the image contains important information thoroughly and extract all information, \
organizing it clearly for easy reading. Dont miss any word. The image contains header and clothes tags \
, return header first, then return content by each tag. Here\'s the image:"

