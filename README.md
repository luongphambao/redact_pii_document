# Redact PII Document

A comprehensive tool for redacting personally identifiable information (PII) from documents using OCR and advanced text processing techniques.

## Overview
This project provides functionality to:

- Extract text from documents using OCR
- Identify and redact PII data
- Convert processed documents to markdown
- Support various document formats

## Installation

### Clone the repository:
```bash
git clone https://github.com/your-repo/redact-pii-document.git
cd redact-pii-document
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Set up conda environment (optional):
```bash
conda create --name pii-redaction python=3.9
conda activate pii-redaction
pip install -r requirements.txt
```

## Usage


### API Access
Use the API script to access functionality:
```bash
python src/api.py --port 8000
```
### Streamlit App
Use the API script to access functionality:
```bash
streamlit run src/app.py
```

## Project Structure
```
deployment/     # Deployment configurations for litellm and vllm
src/            # Core source code
```

## Docker Deployment
A Docker container is available for easy deployment:
```bash
docker build -t pii-redactor .
docker run -p 8000:8000 pii-redactor
```

## Dependencies
See `requirements.txt` for the full list of dependencies.

## License
[Include your license information here]
