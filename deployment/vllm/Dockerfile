# Use vLLM OpenAI-compatible API as the base image
FROM vllm/vllm-openai:latest

RUN pip install git+https://github.com/huggingface/transformers.git@1931a351408dbd1d0e2c4d6d7ee0eb5e8807d7bf 

# Expose the API port (default is 8000 for vLLM OpenAI API)
EXPOSE 8000

# Run vLLM API (entrypoint already defined in base image)
CMD ["python", "-m", "vllm.entrypoints.openai.api_server"]