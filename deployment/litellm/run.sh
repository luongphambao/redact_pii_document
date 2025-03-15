export $(cat .env | xargs)
litellm --port 4000 --config config.yaml