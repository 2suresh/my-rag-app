import os

from dotenv import load_dotenv

# Load variables from a local .env file (if present) into the environment.
# On Render the values come from the dashboard, so .env simply won't exist
# there and this call is a harmless no-op.
load_dotenv()

ENV = os.getenv("APP_ENV", "sit")  # "sit" or "prod"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# e.g. "rag-sit" vs "rag-prod"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Groq powers the LLM step (free tier). Get a key at https://console.groq.com
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Override the model via env if Groq deprecates the default.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
