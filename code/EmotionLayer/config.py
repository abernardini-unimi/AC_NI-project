import os
from dotenv import load_dotenv  # type: ignore

# Caricamento token dal file .env
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")