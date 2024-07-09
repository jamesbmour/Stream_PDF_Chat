import os 
# import chromadb
#
# from chromadb.config import Settings
# CHROMA_SETTINGS = Settings(
#         chroma_db_impl='duckdb+parquet',
#         persist_directory='db',
#         anonymized_telemetry=False
# )

OPENROUTER_REFERRER = "https://github.com/alexanderatallah/openrouter-streamlit"
# OPENROUTER_BASE = "http://localhost:3000"
OPENROUTER_BASE = "https://openrouter.ai"
OPENROUTER_API_BASE = f"{OPENROUTER_BASE}/api/v1"
OPENROUTER_DEFAULT_CHAT_MODEL = "mistralai/mixtral-8x7b-instruct"
OPENROUTER_DEFAULT_INSTRUCT_MODEL = "mistralai/mixtral-8x7b-instruct"
# Default embedding model
# DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_MODELS = ['text-embedding-3-small', 'text-embedding-3-large', 'huggingface_instruct']

LLM_MODELS = ['nousresearch/nous-capybara-7b:free', 'mistralai/mistral-7b-instruct:free', 'gryphe/mythomist-7b:free','gryphe/mythomist-7b:free','mistralai/mixtral-8x7b-instruct'
    ,'openai/gpt-3.5-turbo','openai/gpt-4-turbo','google/gemini-pro-1.5']


# Constants
DEBUG_TAG = "Debug"
CLEAR_CHAT = "Clear Chat"
SESSION_STATE_VARS = ["conversation", "chat_history", "pdf_docs", "all_chats", "current_chat"]
TAB_LABELS = ["Chat", "Manage Docs", "Settings"]


DEFAULT_PDF_PATH = '/home/james/Dropbox/GIT/CS6795-Cognitive_Science/CS6795-Cognitive_Science_Code/pdf_chat-dev/docs/default.pdf'