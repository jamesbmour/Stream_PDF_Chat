from urllib.parse import urlparse

from PyPDF2 import PdfReader
from streamlit_javascript import st_javascript
import streamlit as st
import requests
import json
from constants import SESSION_STATE_VARS, OPENROUTER_API_BASE, TAB_LABELS

import os


def setup_tabs():
    tab_chat, tab_docs, tab_settings = st.tabs(TAB_LABELS)
    return tab_chat, tab_docs, tab_settings


def clear_chat(selected_model, selected_embedding):
    """Resets chat state when the model selection is changed."""
    print("Resetting chat state...")
    print("Selected Model: ", selected_model)
    print("Selected Embedding: ", selected_embedding)
    st.session_state.conversation = None
    st.session_state.chat_history = None
    st.session_state['model'] = selected_model
    st.session_state['embedding_model'] = selected_embedding

    st.experimental_rerun()

def start_new_chat():
    # Archive the current chat if it exists
    if "current_chat" in st.session_state and st.session_state.current_chat:
        if "all_chats" not in st.session_state:
            st.session_state.all_chats = []
        st.session_state.all_chats.append(st.session_state.current_chat)
    st.session_state.current_chat = []

def init_session_state():
    """
    Initialize session state if not already present
    """
    for var in SESSION_STATE_VARS:
        if var not in st.session_state:
            # Initialize as an empty list if the variable is pdf_docs, all_chats or current_chat.
            if var in ["pdf_docs", "all_chats", "current_chat","standardized_chat_history"]:
                st.session_state[var] = []
            else:
                st.session_state[var] = None



def get_url():
    return st_javascript("await fetch('').then(r => window.parent.location.href)")


def open_page(url):
    st_javascript(f"window.open('{url}', '_blank').focus()")


def url_to_hostname(url):
    uri = urlparse(url)
    return f"{uri.scheme}://{uri.netloc}/"


def debug():
    st.header("Debug Info")
    st.write("Session State")
    st.write(st.session_state)
def get_available_models():
    try:
        response = requests.get(OPENROUTER_API_BASE + "/models")
        response.raise_for_status()
        models = json.loads(response.text)["data"]
        return [model["id"] for model in models]
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting models from API: {e}")
        return []

def get_pdf_text(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    return "".join(page.extract_text() for page in pdf_reader.pages)

def get_file_text(file_path_list):
    raw_text = ""
    for file_path in file_path_list:
        file_extension = os.path.splitext(file_path)[1]
        file_name = os.path.splitext(file_path)[0]
        if file_extension == ".pdf":
            raw_text += get_pdf_text(file_path)
        elif file_extension == ".txt":
            with open(file_path, 'r') as txt_file:
                raw_text += txt_file.read()

        elif file_extension == ".csv":
            with open(file_path, 'r') as csv_file:
                raw_text += csv_file.read()

        else:
            raise Exception("File type not supported")

    return raw_text

