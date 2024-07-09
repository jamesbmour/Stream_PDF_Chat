# import json

import os
from typing import Optional

# import requests
import streamlit as st
# from dotenv import load_dotenv
from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings  # , HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

from htmlTemplates import css, bot_template, user_template
# from langchain.llms import HuggingFaceHub, HuggingFacePipeline
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import constants
from utils import debug, get_pdf_text, get_available_models,  get_file_text
# from tiktoken import Tokenizer, models

# specify the tokenizer that you want to use
# tokenizer = Tokenizer(models.BertWordPiece())
# load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
class ChatOpenRouter(ChatOpenAI):
    openai_api_base: str
    openai_api_key: str
    model_name: str
    temperature: float = 0.0  # Default temperature
    max_tokens: int = 50  # Default max_tokens

    def __init__(self,
                 model_name: str,
                 openai_api_key: Optional[str] = None,
                 openai_api_base: str = "https://openrouter.ai/api/v1",
                 temperature: float = 0.3,
                 max_tokens: int = 50,
                 **kwargs):
        # openai_api_key = openai_api_key or os.getenv('OPENROUTER_API_KEY')
        print("Using Model: ", model_name)
        print("Max Tokens: ", max_tokens)
        print("Using API Base: ", openai_api_base)

        openai_api_key = openrouter_api_key
        print(openai_api_key)
        super().__init__(openai_api_base=openai_api_base,
                         openai_api_key=openai_api_key,
                         model_name=model_name, **kwargs)
        self.temperature = temperature
        self.max_tokens = max_tokens

    # def _generate(self, messages, stop=None):
    #     # Get temperature and max_tokens from session state if available
    #     temp = st.session_state.get("model_config", {}).get("temperature", self.temperature)
    #     max_tokens = st.session_state.get("model_config", {}).get("max_tokens", self.max_tokens)

    #     return super()._generate(messages=messages, stop=stop, temperature=temp, max_tokens=max_tokens)
    def _generate(self, messages, stop=None):
        # Get temperature and max_tokens from session state if available
        temp = st.session_state.get("model_config", {}).get("temperature", self.temperature)
        max_tokens = st.session_state.get("model_config", {}).get("max_tokens", self.max_tokens)

        return super()._generate(messages=messages, stop=stop)

# Handle the model selection process
def handle_model_selection(available_models, selected_model, default_model):
    # Determine the index of the selected model
    if selected_model and selected_model in available_models:
        selected_index = available_models.index(selected_model)
    else:
        selected_index = available_models.index(default_model)
    selected_model = st.selectbox(
        "Select LLM model", available_models, index=selected_index
    )
    return selected_model


# handle the embedding model selection process
def handle_embedding_model_selection(available_models, selected_model, default_model):
    # Determine the index of the selected model
    if selected_model and selected_model in available_models:
        selected_index = available_models.index(selected_model)
    else:
        selected_index = available_models.index(default_model)
    selected_model = st.selectbox(
        "Select embedding model", available_models, index=selected_index
    )
    return selected_model


# TODO: Convert the input text to take from form input
def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_chunks = []
    position = 0
    # Iterate over the text until the entire text has been processed
    while position < len(text):
        start_index = max(0, position - chunk_overlap)
        end_index = position + chunk_size
        chunk = text[start_index:end_index]
        text_chunks.append(chunk)
        position = end_index - chunk_overlap
    return text_chunks


def get_embedding_model(model_name="hkunlp/instructor-xl"):
    if model_name == "openai":
        return OpenAIEmbeddings(openai_api_key=openai_api_key)
    else:
        return HuggingFaceInstructEmbeddings()


# get vector store method
# def get_vectorstore(text_chunks):
#     # TODO: make vector store a persistent object that can be reused
#     embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#     # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
#     vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#
#     # vector_store = Qdrant.from_documents(text_chunks, embeddings, "http://localhost:6333")
#     print(type(vector_store))
#
#     return vector_store
def get_vectorstore(text_chunks):
    # Obtain the saved model from the side form choice
    embedding_model_choice = st.session_state.get("embedding_model")

    # Response with the selection, toggling on a relationship to your feedback pool
    # if embedding_model_choice == "openai":
    # Base this argument reference to be as your activity's `apikey` or particular method's name
    # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model=embedding_model_choice)
    # else:# embedding_model_choice == "huggingface_instruct":
    if embedding_model_choice == "huggingface_instruct":
        # Fit this alternative with the about model tag or yours reflective
        embeddings = HuggingFaceInstructEmbeddings()
    else:
        # Any good fall-back or a guesstimate you wish to pull from a product run or class of map
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model=embedding_model_choice)
    #     # Fit this alternative with the about model tag or yours reflective
    # else:
    #     # Any good fall-back or a guesstimate you wish to pull from a product run or class of map
    #     embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)  # Your app's standard

    # Weigh the laden of your lede, as with the carapace it offers
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    print(type(vector_store))

    return vector_store


# get conversation chain method
def get_conversation_chain(vectorstore):
    model_prams = {"temperature": 0.23, "max_length": 4096}
    # TODO: Convert this to OpenRouter
    temp = st.session_state.get("model_config")["temperature"]
    max_tokens = st.session_state.get("model_config")["max_tokens"]
    model = st.session_state.get("model")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # llm = ChatOpenAI(
    #     temperature=temp,
    #     model_name=model,
    #     openai_api_key=openai_api_key,
    #     openai_api_base=constants.OPENROUTER_API_BASE,
    #
    #                  max_tokens=max_tokens)
    llm = ChatOpenRouter(
        model_name=model,
        temperature=temp,
        max_tokens=max_tokens,
    )

    # update model parameters

    # Alternatively, you can use a different language model, like Hugging Face's model
    # llm = HuggingFaceHub(repo_id="decapoda-research/llama-7b-hf", model_kwargs=model_prams)
    print("Creating conversation chain...")
    print("Conversation chain created")
    # Initialize a memory buffer to store conversation history
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),  # Text vector retriever for context matching
        memory=memory,  # Memory buffer to store conversation history
    )


# get handler user input method
def handle_userinput(user_question):
    if st.session_state.conversation is not None:

        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        st.write("Please upload PDFs and click process")


def get_docs():
    # get the uploaded documents
    return st.session_state["pdf_docs"]


def handle_docs():
    # TODO: Add a way to manage the documents
    # display the documents
    st.write("Documents will appear here")
    st.write(get_docs())
    # drop down to select a document
    st.selectbox("Select a document", get_docs())

    # button to delete a document
    if st.button("Delete"):
        # TODO: Add a way to delete the document
        st.write("Document deleted")


# create sidebar header
def sidebar_header(selected_model='mistralai/mistral-7b-instruct:free'):
    with st.sidebar:
        st.subheader("Model Parameters")

        selected_model = handle_model_selection(
            constants.LLM_MODELS, selected_model, constants.OPENROUTER_DEFAULT_CHAT_MODEL
        )
        # load the embedding models
        available_embedding_models = constants.EMBEDDING_MODELS
        selected_embedding = handle_embedding_model_selection(
            available_embedding_models, selected_model, constants.DEFAULT_EMBEDDING_MODEL
        )
        # selected_model = st.selectbox("Select LLM model", LLM_MODELS,
        #                                 on_change=_reset_chat_state)
        # selected_embedding = st.selectbox("Select embedding model", available_embedding_models,
        #                                     on_change=_reset_chat_state)
        if selected_model != st.session_state.get("model") or selected_embedding != st.session_state.get("embedding_model"):
            _reset_chat_state(selected_model, selected_embedding)

        st.session_state["embedding_model"] = selected_embedding

        st.session_state["model"] = selected_model
        # var = st.query_params
        # TODO: add new chat button that clears chat history and resets conversation chain
        # TODO: add save chat button to sidebar saved chat history
        st.subheader("Your PDFs")
        pdf_docs = st.file_uploader("Upload PDFs and click process", type="pdf", accept_multiple_files=True)
        st.session_state["pdf_docs"] = pdf_docs

        if st.button("Process"):
            with st.spinner("Processing PDFs"):
                process_files(pdf_docs, st)

        with st.form(key='options'):
            # form title
            st.write('Model Options')
            temp = st.slider('Temperature', 0.0, 1.0, 0.6)
            # add 2 columns
            col1, col2 = st.columns([1, 1])
            # Column 1
            max_tokens = col1.number_input('Max Tokens', 50)
            freq_penalty = col2.number_input('Frequency Penalty', 0.0)
            top_p = col1.number_input('Top P', 1.0)
            pres_penalty = col2.number_input('Presence Penalty', 0.0)
            stop_seq = col1.text_input('Stop Sequence', None)

            chunck_size = col2.number_input('Chunck Size', 256)
            num_beams = col1.number_input('Num Beams', 1)
            overlap_tokens = col2.number_input('Overtop Tokens', 40)
            # add system prompt
            st.write("System Prompt")
            system_prompt = st.text_area("System Prompt", "Summarize the document in 3 sentences.")

            # submit button centered
            submitted = st.form_submit_button('Save Options', help="Save the options for the model")
            if submitted:
                # st.write("Model Options")
                # print form values
                st.session_state.model_config = {
                    "temperature": temp,
                    "max_tokens": max_tokens,
                    "frequency_penalty": freq_penalty,
                    "top_p": top_p,
                    "presence_penalty": pres_penalty,
                    "stop_sequence": stop_seq,
                    "chunk_size": chunck_size,
                    "num_beams": num_beams,
                    "overtop_tokens": overlap_tokens,
                    "system_prompt": system_prompt
                }
            # st.write(st.session_state.model_config)
                st.success('Options saved!')  # Add success message
                st.experimental_rerun()
            st.session_state.model_config = {
                "temperature": temp,
                "max_tokens": max_tokens,
                "frequency_penalty": freq_penalty,
                "top_p": top_p,
                "presence_penalty": pres_penalty,
                "stop_sequence": stop_seq,
                "chunk_size": chunck_size,
                "num_beams": num_beams,
                "overtop_tokens": overlap_tokens,
                "system_prompt": system_prompt
            }
      # Rerun the app with the new options
        if 'model_config' in st.session_state:
            st.write("Current Model Config:", st.session_state.model_config)

        st.subheader("Chat History")
        st.write("Chat history will appear here")


def main():
    load_dotenv()
    st.set_page_config(page_title="Cognitive LLM",
                       page_icon=":school:")
    st.write(css, unsafe_allow_html=True)

    # init session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "pdf_docs" not in st.session_state:
        st.session_state.pdf_docs = []  # Initialize as an empty list (or whatever structure you need)

    st.header("Chat with multiple PDFs")
    # write model options
    st.write("Model: ", st.session_state.get("model"), "Embedding Model: ", st.session_state.get("embedding_model"))

    # Control Row (Above Chat History)


    ######### Tabs #########
    # create 3 tabs
    control_row = st.container()
    with control_row:
        col1, col2, col3 = st.columns([1, 1, 1])
        d = col1.checkbox("Debug")
        if col3.button("Clear Chat"):
            st.session_state.conversation = None
            st.session_state.chat_history = None
            st.experimental_rerun()
    tab_chat, tab_docs, tab_settings = st.tabs(["Chat", "Manage Docs", "Settings"])
    # Logic for the "Chat" tab

    with tab_chat:
        st.write("Chat with your documents")
        if user_question := st.text_input("Ask a question about your documents:"):

            handle_userinput(user_question)



    with tab_docs:
        st.write("Manage your documents")
        st.write("Upload and manage your documents here")
        handle_docs()

    with tab_settings:
        st.write("Settings")
        st.write("Change settings here")

    if d:
        debug()

    # init sidebar
    sidebar_header()

    # TODO: add select model dropdown in sidebar to select model to use


# TODO Rename this here and in `main`
def process_files(file_list, st):  # sourcery skip: raise-specific-error
    # get model config for session state
    model_config = st.session_state.model_config
    print(model_config)
    for file in file_list:
        file_extension = os.path.splitext(file.name)[1]
        file_name = os.path.splitext(file.name)[0]
        if file_extension == ".pdf":
            raw_text = get_pdf_text(file)
        elif file_extension == ".txt":
            with open(file, 'r') as txt_file:
                raw_text = txt_file.read()

        elif file_extension == ".csv":
            with open(file, 'r') as csv_file:
                raw_text = csv_file.read()

        else:
            raise Exception("File type not supported")
    print(raw_text)
    text_chunks = get_text_chunks(raw_text, chunk_size=model_config['chunk_size'],
                                  chunk_overlap=model_config['overtop_tokens'])
    print(f'Number of text chunks: {len(text_chunks)}')
    print("Creating vector store")
    vector_store = get_vectorstore(text_chunks)
    print("Vector store created")
    print("Creating conversation chain")
    st.session_state.conversation = get_conversation_chain(vector_store)
    print("Conversation chain created")

def _reset_chat_state(selected_model, selected_embedding):
    """Resets chat state when the model selection is changed."""
    print("Resetting chat state...")
    print("Selected Model: ", selected_model)
    print("Selected Embedding: ", selected_embedding)
    st.session_state.conversation = None
    st.session_state.chat_history = None
    st.session_state['model'] = selected_model
    st.session_state['embedding_model'] = selected_embedding

    st.experimental_rerun()

if __name__ == '__main__':
    main()
