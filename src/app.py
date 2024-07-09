import os
from typing import Optional
import streamlit as st

from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings  # , HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

from htmlTemplates import css, bot_template, user_template
import constants
from utils import debug, get_pdf_text, get_available_models, get_file_text, init_session_state, start_new_chat, \
    clear_chat, setup_tabs
from constants import *
import io
# if .env file exists, load it

if os.path.exists('.env'):
    from dotenv import load_dotenv
    load_dotenv('.env')
else:
os.environ["OPENAI_API_KEY"] == st.secrets["OPENAI_API_KEY"]
os.environ["OPENROUTER_API_KEY"] == st.secrets["OPENROUTER_API_KEY"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] == st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    # os.environ["TOGETHER_API_KEY"] == st.secrets["TOGETHER_API_KEY"]
    # os.environ["TOGETHER_BASE_URL"] == st.secrets["TOGETHER_BASE_URL"]



    # And the root-level secrets are also accessible as environment variables:



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

    def _generate(self, messages, stop=None):
        # Get temperature and max_tokens from session state if available
        temp = st.session_state.get("model_config", {}).get("temperature", self.temperature)
        max_tokens = st.session_state.get("model_config", {}).get("max_tokens", self.max_tokens)

        return super()._generate(messages=messages, stop=stop)


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


@st.cache_resource
def select_embedding_model(embedding_model_choice):
    if embedding_model_choice == "huggingface_instruct":
        return HuggingFaceInstructEmbeddings()
    else:
        return OpenAIEmbeddings(openai_api_key=openai_api_key, model=embedding_model_choice)


def get_vectorstore(text_chunks, embedding_model_choice):
    embeddings = select_embedding_model(embedding_model_choice)
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    print(type(vector_store))
    return vector_store


def update_conv_chain():
    # get files from session state
    files = st.session_state.get("pdf_docs")
    # process files
    process_files(files)


# get conversation chain method
def get_conversation_chain(vectorstore):
    temp = st.session_state.get("model_config")["temperature"]
    max_tokens = st.session_state.get("model_config")["max_tokens"]
    model = st.session_state.get("model")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    llm = ChatOpenRouter(
        model_name=model,
        temperature=temp,
        max_tokens=max_tokens,
    )
    # Alternatively, you can use a different language model, like Hugging Face's model
    # llm = HuggingFaceHub(repo_id="decapoda-research/llama-7b-hf", model_kwargs=model_prams)
    print("Creating conversation chain...")
    print("Conversation chain created")
    # Initialize a memory buffer to store conversation history
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    num_docs = st.session_state.get("model_config")["num_sources"]

    if num_docs == 0:
        retriever = vectorstore.as_retriever()
    else:
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": num_docs}
        )

    # set session state retriever to vectorstore


    st.session_state["retriever"] = retriever
    st.session_state["vectorstore"] = vectorstore
    converational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,  # Text vector retriever for context matching
        memory=memory,  # Memory buffer to store conversation history
    )
    # save chain to session state
    st.session_state["conversational_chain"] = converational_chain
    return converational_chain


def get_docs():
    # get the uploaded documents
    return st.session_state["pdf_docs"]


def display_chat(chat_history):
    for message in chat_history:
        # Check if the message is from the user or the bot and display accordingly
        if message['sender'] == 'user':
            st.write(f"You: {message['content']}")
        else:
            st.write(f"Bot: {message['content']}")


################################ Handlers ################################
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


def your_retrieval_function(query):
    # Assuming `retriever` is accessible here, possibly through session state or as a global
    retriever = st.session_state.get("retriever")
    if not retriever:
        st.error("Retriever is not configured.")
        return []
    # Perform the search
    results = retriever.get_relevant_documents(query)

    # results = retriever.search(query)  # Your actual method call might differ
    st.session_state['last_search_results'] = results  # Store results for display
    return results


def handle_userinput(user_question):
    if st.session_state.conversation is not None:
        vector_store = st.session_state.get("vectorstore")
        # st.session_state.conversation = get_conversation_chain(vector_store)

        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        # Call to retrieve and log search results
        r = your_retrieval_function(user_question)

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        st.write("Please upload PDFs and click process")


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


def load_chat(index):
    # Load a previous chat session
    if "all_chats" in st.session_state and index < len(st.session_state.all_chats):
        st.session_state.current_chat = st.session_state.all_chats[index]


def sidebar_header(selected_model='mistralai/mistral-7b-instruct:free'):
    with st.sidebar:
        st.subheader("Model Parameters")
        selected_model, selected_embedding = select_models(selected_model)
        if update_required(selected_model, selected_embedding):
            clear_chat(selected_model, selected_embedding)
            st.session_state.update({"embedding_model": selected_embedding, "model": selected_model})

        st.subheader("Your PDFs")
        handle_pdf_uploads()

        model_options_form()

        handle_user_actions()


def select_models(selected_model):
    selected_model = handle_model_selection(
        constants.LLM_MODELS, selected_model, constants.OPENROUTER_DEFAULT_CHAT_MODEL)
    selected_embedding = handle_embedding_model_selection(
        constants.EMBEDDING_MODELS, selected_model, constants.DEFAULT_EMBEDDING_MODEL)
    return selected_model, selected_embedding


def update_required(selected_model, selected_embedding):
    return selected_model != st.session_state.get("model") or selected_embedding != st.session_state.get(
        "embedding_model")


def process_files(file_list):
    model_config = st.session_state.model_config
    print(model_config)

    for file in file_list:
        file_extension = os.path.splitext(file["name"])[1] if isinstance(file, dict) else os.path.splitext(file.name)[1]

        if file_extension == ".pdf":
            # Handle bytes for default file or read file content for uploaded file
            file_content = file["content"] if isinstance(file, dict) else file.getvalue()
            # Wrap the bytes content in a BytesIO object to make it file-like
            file_stream = io.BytesIO(file_content)
            raw_text = get_pdf_text(file_stream)
            # Continue with your processing logic using raw_text
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return
        # Proceed with the rest of your processing logic here
        print(raw_text)
        # Other processing steps...

        text_chunks = get_text_chunks(raw_text, chunk_size=model_config['chunk_size'],
                                      chunk_overlap=model_config['overtop_tokens'])
        print(f'Number of text chunks: {len(text_chunks)}')
        print("Creating vector store")
        vector_store = get_vectorstore(text_chunks, st.session_state.get("embedding_model"))
        print("Vector store created")
        print("Creating conversation chain")
        st.session_state.conversation = get_conversation_chain(vector_store)
        print("Conversation chain created")


def handle_pdf_uploads():
    uploaded_files = st.file_uploader("Upload PDFs and click process", type="pdf", accept_multiple_files=True)
    st.session_state["pdf_docs"] = uploaded_files
    process_button_clicked = st.button("Process")

    # If no files are uploaded and the "Process" button is clicked, use the default file
    if process_button_clicked:
        if not uploaded_files:  # No files uploaded
            # Path to your default PDF (ensure this path is correct relative to your Streamlit app script)
            default_file_path = DEFAULT_PDF_PATH
            try:
                # Open the default PDF file in binary read mode
                with open(default_file_path, "rb") as default_file:
                    # Simulate the structure of an uploaded file
                    default_file_content = default_file.read()
                    default_file_details = {
                        "name": "default.pdf",
                        "type": "application/pdf",
                        "content": default_file_content
                    }
                    # Assign the default file as if it were uploaded
                    uploaded_files = [default_file_details]
            except IOError:
                st.error("Failed to load the default PDF file. Please check the file path.")
                return  # Exit the function to avoid further processing

        # Proceed with processing either the uploaded files or the default file
        if uploaded_files:
            with st.spinner("Processing PDFs..."):
                process_files(uploaded_files)
        else:
            st.error("Please upload PDFs and click process.")

        # add session state for pdf_docs
        st.session_state["uploaded_files"] = uploaded_files


def settings_changed():
    current_config = st.session_state.get("model_config", {})
    needed_config = {
        "num_sources": st.session_state.get("num_sources_applied", None),
        "temperature": st.session_state.get("temperature_applied", None),
        "max_tokens": st.session_state.get("max_tokens_applied", None)
    }
    return current_config != needed_config


def apply_and_reinitialize():
    if settings_changed():
        # Save the current settings as applied settings
        st.session_state.num_sources_applied = st.session_state.model_config["num_sources"]
        st.session_state.temperature_applied = st.session_state.model_config["temperature"]
        st.session_state.max_tokens_applied = st.session_state.model_config["max_tokens"]

        # Reinitialize components here
        text_chunks = get_text_chunks(get_docs(), st.session_state.model_config["chunk_size"],
                                      st.session_state.model_config["overtop_tokens"])
        vector_store = get_vectorstore(text_chunks, st.session_state["embedding_model"])
        st.session_state.conversation = get_conversation_chain(vector_store)
        st.success("Model configuration updated and components reinitialized.")


def model_options_form():
    with st.form(key='options'):
        st.write('Model Options')
        temp = st.slider('Temperature', 0.0, 1.0, 0.2)
        # add 2 columns
        col1, col2 = st.columns([1, 1])
        # Column 1
        max_tokens = col1.number_input('Max Tokens', 50, 2048, 256)
        freq_penalty = col2.number_input('Frequency Penalty', 0.0)
        top_p = col1.number_input('Top P', 1.0)
        pres_penalty = col2.number_input('Presence Penalty', 0.0)
        # stop_seq = col1.text_input('Stop Sequence', None)

        chunck_size = col2.number_input('Chunk Size', 50, 256, 256)
        num_beams = col1.number_input('Num Beams', 1)
        num_sources = col1.number_input('Num Sources', 0, 20, 5)
        overlap_tokens = col2.number_input('Overtop Tokens', 40)
        # add system prompt
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
                # "stop_sequence": stop_seq,
                "chunk_size": chunck_size,
                "num_beams": num_beams,
                "overtop_tokens": overlap_tokens,
                "system_prompt": system_prompt,
                "num_sources": num_sources
            }
            # st.write(st.session_state.model_config)
            st.success('Options saved!')  # Add success message
            # st.experimental_rerun()
            update_conv_chain()
            # apply_and_reinitialize()  # Call this to check and apply changes immediately

        st.session_state.model_config = {
            "temperature": temp,
            "max_tokens": max_tokens,
            "frequency_penalty": freq_penalty,
            "top_p": top_p,
            "presence_penalty": pres_penalty,
            # "stop_sequence": stop_seq,
            "chunk_size": chunck_size,
            "num_beams": num_beams,
            "overtop_tokens": overlap_tokens,
            "system_prompt": system_prompt,
            "num_sources": num_sources

        }


def handle_user_actions():
    if st.button("New Chat"):
        start_new_chat()
    st.write("Previous Chats")
    if "all_chats" in st.session_state:
        list_previous_chats()


def list_previous_chats():
    for i, chat in enumerate(st.session_state.all_chats):
        if st.button(f"Chat {i + 1}"):
            load_chat(i)


def main():
    # load_dotenv()
    st.set_page_config(page_title="Cognitive LLM",
                       page_icon=":school:")
    st.write(css, unsafe_allow_html=True)

    # init session state
    init_session_state()

    # write model options
    st.write("Model: ", st.session_state.get("model"), "Embedding Model: ", st.session_state.get("embedding_model"))

    ######### Tabs #########
    control_row = st.container()
    with control_row:
        col1, col2, col3 = st.columns([1, 1, 1])
        debug_mode = col1.checkbox(DEBUG_TAG)
        search_results = col2.checkbox("Vector Search Results")

        if col3.button(CLEAR_CHAT):
            st.session_state.conversation = None
            st.session_state.chat_history = None
            st.experimental_rerun()

    # Setup tabs
    tab_chat, tab_docs, tab_settings = setup_tabs()

    # Logic for the "Chat" tab
    with tab_chat:
        st.write("Chat with your documents")
        user_question = st.text_input("Ask a question about your documents:", key="user_query")
        if st.button("Send"):
            handle_userinput(user_question)
    with tab_docs:
        st.write("Manage your documents")
        st.write("Upload and manage your documents here")
        handle_docs()
    with tab_settings:
        st.write("Settings")
        st.write("Change settings here")

    if search_results and 'last_search_results' in st.session_state:
        st.subheader("Search Results")
        r = st.session_state['last_search_results']
        for i, result in enumerate(r):
            st.write(f"Result {i + 1}")
            st.write(result.page_content)
        # st.json(r)  # Display the search results

    if debug_mode:
        debug()

    # init sidebar
    sidebar_header(selected_model=constants.OPENROUTER_DEFAULT_CHAT_MODEL)
    if "current_chat" in st.session_state:
        # Function to display current chat; you might need to write this based on how you display messages
        display_chat(st.session_state.current_chat)


def _get_chat_history(chat_history):
    chat_history_str = ""
    for message in chat_history:
        if message.get('role') == 'user':
            chat_history_str += "User: " + message.get('content') + "\n"
        elif message.get('role') == 'bot':
            chat_history_str += "Assistant: " + message.get('content') + "\n"
    return chat_history_str


if __name__ == '__main__':
    main()
