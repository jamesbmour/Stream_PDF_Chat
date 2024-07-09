Here's a sample README.md file for the provided Streamlit app:

```markdown
# Cognitive LLM Streamlit App

This Streamlit app allows you to interact with a Language Model (LLM) to ask questions about your uploaded PDF documents. It uses various libraries such as Langchain, FAISS, and Hugging Face to process the documents, create embeddings, and generate responses to user queries.

## Features

- Upload multiple PDF documents
- Select an LLM model and an embedding model
- Adjust model parameters such as temperature, max tokens, and chunk size
- Chat with the LLM about the content of the uploaded documents
- View and manage uploaded documents
- View search results from the vector store
- Save and load previous chat sessions

## Installation

1. Clone the repository:
   ``` shell
   git clone https://github.com/your-username/cognitive-llm-app.git
   ```

2. Install the required dependencies:
   ``` shell
   pip install -r requirements.txt
   ```

3. Set up the necessary environment variables:
   - Create a `.env` file in the project root directory
   - Add the following variables to the `.env` file:
     ``` shell
     OPENAI_API_KEY=your_openai_api_key
     OPENROUTER_API_KEY=your_openrouter_api_key
     ```

## Usage

1. Run the Streamlit app:
   ``` shell
   
   streamlit run app.py
   
   ```

2. Open the app in your web browser using the provided URL.

3. Upload PDF documents using the file uploader in the sidebar.

4. Click the "Process" button to process the uploaded documents.

5. Adjust the model parameters in the sidebar if needed.

6. Go to the "Chat" tab and ask a question about the uploaded documents.

7. View the generated response from the LLM.

8. Optionally, you can view the search results from the vector store by enabling the "Vector Search Results" checkbox.

9. Manage your uploaded documents in the "Docs" tab.

10. Change settings and model options in the "Settings" tab.

## Configuration

The app uses the following configuration options:

- `OPENAI_API_KEY`: Your OpenAI API key for using OpenAI models.
- `OPENROUTER_API_KEY`: Your OpenRouter API key for using OpenRouter models.
- `DEFAULT_PDF_PATH`: The path to a default PDF file to be used if no files are uploaded.
- `DEBUG_TAG`: A tag used for enabling debug mode.
- `CLEAR_CHAT`: A tag used for clearing the chat history.

Make sure to set these options according to your requirements.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
```

Feel free to customize the README.md file based on your specific app requirements, additional features, or deployment instructions.