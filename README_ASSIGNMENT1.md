# Assignment 1 — Document Q&A Chatbot

This Streamlit app lets you upload documents (PDFs, text files, or markdown files) and chat with them using AI.
The chatbot answers your questions based on the content in your files and shows you where it found the information.

---

## What It Does

* Upload and process multiple files at once (`.pdf`, `.txt`, `.md`)
* Reads PDFs page by page and extracts text (non-scanned only)
* Uses OpenAI’s `text-embedding-3-large` model to create embeddings for each text chunk
* Stores and retrieves embeddings with **Chroma** (in-memory vector store)
* Splits long documents into overlapping chunks for better context understanding (400 characters with 60 overlap)
* Uses LangChain’s `ConversationalRetrievalChain` to enable context-aware chat that remembers prior questions
* Displays which document and page each answer came from
* Custom dark-themed interface with background image and styled chat bubbles
* Warns if uploaded PDFs don’t contain readable text (e.g., scanned images)
* Supports real-time conversation updates through Streamlit’s chat interface

---

## What You’ll Need

* Python 3.11 or newer
* OpenAI API key (do **not** hardcode it in the code)

---

## Getting Started

### 1. Install Dependencies

```bash
python3 -m pip install -r requirements.txt
```

### 2. Set API Key

```bash
export OPENAI_API_KEY="your_api_key_here"
```

> The app uses a Cornell proxy for OpenAI API calls:
>
> ```
> base_url="https://api.ai.it.cornell.edu"
> ```
>
> If you’re using a standard OpenAI endpoint, remove or update this in the code.

### 3. Run the App

```bash
streamlit run chat_with_pdf.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## How to Use

1. Drag and drop your PDF, TXT, or MD files (or click to browse).
2. Wait a few seconds while the app processes your documents.
3. Type your question into the chat box.
4. The assistant will respond with an answer and, if available, show which file and page it came from.

---

## Technical Details

| Feature                 | Implementation                                                |
| ----------------------- | ------------------------------------------------------------- |
| **Text Extraction**     | `pypdf` for PDFs, UTF-8/Latin-1 decoding for text files       |
| **Chunking**            | `RecursiveCharacterTextSplitter` with 400 size and 60 overlap |
| **Embeddings**          | `text-embedding-3-large`                                      |
| **Vector Store**        | Chroma (in-memory)                                            |
| **Retriever**           | Top-10 similarity search                                      |
| **Conversation Memory** | `ConversationBufferMemory`                                    |
| **Chat Model**          | `openai.gpt-4o-mini`                                          |
| **UI Theme**            | Dark mode with custom CSS and background image                |

---

## Troubleshooting

* **Scanned PDFs:** The app can’t read images or scanned PDFs. Run OCR first.
* **Embedding errors:** Check your API key and internet connection.
* **Import errors:** Try reinstalling dependencies:

  ```bash
  python3 -m pip install -U -r requirements.txt
  ```
* **Verify environment:**

  ```bash
  python3 -c "import langchain; print(langchain.__version__)"
  python3 -m pip list | grep -i langchain
  if [ -n "${OPENAI_API_KEY+x}" ]; then echo "OPENAI_API_KEY is set"; else echo "NOT set"; fi
  ```

---

## Example Output

When you ask a question, you’ll see:

* The assistant’s answer
* A collapsible **Sources** section listing filenames and page numbers
* Clean dark UI with styled user and assistant chat bubbles

---

## Notes

* The app doesn’t store your documents or chat history permanently — all processing happens in memory during the session.
* If you re-upload different files, the system automatically reprocesses embeddings.
