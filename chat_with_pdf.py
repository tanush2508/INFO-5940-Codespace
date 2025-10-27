import os
import io
import base64
import streamlit as st
from pypdf import PdfReader
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from openai import OpenAI

openai_client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.ai.it.cornell.edu",
)

def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

background_image_path = "background1.jpg"
if os.path.exists(background_image_path):
    base64_image = get_base64_of_bin_file(background_image_path)
else:
    base64_image = ""

page_bg_img = f"""
<style>
.stApp {{
    background-image: url("data:image/jpg;base64,{base64_image}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;
    color: #E0E0E0 !important;
}}
.stChatMessage {{
    background-color: rgba(25, 25, 25, 0.85) !important;
    border-radius: 12px;
    padding: 1rem;
    color: #E0E0E0 !important;
}}
[data-testid="stChatMessageUser"] {{
    background-color: rgba(50, 50, 50, 0.85) !important;
    border-left: 4px solid #6C63FF;
}}
[data-testid="stChatMessageAssistant"] {{
    background-color: rgba(35, 35, 35, 0.85) !important;
    border-left: 4px solid #03DAC6;
}}
.streamlit-expanderHeader, .stFileUploader, .stTextInput, .stButton button {{
    background-color: rgba(40, 40, 40, 0.9) !important;
    color: #E0E0E0 !important;
    border: 1px solid #555;
    border-radius: 8px;
}}
::-webkit-scrollbar {{
    width: 8px;
}}
::-webkit-scrollbar-thumb {{
    background: #444; 
    border-radius: 10px;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("Ask Questions/Chat with the files added.")

uploaded_documents = st.file_uploader(
    "Upload your docs to start",
    type=("txt", "pdf", "md"),
    accept_multiple_files=True
)

input_placeholder = "Ask me anything related to the docs uploaded." if uploaded_documents else "Upload files"

if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = [
        {"role": "assistant", "content": "Upload files. Processing takes a short while"}
    ]

for message in st.session_state.conversation_history:
    st.chat_message(message["role"]).write(message["content"])

def extract_text_from_uploaded_file(uploaded_file) -> str:
    file_name = uploaded_file.name.lower()
    file_data = uploaded_file.getvalue()
    if file_name.endswith(".pdf"):
        pdf_reader = PdfReader(io.BytesIO(file_data))
        extracted_pages = []
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                extracted_pages.append(page_text)
        return "\n".join(extracted_pages).strip()
    try:
        return file_data.decode("utf-8")
    except UnicodeDecodeError:
        return file_data.decode("latin-1", errors="ignore")

def create_document_objects(uploaded_files) -> list[Document]:
    document_list = []
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        if file_name.lower().endswith(".pdf"):
            file_bytes = uploaded_file.getvalue()
            pdf_reader = PdfReader(io.BytesIO(file_bytes))
            for page_number, page in enumerate(pdf_reader.pages, start=1):
                page_text = (page.extract_text() or "").strip()
                if page_text:
                    document_list.append(Document(
                        page_content=page_text, 
                        metadata={"source": file_name, "page": page_number}
                    ))
        else:
            file_text = extract_text_from_uploaded_file(uploaded_file).strip()
            if file_text:
                document_list.append(Document(
                    page_content=file_text, 
                    metadata={"source": file_name}
                ))
    return document_list

def split_documents_into_chunks(documents: list[Document],
                                chunk_size: int = 400,
                                chunk_overlap: int = 60) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)

def generate_files_fingerprint(uploaded_files):
    return tuple(sorted((file.name, getattr(file, "size", None)) for file in uploaded_files))

if uploaded_documents:
    with st.expander("Loaded files", expanded=False):
        for file in uploaded_documents:
            st.write("•", file.name)
    current_fingerprint = generate_files_fingerprint(uploaded_documents)
    if st.session_state.get("current_files_fingerprint") != current_fingerprint:
        document_objects = create_document_objects(uploaded_documents)
        document_chunks = split_documents_into_chunks(document_objects)
        st.session_state["current_files_fingerprint"] = current_fingerprint
        st.session_state["document_objects"] = document_objects
        st.session_state["document_chunks"] = document_chunks
        st.session_state.pop("vector_store", None)
        st.session_state.pop("qa_chain", None)

def create_vector_store(document_chunks) -> Chroma:
    embedding_model = OpenAIEmbeddings(model="openai.text-embedding-3-large")
    return Chroma.from_documents(document_chunks, embedding=embedding_model)

def create_qa_chain(vector_store) -> ConversationalRetrievalChain:
    language_model = ChatOpenAI(model="openai.gpt-4o-mini", temperature=0.3)
    conversation_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer",
    )
    return ConversationalRetrievalChain.from_llm(
        llm=language_model,
        retriever=vector_store.as_retriever(search_kwargs={"k": 10}),
        memory=conversation_memory,
        return_source_documents=True,
    )

if st.session_state.get("document_chunks") and "vector_store" not in st.session_state:
    if not st.session_state["document_chunks"]:
        st.error("No readable text in the uploaded files. Please make sure PDFs aren't scanned images.")
    else:
        try:
            with st.status("Loading", expanded=True) as loading_status:
                loading_status.write("Analyzing your docs...")
                vector_store = create_vector_store(st.session_state["document_chunks"])
                loading_status.write("Connecting you to your documents...")
                st.session_state["vector_store"] = vector_store
                st.session_state["qa_chain"] = create_qa_chain(vector_store)
                loading_status.update(
                    label="Ready! You can now ask questions.",
                    state="complete",
                    expanded=False
                )
        except Exception as error:
            st.error(f"Something went wrong while preparing the docs: {error}")

user_input = st.chat_input(
    input_placeholder,
    disabled=not st.session_state.get("qa_chain"),
)

if user_input:
    st.session_state.conversation_history.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    if not st.session_state.get("qa_chain"):
        st.error("Please upload files first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Loading....."):
                query_result = st.session_state["qa_chain"].invoke({"question": user_input})
                response_text = (query_result.get("answer", "") or "").strip()
                source_documents = query_result.get("source_documents", [])
                st.markdown(response_text)
                if source_documents:
                    with st.expander("Sources"):
                        for index, document in enumerate(source_documents, 1):
                            source_name = document.metadata.get("source", "(unknown)")
                            page_number = document.metadata.get("page")
                            st.write(f"[{index}] {source_name}" + (f", page {page_number}" if page_number else ""))
                question_preview = (user_input or "")[:120].replace("\n", " ")
                source_count = len(source_documents) if source_documents else 0
                print(f"RAG_USED question='{question_preview}' sources={source_count} answer_len={len(response_text)}")
    st.session_state.conversation_history.append({"role": "assistant", "content": response_text})
