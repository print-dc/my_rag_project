import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

# Load environment
load_dotenv("API_KEYS.env")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Init Pinecone + embeddings
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "my-streamlit-index"
index = pc.Index(index_name)
embeddings = CohereEmbeddings(model="embed-english-light-v3.0")

# Streamlit
st.set_page_config(page_title="Index App", layout="wide")
st.title("📂 Upload & Index Documents")

# Namespace input
namespace = st.sidebar.text_input("Namespace to store under", value=f"project_{uuid.uuid4().hex[:6]}")

# Upload file
uploaded_file = st.file_uploader("Upload a text or PDF file", type=["txt", "pdf"])
if uploaded_file and namespace:
    with open("uploaded_file", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File `{uploaded_file.name}` uploaded!")

    # Load
    loader = PyPDFLoader("uploaded_file") if uploaded_file.name.endswith(".pdf") else TextLoader("uploaded_file")
    docs = loader.load()
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

    # Store in Pinecone
    db = PineconeVectorStore(index=index, embedding=embeddings, namespace=namespace)
    db.add_documents(splits)

    st.success(f"Indexed under namespace `{namespace}`. Now open `chat_app.py` and use this namespace to chat.")