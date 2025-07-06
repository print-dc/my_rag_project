import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

                                  # Load environment
load_dotenv("API_KEYS.env")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

                                    # Init Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "my-streamlit-index"
index = pc.Index(index_name)

                                    # Embeddings & LLM
embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.1)
parser = StrOutputParser()

                                       # Streamlit UI
st.set_page_config(page_title="Gen_Alpha", layout="wide")
st.title("🤖 Gen_Alpha")

                                        # Session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "namespace" not in st.session_state:
    st.session_state.namespace = None
if "full_text" not in st.session_state:
    st.session_state.full_text = ""
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

                                      # Sidebar upload
uploaded_file = st.sidebar.file_uploader("Upload a text or PDF file", type=["txt", "pdf"])
if uploaded_file:
    with open("uploaded_file", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"File `{uploaded_file.name}` uploaded!")

    st.session_state.namespace = str(uuid.uuid4())
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader("uploaded_file")
    else:
        loader = TextLoader("uploaded_file")
    docs = loader.load()
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    st.session_state.full_text = "\n\n".join([doc.page_content for doc in splits])

    db = PineconeVectorStore(index=index, embedding=embeddings, namespace=st.session_state.namespace)
    db.add_documents(splits)

    retriever = db.as_retriever()
    retriever.search_kwargs["namespace"] = st.session_state.namespace
    retriever.search_kwargs["k"] = 5
    st.sidebar.info("✅ File loaded. Ask me anything!")
else:
    retriever = None

                                     # Display chat
for idx, (q, a) in enumerate(st.session_state.chat_history):
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)
        if idx < len(st.session_state.last_sources):
            sources = st.session_state.last_sources[idx]
            if sources:
                st.markdown("**🔍 Sources:**")
                for src in sources:
                    st.markdown(f"> {src}")

                                         # Input
user_input = st.chat_input("Ask me anything...")
if user_input:
    if uploaded_file:
        if any(kw in user_input.lower() for kw in ["summary", "summarise", "context", "overview"]):
            with st.spinner("Summarizing document..."):
                prompt = f"Give a crisp summary of this document:\n\n{st.session_state.full_text[:4000]}"
                response = model.invoke(prompt)
                answer = parser.invoke(response)
            st.session_state.last_sources.append([])
        else:
            with st.spinner("Thinking..."):
                retrieved_docs = retriever.get_relevant_documents(user_input)
                sources = [doc.page_content[:200]+"..." for doc in retrieved_docs]
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])

                prompt = f"""Use the following context to answer the question crisply and directly.

Context:
{context}

Question: {user_input}

Answer:"""
                response = model.invoke(prompt)
                answer = parser.invoke(response)
                st.session_state.last_sources.append(sources)

        st.session_state.chat_history.append((user_input, answer))
    else:
        st.session_state.chat_history.append((user_input, "⚠️ Please upload a file first."))
        st.session_state.last_sources.append([])

    st.rerun()