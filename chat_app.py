import os
import streamlit as st
from dotenv import load_dotenv
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

# Init
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "my-streamlit-index"
index = pc.Index(index_name)
embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.1)
parser = StrOutputParser()

# Streamlit UI
st.set_page_config(page_title="Chat App", layout="wide")
st.title("Chat with Your Documents")

# Session history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Namespace input
namespace = st.sidebar.text_input("Namespace to chat with")

# If namespace entered, set up retriever
if namespace:
    retriever = PineconeVectorStore(index=index, embedding=embeddings, namespace=namespace).as_retriever()
    retriever.search_kwargs["k"] = 5

    # Chat loop
    user_input = st.chat_input("Ask your question...")
    if user_input:
        with st.spinner("Thinking..."):
            docs = retriever.get_relevant_documents(user_input)
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = f"""Use this context to answer crisply.

Context:
{context}

Question: {user_input}

Answer:"""
            response = model.invoke(prompt)
            answer = parser.invoke(response)

        st.session_state.chat_history.append((user_input, answer))

# Show chat history
for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)