# 🚀 Gen_Alpha RAG

A two-part modular Retrieval-Augmented Generation (RAG) system built with Streamlit, Pinecone, Langchain, Groq and Cohere embeddings.

✅ Upload & index text/PDF files into Pinecone under a namespace  
✅ Chat with your indexed data in real-time via Groq’s blazing-fast LLaMA-4

---

## ⚡ Features
- 📂 **`index_app.py`** — Upload TXT/PDF, chunk, embed and index into Pinecone under a chosen namespace
- 💬 **`chat_app.py`** — Select namespace & chat with your indexed documents
- 🔍 Auto dropdown of existing namespaces from Pinecone — no manual typing
- 🔥 Hybrid secret loading (local via `.env`, Streamlit Cloud via `st.secrets`)

---

## 🚀 Local quickstart

### 1️⃣ Clone the repo
```bash
git clone https://github.com/print-dc/my_rag_project.git
cd my_rag_project
