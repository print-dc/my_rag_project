import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

load_dotenv("API_KEYS.env")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.1)
embeddings = CohereEmbeddings(model="embed-english-light-v3.0")

index_name = input("Enter your Pinecone index name: ").strip()
if index_name not in pc.list_indexes():
    pc.create_index(index_name, dimension=384, metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"))

index = pc.Index(index_name)
db = PineconeVectorStore(index=index, embedding=embeddings)

loader = TextLoader("youtube_transcript.txt")
docs = loader.load()
splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
db.add_documents(splits)

retriever = db.as_retriever()
chain = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=retriever,
)

print("\nChatbot is ready! Type 'exit' to quit.\n")

chat_history = []

while True:
    question = input("You: ").strip()
    if question.lower() in ["exit", "quit"]:
        print("Bye.")
        break

    response = chain.invoke({
        "question": question,
        "chat_history": chat_history
    })

    answer = response["answer"]
    print(f"AI: {answer}")

    chat_history.append((question,answer))