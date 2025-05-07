import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# Embedding model (faster + optimized)
embedding_model_name = "nomic-embed-text"

# Caching Ollama LLM
@st.cache_resource
def load_llm():
    return Ollama(model="llama3")  # llama3 for generation

# Caching FAISS DB
@st.cache_resource
def load_db(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model=embedding_model_name)
    return FAISS.from_documents(chunks, embeddings)

# Streamlit UI
st.title("🧪 LLaMA 3-Powered Biosafety Assistant")
st.info("Ask any question related to biosafety lab procedures.")

llm = load_llm()
db = load_db("data/who_biosafety_manual.pdf")  # Make sure file exists!

qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

user_query = st.text_input("What do you want to know?", "")

if user_query:
    response = qa.run(user_query)
    st.success(response)
