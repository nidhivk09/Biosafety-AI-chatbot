import os
import base64
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from streamlit_mic_recorder import mic_recorder

# Load API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("Please set the GEMINI_API_KEY environment variable.")
    st.stop()

# Load Gemini LLM
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.2,
        google_api_key=GEMINI_API_KEY
    )

# Translate to English
def translate_to_english(text, llm):
    prompt = f"Translate this to English:\n\n{text}"
    return llm.invoke(prompt).content.strip()

# Transcribe base64-encoded audio using Gemini
def transcribe_audio(audio_bytes, llm):
    b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
    prompt = f"""
You are a transcription expert.

The following audio is base64-encoded WAV data. Please transcribe the spoken content as accurately as possible.

Audio (base64):
{b64_audio}

Transcription:
"""
    try:
        result = llm.invoke(prompt)
        return result.content.strip()
    except Exception as e:
        return f"[Error transcribing audio: {e}]"

# Load DB
@st.cache_resource
def load_db(pdf_paths):
    all_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_docs.extend(docs)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    return FAISS.from_documents(chunks, embeddings)


# Set up Streamlit
st.title("🎤 Gemini Biosafety Assistant (with Audio)")
st.info("Ask questions in any language via voice or text.")

llm = load_llm()
pdf_files = [
    "data/who_biosafety_manual.pdf",
    "data/File608.pdf",
    
]

db = load_db(pdf_files)


prompt_template = PromptTemplate.from_template("""
You are an expert assistant on biosafety lab procedures.

Use the following extracted context from a biosafety manual to answer the user's question. 
If the answer is not present in the context, respond based on your own knowledge.

Context:
{context}

Question:
{question}

Helpful Answer:
""")

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    chain_type_kwargs={"prompt": prompt_template}
)

# Input
typed_query = st.text_input("Type your question (any language):")

audio = mic_recorder(start_prompt="🎤 Record", stop_prompt="🔴 Stop", key="voice_input")

final_query = ""

# Handle audio
if audio and audio["audio"]:
    st.audio(audio["audio"], format="audio/wav")
    with st.spinner("Transcribing audio using Gemini..."):
        audio_bytes = audio["audio"]
        transcription = transcribe_audio(audio_bytes, llm)
        st.write("🗣️ Transcribed:", transcription)
        final_query = transcription

# Handle typed input
if typed_query.strip():
    final_query = typed_query

# Translate and run QA
if final_query:
    with st.spinner("Translating to English..."):
        translated = translate_to_english(final_query, llm)
        st.write("📝 Translated:", translated)

    with st.spinner("Generating answer..."):
        answer = qa.run(translated)
        st.success(answer)
