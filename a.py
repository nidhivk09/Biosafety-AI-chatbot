import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
from google.cloud import speech
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Set your Gemini API Key directly (Option 2)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("Please set the GEMINI_API_KEY environment variable.")
    st.stop()

# Set up Google Cloud Speech client (ensure credentials are set)
client = speech.SpeechClient()

# LangChain Gemini LLM using the API key
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        google_api_key=GEMINI_API_KEY  # manually passed to avoid ADC error
    )

# Load the PDF and embed it
@st.cache_resource
def load_db(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    return FAISS.from_documents(chunks, embeddings)

# Function to process audio input and convert it to text using Google Cloud Speech-to-Text
def transcribe_audio(audio_data):
    # Initialize audio data
    audio = speech.RecognitionAudio(content=audio_data)

    # Configure recognition settings
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    # Send the request to Google Cloud Speech API
    response = client.recognize(config=config, audio=audio)

    # Retrieve and return the transcribed text
    if response.results:
        return response.results[0].alternatives[0].transcript
    else:
        return "Sorry, I couldn't understand the audio."

# Custom Audio Processor to Capture Audio
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_data = None

    def recv(self, frame):
        self.audio_data = frame.to_bytes()  # Convert the frame to byte data
        return frame

# Streamlit UI
st.title("🧪 Gemini-Powered Biosafety Assistant (LangChain Wrapper)")
st.info("Ask anything about biosafety lab procedures using audio input.")

# Set up the webrtc streamer to capture audio
audio_processor = AudioProcessor()
webrtc_streamer(key="audio", audio_processor_factory=lambda: audio_processor)

# Process the audio input if recorded
if audio_processor.audio_data:
    # Convert the audio to text using Google Cloud Speech-to-Text
    text_input = transcribe_audio(audio_processor.audio_data)
    st.write("Transcribed Text: ", text_input)

    # Now, use this transcribed text as the input for your biosafety assistant
    if text_input:
        llm = load_llm()
        db = load_db("data/who_biosafety_manual.pdf")  # Ensure this file exists

        # Custom prompt
        custom_prompt = PromptTemplate.from_template("""
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
            chain_type_kwargs={"prompt": custom_prompt}
        )

        response = qa.run(text_input)  # Use the transcribed text as the question
        st.success(response)

# Text input option (fallback if no audio recorded)
else:
    user_query = st.text_input("Or type your question here:", "")
    if user_query:
        llm = load_llm()
        db = load_db("data/who_biosafety_manual.pdf")  # Ensure this file exists

        response = qa.run(user_query)
        st.success(response)
