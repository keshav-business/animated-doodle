import os
import uuid
import time
import logging
import threading
from datetime import datetime
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import base64
import requests
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import speech_recognition as sr
from dotenv import load_dotenv
from openai import OpenAI

from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1000))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 200))
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-3.5-turbo')

# Default system message
DEFAULT_SYSTEM_MESSAGE = """strictly answer in english and dont go out of the context that is provided to u but please cosider that there will be major speeech recognition errors so please work accordingly, You are a helpful AI assistant with access to UBIK Solutions. 
Answer questions based on the provided context. If you don't know something or if it's not in the context, dont go out of context, stick to what user said and form answer on that with help of context
say so directly instead of making up information. Your name is ubik ai, answer mostly under 50 words unless very much required, send back data in a clean format"""

def initialize_global_vectorstore():
    global global_vectorstore
    with vectorstore_lock:
        if global_vectorstore is not None:
            return True, "[SYSTEM MESSAGE] Vectorstore is already initialized."

        pdf_paths = [
            os.path.join('data', "Ilesh Sir (IK) - Words.pdf"),
            os.path.join('data', "UBIK SOLUTION.pdf"),
            os.path.join('data', "illesh3.pdf"),
            os.path.join('data', "website-data-ik.pdf")
        ]

        combined_text = ""
        for path in pdf_paths:
            combined_text += get_pdf_text(path) + " "

        if not combined_text.strip():
            return False, "No text could be extracted from the PDFs."

        text_chunks = get_text_chunks(combined_text)
        
        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        global_vectorstore = FAISS.from_texts(
            texts=text_chunks,
            embedding=embeddings
        )
        logger.info("Vectorstore has been created with OpenAI embeddings.")
    return True, "[SYSTEM MESSAGE] Vectorstore was created successfully."
