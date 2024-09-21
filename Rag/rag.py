# import libraries
import os, sys
import streamlit as st
import langchain
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import faiss
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

import logging
logging.basicConfig(level=logging.INFO)


custom_template = """
Given a following conversation and a follow up question rephrase the follow up question to be a standalone question 
in its original language
chat_history: {chat_history}
follow_up_input: {question}
standalone question:
"""
CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

# step of rag:
"""
step1: extract the data from the data source
step2: convert the extracted data into chunks
step3: convert chunks into embedding 
step4: storing the embedding in vector database
step5: generating the answer using llm model
"""

def extract_text_from_pdf(docs):
    """
    extact the data from the documents and documents can be pdf, execl file ,etc.

    arg: 
        docs: the documents from which going to extract the text
    
    return:
        text: extracted text from the documents
    """

    logging.info("Extracting text from pdf")
    try:
        text = ""
        for pdf in docs:
            pdf_reader = PdfReader(pdf)
            for pages in pdf_reader.pages:
                text += pages.extract_text()
            
            return text
        logging.info("extracting text from pdf is completed successfully")
    except Exception as e:
        logging.info(f"error in extraction of text: {e}")
    

def text_chunk(text):
    """
    this function will chunk the extracted text into specific chunks methods
    
    args:
        text: extracted text from documents
    
    return:
        text: chunk text from the documents 
    """
    logging.info("chunking extracted text using charater wise splitting")
    try:
        text_splitter = CharacterTextSplitter(
            separator= "\n",
            chunk_size = 512,
            chunk_overlap = 100,
            lenght_function = "len",
        )
        chunks = text_splitter.split_text(text)
        logging.info("chunking extracted text is completed successfully")
        return chunks
    except Exception as e:
        logging.info(f"error in chunking the extracted text: {e}")

def get_vectorstore(chunk_data):
    """
    it will convert the chunk_data into a embedding vector and then embedding vector will store in vector database

    arg:
        chunk_data: text data which is chunked.
    
    return:
        store the vector of chunk_data into the vector database
    """

    try:
        logging.info("converting the chunk_data into a embedding vector and storing in vector database")
        embedding = HuggingFaceBgeEmbeddings(
            model_name = "sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs = {"device": 'cpu'}
        )
        vectorstore = faiss.FAISS.from_texts(
            texts= chunk_data,
            embedding= embedding
        )
        logging.info("sucessfully done the embedding vector and storing")
        return vectorstore
    except Exception as e:
        logging.info(f"error in the vectorstore: {e}")

def get_conversational_chain(vectorstore):
    """

    """
    try:
        logging.info("calling the LLM")
        llm = ChatGoogleGenerativeAI(
            temperature= 0.2,
            top_k=0.3
        )

        logging.info("making the memory buffer")
        memory = ConversationBufferMemory(
            memory_key= "chat_history",\
            return_messages= True,
            output_key= "answer",
            )
        
        logging.info("making the conversation chains")
        coversation_chain = ConversationalRetrievalChain.from_llm(
            llm= llm,
            retriever= vectorstore,
            condense_question_llm= CUSTOM_QUESTION_PROMPT,
            memory= memory,
        )

        logging.info("coversation chain are completed sucessfully")
        return coversation_chain
    except Exception as e:
        logging.info(f"error in conversational chain building: {e}")

def generate_answer(question):
    pass


