import base64
from io import BytesIO
import argparse
import os
import tempfile
from bson import json_util
import shutil
from langchain.prompts import ChatPromptTemplate
from flask import Flask, request, jsonify
from flask_cors import CORS
from waitress import serve
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema.document import Document 
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from fpdf import FPDF
from pdfminer.high_level import extract_text
from langchain_community.chat_models import ChatOllama
from langchain_community.llms.ollama import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import populate_TempDatabase

# Flask app setup
app = Flask(__name__)
CORS(app)#...

PROMPT_TEMPLATE = """
Please provide only an answer to the following question by considering not only the context provided below
Context:
{context}

Question:
{question}

Your Answer:
"""

# ----------------- ADD file to chroma database --------- QA page ------1
filePath = 'dataTemp\output.pdf'
CHROMA_PATH_TEMP = "chromaTemp"
CHROMA_PATH = "chroma"
DATA_PATH = "data"
DATA_PATH_TEMP = "dataTemp"
filename= "output.pdf"
@app.route('/addFile', methods=['POST'])
def addFile():

    try:
        data = request.get_json()
        pdf_base64 = data['pdf_base64']
        if not os.path.exists(DATA_PATH_TEMP):
            raise ValueError(f"Directory '{DATA_PATH}' does not exist.")
        pdf_bytes = base64.b64decode(pdf_base64)#pdf file
        pdf= BytesIO(pdf_bytes) 

        save_path = os.path.join(DATA_PATH, filename)
        with open(save_path,'wb') as f :
            f.write(pdf.getvalue())

        print(f"PDF saved successfully to: {save_path}")

        # --------add to db section -------
        populate_TempDatabase.main()
        # ------- success response --------
        return jsonify({'result': 'PDF saved successfully'})
    
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500


# -------------- Delete File ------------------------ QA page -----2

@app.route('/deleteFile', methods=['POST'])
def deleteFile():
  
    data = request.get_json()
    remove_order =data['query_text']
    if str(remove_order) == 'remove':
        if os.path.exists(filePath):
            os.remove(filePath)
            print(f"the file {filePath} hs been deleted") 
            return jsonify({'result': f"the file {filePath} hs been deleted"})
        else:

            return jsonify({'result': f"the file {filePath} does not exist."})
            

    else:
        return jsonify({'result': 'this is not the delete order'})
        



# ------------ Q/A Text Generation code --------------- QA page -----3






def QAquery_rag(query_text: str):
    # Prepare the DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH_TEMP, embedding_function=embedding_function)

    # Search the DB
    results = db.similarity_search_with_score(query_text, k=5)

    # Format the context and the prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Invoke the model and get the response
    model = Ollama(model="phi3:mini",callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    response_text = model.invoke(prompt)
    response_content = response_text['content'] if isinstance(response_text, dict) else response_text
    # Gather sources and format the response
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_content}"
    print(formatted_response)
    return  formatted_response

@app.route('/QA', methods=['POST'])
def QA_api():
    data = request.get_json()
    query_text = data['query_text']
    response = QAquery_rag(query_text)
    return jsonify({'result':response})

# ------------ Transalte files ----------------translate page---4

def extract_text_from_pdf(pdf_base64):
    try:
        pdf_bytes = base64.b64decode(pdf_base64)
        pdf = BytesIO(pdf_bytes)
        text = extract_text(pdf)
        return text
    except Exception as e:
        print(f"Error: {e}")
        return None



def TranslateQuery_rag(query_text: str):
    # Prepare the DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB
    results = db.similarity_search_with_score(query_text, k=5)

    # Format the context and the prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context="", question=query_text)

    # Invoke the model and get the response
    model = Ollama(model="llama3")
    response_text = model.invoke(prompt)

    # Gather sources and format the response
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f" {response_text}"
    return formatted_response



# API endpoint لاستخراج النص من PDF وترجمته وإرجاعه كـ PDF جديد بتنسيق base64
@app.route('/translate', methods=['POST'])
def translate_and_create_pdf_api():
    try:
        data = request.get_json()
        pdf_base64 = data['pdf_base64']
        select_Lang = data['select_Lang']
        
        # Extract text from PDF
        extracted_text = extract_text_from_pdf(pdf_base64)
        
        
        # Translate text
        translated_text_response = TranslateQuery_rag( f"Translate to {select_Lang}: {extracted_text}.(Answer with the translated text only)" )

        print(translated_text_response)
        return jsonify({'pdf_base64': translated_text_response})
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500
    

# ------------------ summarize texts -------------- summarize page ---5


def summarizeQuery_rag(query_text: str):
    # Prepare the DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB
    results = db.similarity_search_with_score(query_text, k=5)

    # Format the context and the prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context="", question=query_text)

    # Invoke the model and get the response
    model = Ollama(model="phi3:mini")
    response_text = model.invoke(prompt)

    # Gather sources and format the response
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"your summarized text :  {response_text}"
    return formatted_response

# API endpoint for the chat function
@app.route('/summarize', methods=['POST'])
def summarize_api():
    data = request.get_json()
    query_text = data['query_text']
    response = summarizeQuery_rag(f"Summarize this text: {query_text}.(Answer with the summarized text only)")
    return jsonify({'result': response})



# --------- normal chat ---------


def chatQuery_rag(query_text: str):
    # Prepare the DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB
    results = db.similarity_search_with_score(query_text, k=5)

    # Format the context and the prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context="", question=query_text)

    # Invoke the model and get the response
    model = Ollama(model="phi3:mini")
    response_text = model.invoke(prompt)

    # Gather sources and format the response
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    # formatted_response = f"your summarized text :  {response_text}"
    return response_text

# API endpoint for the chat function
@app.route('/chat', methods=['POST'])
def chat_api():
    data = request.get_json()
    query_text = data['query_text']
    response = chatQuery_rag(f"Answer this: {query_text}.(Answer the question only)")
    return jsonify({'result': response})


def main():
    app.run(debug=True, host='192.168.137.1', port=5000)#....

if __name__ == "__main__":
    main()



