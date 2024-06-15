import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from flask import Flask, request, jsonify
from get_embedding_function import get_embedding_function
# import fitz  # PyMuPDF
from werkzeug.utils import secure_filename

# Constants for the Chroma path and the prompt template
CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Please provide an answer to the following question by considering not only the context provided below. Use also external knowledge or assumptions.

Context:
{context}

Question:
{question}

Your Answer:
"""

# Flask app setup
app = Flask(__name__)

# Function to query the RAG model
def query_rag(query_text: str):
    # Prepare the DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB
    results = db.similarity_search_with_score(query_text, k=5)

    # Format the context and the prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Invoke the model and get the response
    model = Ollama(model="llama3")
    response_text = model.invoke(prompt)

    # Gather sources and format the response
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response

# API endpoint for the chat function
@app.route('/chat', methods=['POST'])
def my_api():
    data = request.get_json()
    query_text = data['query_text']
    response = query_rag(query_text)
    return jsonify({'result': response})



# Main function to run the Flask app
def main():
    app.run(debug=True)

# Entry point for the script
if __name__ == "__main__":
    main()








# original_code

# import argparse
# from langchain_community.vectorstores import Chroma
# from langchain.prompts import ChatPromptTemplate
# from langchain_community.llms.ollama import Ollama

# from get_embedding_function import get_embedding_function

# CHROMA_PATH = "chroma"

# PROMPT_TEMPLATE = """
# Answer the question based only on the following context:

# {context}

# ---

# Answer the question based on the above context: {question}
# """


# def main():
#     # Create CLI.
#     parser = argparse.ArgumentParser()
#     parser.add_argument("query_text", type=str, help="The query text.")
#     args = parser.parse_args()
#     query_text = args.query_text
#     query_rag(query_text)


# def query_rag(query_text: str):
#     # Prepare the DB.
#     embedding_function = get_embedding_function()
#     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

#     # Search the DB.
#     results = db.similarity_search_with_score(query_text, k=5)

#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
#     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     prompt = prompt_template.format(context=context_text, question=query_text)
#     # print(prompt)

#     model = Ollama(model="mistral")
#     response_text = model.invoke(prompt)

#     sources = [doc.metadata.get("id", None) for doc, _score in results]
#     formatted_response = f"Response: {response_text}\nSources: {sources}"
#     print(formatted_response)
#     return response_text


# if __name__ == "__main__":
#     main()








# import base64
# import PyPDF2
# from io import BytesIO
# import argparse
# from langchain_community.vectorstores import Chroma
# from langchain.prompts import ChatPromptTemplate
# from langchain_community.llms.ollama import Ollama
# from flask import Flask, request, jsonify
# from get_embedding_function import get_embedding_function
# from fpdf import FPDF

# # Constants for the Chroma path and the prompt template
# CHROMA_PATH = "chroma"
# PROMPT_TEMPLATE = """
# Answer the question based only on the following context:

# {context}

# ---

# Answer the question based on the above context: {question}
# """

# # Flask app setup
# app = Flask(__name__)

# # Function to extract text from PDF
# def extract_text_from_pdf(pdf_base64):
#     # Decode the base64 string to bytes
#     pdf_bytes = base64.b64decode(pdf_base64)
#     pdf = BytesIO(pdf_bytes)
    
#     # Read text from PDF
#     reader = PyPDF2.PdfFileReader(pdf)
#     text = ""
#     for page_num in range(reader.numPages):
#         text += reader.getPage(page_num).extractText()
#     return text

# # Function to query the RAG model
# def query_rag(query_text: str):
#     # Prepare the DB
#     embedding_function = get_embedding_function()
#     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

#     # Search the DB
#     results = db.similarity_search_with_score(query_text, k=5)

#     # Format the context and the prompt
#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
#     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     prompt = prompt_template.format(context=context_text, question=query_text)

#     # Invoke the model and get the response
#     model = Ollama(model="llama3")
#     response_text = model.invoke(prompt)

#     # Gather sources and format the response
#     sources = [doc.metadata.get("id", None) for doc, _score in results]
#     formatted_response = f"Response: {response_text}\nSources: {sources}"
#     return formatted_response



# # API endpoint لاستخراج النص من PDF وترجمته وإرجاعه كـ PDF جديد بتنسيق base64
# @app.route('/translate_and_create_pdf', methods=['POST'])
# def translate_and_create_pdf_api():
#     data = request.get_json()
#     pdf_base64 = data['pdf_base64']
#     select_Lang = data['select_Lang']
    
#     # استخراج النص من PDF
#     extracted_text = extract_text_from_pdf(pdf_base64)
    
#     # ترجمة النص
#     translated_text_response = query_rag(extracted_text + " قم بترجمة الن ص للغة " + select_Lang)
    
#     # إنشاء PDF جديد يحتوي على النص المترجم
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", size=12)
#     pdf.multi_cell(0, 10, translated_text_response)
    
#     # حفظ PDF في ذاكرة الوصول العشوائي
#     pdf_output = BytesIO()
#     pdf.output(pdf_output)
#     pdf_output.seek(0)
    
#     # تحويل PDF إلى base64 string
#     pdf_base64 = base64.b64encode(pdf_output.read()).decode('utf-8')
    
#     # إرجاع base64 string كـ JSON response
#     return jsonify({'pdf_base64': pdf_base64})

# # API endpoint for the chat function
# @app.route('/chat', methods=['POST'])
# def my_api():
#     data = request.get_json()
#     query_text = data['query_text']
#     response = query_rag(query_text)
#     return jsonify({'result': response})

# # Main function to run the Flask app
# def main():
#     app.run(debug=True)

# # Entry point for the script
# if __name__ == "__main__":
#     main()








# import argparse
# from langchain_community.vectorstores import Chroma
# from langchain.prompts import ChatPromptTemplate
# from langchain_community.llms.ollama import Ollama
# from flask import Flask, request, jsonify
# from get_embedding_function import get_embedding_function

# # Constants for the Chroma path and the prompt template
# CHROMA_PATH = "chroma"
# PROMPT_TEMPLATE = """
# Answer the question based only on the following context:

# {context}

# ---

# Answer the question based on the above context: {question}
# """

# # Flask app setup
# app = Flask(__name__)

# # Function to query the RAG model
# def query_rag(query_text: str):
#     # Prepare the DB
#     embedding_function = get_embedding_function()
#     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

#     # Search the DB
#     results = db.similarity_search_with_score(query_text, k=5)

#     # Format the context and the prompt
#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
#     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     prompt = prompt_template.format(context=context_text, question=query_text)

#     # Invoke the model and get the response
#     model = Ollama(model="llama3")
#     response_text = model.invoke(prompt)

#     # Gather sources and format the response
#     sources = [doc.metadata.get("id", None) for doc, _score in results]
#     formatted_response = f"Response: {response_text}\nSources: {sources}"
#     return formatted_response

# # API endpoint for the chat function
# @app.route('/chat', methods=['POST'])
# def my_api():
#     data = request.get_json()
#     query_text = data['query_text']
#     response = query_rag(query_text)
#     return jsonify({'result': response})

# # Main function to run the Flask app
# def main():
#     app.run(debug=True)

# # Entry point for the script
# if __name__ == "__main__":
#     main()









# import argparse
# # from langchain.vectorstores.chroma import Chroma
# from langchain_community.vectorstores import Chroma 
# from langchain.prompts import ChatPromptTemplate
# from langchain_community.llms.ollama import Ollama
# from flask import Flask, request, jsonify

# from get_embedding_function import get_embedding_function



# CHROMA_PATH = "chroma"

# PROMPT_TEMPLATE = """
# Answer the question based only on the following context:

# {context}

# ---

# Answer the question based on the above context: {question}
# """


# def main():
#     # Create CLI.
#     parser = argparse.ArgumentParser()
#     parser.add_argument("query_text", type=str, help="The query text.")
#     args = parser.parse_args()
#     query_text = args.query_text
#     query_rag(query_text)


# def query_rag(query_text: str):
#     # Prepare the DB.
#     embedding_function = get_embedding_function()
#     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

#     # Search the DB.
#     results = db.similarity_search_with_score(query_text, k=5)

#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
#     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     prompt = prompt_template.format(context=context_text, question=query_text)
#     print(prompt)

#     model = Ollama(model="llama3")
#     response_text = model.invoke(prompt)

#     sources = [doc.metadata.get("id", None) for doc, _score in results]
#     formatted_response = f"Response: {response_text}\nSources: {sources}"
#     print(formatted_response)
#     return response_text

# app = Flask(__name__)
# @app.route('/chat', methods=['POST'])
# def my_api():
#     data = request.get_json()  # الحصول على البيانات من الطلب
#     my_string = data['my_string']  # استخراج السلسلة من البيانات
#     return jsonify({'result': my_string})  # إرجاع السلسلة كجزء من الاستجابة

# if __name__ == '__main__':
#     app.run(debug=True)


# if __name__ == "__main__":
#     main()
    
