import base64
from io import BytesIO
import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from flask import Flask, request, jsonify
from flask_cors import CORS
from waitress import serve
from get_embedding_function import get_embedding_function
from fpdf import FPDF
from pdfminer.high_level import extract_text

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


# PROMPT_TEMPLATE = """
# Answer the question based only on the following context:

# {context}

# ---

# Answer the question based on the above context: {question}
# """

# Flask app setup
app = Flask(__name__)
CORS(app)#...

# Function to extract text from PDF



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
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Invoke the model and get the response
    model = Ollama(model="llama3")
    response_text = model.invoke(prompt)

    # Gather sources and format the response
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
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


def main():
    app.run(debug=True, host='0.0.0.0', port=5000)

# Entry point for the script
if __name__ == "__main__":
    main()


