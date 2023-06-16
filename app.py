from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import os
import glob
from flask import Flask, request, make_response, jsonify

def main(ask):
    load_dotenv()
    
    file_path = "./documents"
    chunks = []
    
    arquivos_pdf = []

    for arquivo in glob.glob(os.path.join(file_path, "*.pdf")):
        arquivos_pdf.append(arquivo)

        for arquivo_pdf in arquivos_pdf:
            
            with open(arquivo_pdf, "rb") as file:

                if file is not None:
                    pdf_reader = PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()

                    text_splitter = CharacterTextSplitter(
                        separator="\n",
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len
                    )
                    
                    for chunk in text_splitter.split_text(text):
                        chunks.append(chunk)

    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    while True:
        user_question = ask
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)
        
        return str(response)


app = Flask(__name__)

@app.route('/safety-ask', methods=['GET'])
def get():
    
    ask = request.args.get('value')
    
    response = main(ask)

    return response

app.run()
        