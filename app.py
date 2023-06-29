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
from flask import Flask, request, jsonify
from twilio.twiml.messaging_response import MessagingResponse
from queue import Queue
import time
import os
from twilio.rest import Client

load_dotenv()

chunks = []

account_sid = os.getenv('ACCOUNT_SID')
auth_token = os.getenv('AUTH_TOKEN')

def makeChunks():
    load_dotenv()
    
    file_path = "./documents"    
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
    
    
def main(ask):
    
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
makeChunks()

fila = Queue()
client = Client(account_sid, auth_token)

def funcao_segundo_plano():
    while True:
        if not fila.empty():
            fila.get
        time.sleep(1)


@app.route('/twilio-ask-ai', methods=['POST'])
def post():

    message = client.messages.create(
        from_=request.form['To'],
        body='Analisando sua pergunta',
        to=request.form['From']
    )

    print(message)
    
    fila.put(funcao_em_segundo_plano(request.form['To'], request.form['Body'], request.form['From']))
    
    return str(message)

def funcao_em_segundo_plano(fromP, body, toP):
    
    time.sleep(30)
    print(toP)
    print(fromP)
    
    responseIA = main(body)
    
    message = client.messages.create(
        from_=fromP,
        body=responseIA,
        to=toP
    )

    print(message)
    


@app.route('/', methods=['GET'])
def get():
   return "<h1 style='color:black'>Hi!</h1>"
    
    
    