import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Carrega variáveis de ambiente do .env
load_dotenv()

# Obtém a chave da API do Google a partir das variáveis de ambiente
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Função para extrair texto de um ou mais arquivos PDF
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text


# Função para dividir o texto em pedaços menores
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    chunks = text_splitter.split_text(text)
    return chunks


# Função para criar e salvar o armazenamento vetorial
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# Função para criar a cadeia de conversação
def get_conversational_chain():

    prompt_template = """
    Responda à pergunta o mais detalhadamente possível a partir do contexto fornecido, certifique-se de fornecer todos os detalhes, se a resposta não estiver correta
    fornecido o contexto, basta dizer "a resposta não está disponível no contexto", não forneça a resposta errada\n\n
    Contexto:\n {context}?\n
    Pergunta: \n{question}\n
    Resposta:
    """

    # Cria um modelo de linguagem Google Generative AI
    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    # Cria um objeto PromptTemplate a partir do template definido
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    # Cria a cadeia de perguntas e respostas usando o modelo e o prompt
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


# Função para processar a entrada do usuário e gerar uma resposta
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    print(response)
    st.write("Reply: ", response["output_text"])



# Função principal da aplicação Streamlit
def main():
    st.set_page_config("Implementação de Assistente Conversacional Baseado em LLM")
    st.header("Implementação de Assistente Conversacional Baseado em LLM")
    user_question = st.text_input("Faça uma pergunta a respeito dos PDFs")
    if user_question:
        user_input(user_question)
    # Barra lateral
    with st.sidebar:
        st.title("Seus Documentos")
        pdf_docs = st.file_uploader("Faça Upload dos seus PDFs e clique em 'Processar'", accept_multiple_files=True)
        if st.button("Processar"):
            with st.spinner("Processando..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processado!")


# Executa a função main se o script for executado diretamente
if __name__ == "__main__":
    main()