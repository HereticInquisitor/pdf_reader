import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import io
import tempfile

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# def get_pdf_text(pdf_docs):
#     text=""
#     # Create a temporary file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#         temp_file.write(pdf_docs.read())
#         temp_file_path= temp_file.name

#     # Read the pages
#     pdf_reader= PdfReader(temp_file_path)
#     for page in pdf_reader.pages:
#         text+= page.extract_text()

#     # # We Read the pages the extract the text from each pages of the pdf
#     # for pdf in pdf_docs:
#     #     pdf_reader=PdfReader(io.BytesIO(pdf.read()))
#     #     for page in pdf_reader.pages:
#     #         text+=page.extract_text()
#     os.remove(temp_file_path)
#     return text

def get_pdf_text(pdf_docs):
    text = ""
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_file.write(pdf_docs.read())
    temp_file_path = temp_file.name

    # Read the pages and extract text
    pdf_reader = PdfReader(temp_file_path)
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Remove the temporary file
    os.remove(temp_file_path)

    return text


def get_chunks(text):
    text_splitter= RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks= text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings= GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store= FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template_str="""
    Answer the question as detailed as possible from the provided context, make sure to provide all the details,
    if the details is not provided in the provided context just say, "answer is not available in the context",
    don't provide wrong answers\n\n
    Context:\n {context}?\n
    Question:\n{question}\n

    Answer: 

    """

    model=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    prompt=PromptTemplate(template=prompt_template_str, input_variables=["context","question"])
    chain=load_qa_chain(model, chain_type="stuff",prompt=prompt)
    return chain

def user_input(user_question):
    embeddings= GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db= FAISS.load_local("faiss_index", embeddings)
    docs= new_db.similarity_search(user_question)

    chain= get_conversational_chain()

    response= chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    
    print(response)
    st.write("Reply:", response["output_text"])


def main():
    st.set_page_config("Chat with PDF")
    st.header("Gemini Pdf reader")

    user_question= st.text_input("Ask a question from the PDF files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs= st.file_uploader("Upoload your PDF files here")
        if st.button("Submit & Process"):
            with st.spinner("Processing"):
                raw_text= get_pdf_text(pdf_docs)
                text_chunks= get_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ =="__main__":
    main()