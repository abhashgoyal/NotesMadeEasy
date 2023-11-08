import os
import streamlit as st
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# sidebar
with st.sidebar:
    st.title("PDF ChatBot:")
    st.markdown("Hello, this will help you summarize your PDFs and text.")
    st.write('Made By Abhash')

def main():
    st.header("Chat with PDF ☁️")
    load_dotenv()


    # Uploading PDF files
    pdf = st.file_uploader("Upload your PDF here:", type='pdf')
    # st.write(pdf.name)
    # load_dotenv()

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = " "
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len  # Corrected the typo
        )
        chunks = text_splitter.split_text(text=text)
        st.write(chunks)

        # #embeddings
        embeddings = OpenAIEmbeddings()

        VectorStore= FAISS.from_texts(chunks,embedding=embeddings)

        store_name=pdf.name[:-4]
        with open(f"{store_name}.pkl","wb") as f:
            pickle.dump(VectorStore,f)

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                VectorStore=pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore= FAISS.from_texts(chunks,embedding=embeddings)
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(VectorStore,f)
        
    #USER QUESTIONS
        query = st.text_input("Ask questions On DOCS:")
    # st.write(query)
        if query:
            docs = VectorStore.similarity_search(query=query,k=3)
            llm=OpenAI(temperature=0,)
            chain=load_qa_chain(llm=llm,chain_type="stuff")
            response=chain.run(input_documents=docs,question=query)
            st.write(response)
        # st.write(docs)

if __name__ == '__main__':
    main()
