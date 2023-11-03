
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# sidebar
with st.sidebar:
    st.title("PDF ChatBot:")
    st.markdown("Hello, this will help you summarize your PDFs and text.")
    st.write('Made By Abhash')

def main():
    st.header("Chat with PDF ☁️")

    # Uploading PDF files
    pdf = st.file_uploader("Upload your PDF here:", type='pdf')

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

        

if __name__ == '__main__':
    main()
