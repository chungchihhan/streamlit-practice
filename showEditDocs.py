from langchain import hub
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
import streamlit as st
import os 

st.set_page_config(layout="wide")
st.title("NTU 115 chatbot")
col1, col2 = st.columns(2)

def load_text(file_path):
    loader = TextLoader(file_path)
    docs = loader.load()
    return docs

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

def load_document(file_path):
    """Load document based on file extension."""
    if file_path.endswith('.txt') or file_path.endswith('.md'):
        return load_text(file_path)
    elif file_path.endswith('.pdf'):
        return load_pdf(file_path)
    else:
        st.error("Unsupported file format")
        return []

files_list = sorted(os.listdir("files"))
col1.header("Read file")
options = col1.selectbox('Select a file:', files_list)
file_path = f"files/{options}"
docs = load_document(file_path)

updated_docs_content = {}  # To collect updated content

for i, doc in enumerate(docs):
    col1.write(f"Page {doc.metadata['page']}")
    # Create a text area for each page's content, the user can edit it
    updated_content = col1.text_area(f"Original Page {doc.metadata['page']} Content:", value=doc.page_content, key=i, height=200)
    updated_docs_content[i] = updated_content
    # updated_docs_content.append(updated_content)
    col1.divider()


combined_content = []
for i in range(len(updated_docs_content)):
    combined_content.append(updated_docs_content[i])
combined_content = "\n".join(combined_content)


col2.header('Updated content')
updated_file_name = col2.text_input("Enter the updated file name:",value=f"Updated_{options[:-4]}")+".txt"
# col2.write(updated_file_name)

for i, doc in enumerate(updated_docs_content):
    col2.write(f"Page {i}")
    col2.text_area(f"Edited Page {i} content :", value=updated_docs_content[doc], height=200, key=f"{i}+updated",disabled=True)
    col2.divider()

col2.download_button(
    label="Download Updated file",
    data=combined_content,
    file_name=updated_file_name,
    mime="application/octet-stream",
    use_container_width=True
)   