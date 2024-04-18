from langchain import hub
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st
import os
import time

st.title("RAG: Single Select PDF reading")

with st.sidebar:
    if uploaded_file := st.file_uploader("Choose a file"):
        with open(os.path.join("files",uploaded_file.name),"wb") as f: 
            f.write(uploaded_file.getbuffer())         
        st.success("Saved File")
    files_list = os.listdir("files")
    options = st.selectbox('Select a PDF file',files_list)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

file_path = f"files/{options}"

# loader = PyPDFLoader("files/"+ read_PDF)
loader = PyPDFLoader(file_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)
vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever(k=10)
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def response_generator(prompt):
    # Start the stream from the RAG chain with the provided prompt
    stream = rag_chain_with_source.stream(prompt)
    for chunk in stream:
        if 'answer' in chunk and chunk['answer']:
            # Assuming output['answer'] is a string, split and yield each word
            for word in chunk['answer'].split(" "):
                yield word + " "


    
if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt= prompt))
    st.session_state.messages.append({"role": "assistant", "content": response})