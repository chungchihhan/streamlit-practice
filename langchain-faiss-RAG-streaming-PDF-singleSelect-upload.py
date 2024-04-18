import streamlit as st
import os
import tempfile
from langchain import hub
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

st.title("RAG: Single Select PDF Reading")

# Create a temporary directory to store uploaded files
if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = tempfile.TemporaryDirectory()

# Get the path of the temporary directory
temp_dir_path = st.session_state.temp_dir.name

with st.sidebar:
    uploaded_file = st.file_uploader("Choose a PDF file")
    if uploaded_file is not None:
        file_path = os.path.join(temp_dir_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Saved File")

    # List files in the temporary directory
    files_list = os.listdir(temp_dir_path)
    if files_list:
        options = st.selectbox('Select a PDF file', files_list)

# Define the function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if 'options' in locals():
    file_path = os.path.join(temp_dir_path, options)
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
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

    # Display and manage chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    def response_generator(user_prompt):
        stream = rag_chain_with_source.stream(user_prompt)
        for chunk in stream:
            if 'answer' in chunk and chunk['answer']:
                for word in chunk['answer'].split(" "):
                    yield word + " "

    if user_prompt := st.chat_input("What is up?"):
        with st.chat_message("user"):
            st.markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        with st.chat_message("assistant"):
            response = st.write_stream(response_generator(user_prompt=user_prompt))
        st.session_state.messages.append({"role": "assistant", "content": response})


# # Clean up the session state
# for key in st.session_state.keys():
#     del st.session_state[key]
