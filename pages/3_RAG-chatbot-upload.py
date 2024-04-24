from langchain import hub
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
import streamlit as st
import os 
import tempfile

st.title("NTU 115 chatbot-upload")

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


def response_generator(prompt, rag_chain_with_source):
    # Start the stream from the RAG chain with the provided prompt
    stream = rag_chain_with_source.stream(prompt)
    for output in stream:
        if 'answer' in output and output['answer']:
            for word in output['answer']:
                yield word 

@st.cache_resource(experimental_allow_widgets=True, show_spinner=False)
def setup_response_generator(openai_api_key, _docs):
    """Setup the response generator components."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings(openai_api_key=openai_api_key))
    retriever = vectorstore.as_retriever(k=5)
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key)
    
    return RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=(
        RunnablePassthrough.assign(context=(lambda x: "\n\n".join(doc.page_content for doc in x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    ))

if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = tempfile.TemporaryDirectory()
temp_dir_path = st.session_state.temp_dir.name

with st.sidebar:
    uploaded_file = st.file_uploader("Choose a PDF file")
    if uploaded_file is not None:
        file_path = os.path.join(temp_dir_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Saved File")


    openai_api_key = st.secrets["OPENAI_API_KEY"]
    files_list = os.listdir(temp_dir_path)

    if files_list and openai_api_key:
        # Set default selection to the first file in the list or any specific index
        default_index = len(files_list)-1 
        options = st.selectbox('Select a file:', files_list, index=default_index)
        file_path = f"{temp_dir_path}/{options}"
        docs = load_document(file_path)
        if 'file_path' not in st.session_state or st.session_state.file_path != file_path:
            st.session_state.file_path = file_path
            setup_response_generator.clear()
        with st.spinner("Your chatbot is cooking"):
            rag_chain_with_source = setup_response_generator(openai_api_key, docs)
        # st.toast("Your chatbot is ready !", icon='😍')

# Main code
if openai_api_key and setup_response_generator :
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if 'rag_chain_with_source' in locals():  # 確保 rag_chain_with_source 存在
            with st.chat_message("assistant"):
                response = st.write_stream(response_generator(prompt, rag_chain_with_source))
            st.session_state.messages.append({"role": "assistant", "content": response})