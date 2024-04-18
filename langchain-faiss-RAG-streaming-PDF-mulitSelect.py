import bs4
from langchain import hub
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st

st.title("RAG: Multi-Select PDF reading")

options = st.multiselect(
    'Select a PDF file',
    ["TS31103 - Adjustment of Backlash for GXA-S Series GXA-S背隙調整.pdf","TS31510 - Adjustment of IFM Pressure Switch壓力開關(IFM宜福門)調整及使用.pdf","TS31621 - Replacement of Lubrication(For GXA-S Series)GXA-S潤滑油更換.pdf"])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if not options:
    st.stop()

for i in options:
    file_path = f"files/{i}"

    # loader = PyPDFLoader("files/"+ read_PDF)
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
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
    for output in stream:
        if 'answer' in output and output['answer']:
            # Assuming output['answer'] is a string, split and yield each word
            for word in output['answer'].split():
                yield word + " "

if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt= prompt))
    st.session_state.messages.append({"role": "assistant", "content": response})