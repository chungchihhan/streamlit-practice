from openai import OpenAI
import streamlit as st

st.title("ChatGPT-like clone")

# Let user enter their own api-key
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
client = OpenAI(api_key=openai_api_key)

# Use the OpenAI API key from the secrets. connect it with the secret in streamlit/secrets.toml
# client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if not openai_api_key or openai_api_key[:3] != "sk-" or openai_api_key == " ":
    st.error("Please enter the OpenAI API Key in the sidebar correctly to continue.")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})