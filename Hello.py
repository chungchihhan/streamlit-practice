import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to NTU 115 chatbot! 👋")

# st.sidebar.success("Select a function above.")

st.markdown(
    """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed nec purus eget
    ipsum elementum aliquam. Nullam nec nunc et nisl ultricies tincidunt.
    **👈 Select a page from the sidebar** to use our chatbot
    ### Want to learn more?
    - Lorem ipsum dolor sit amet, consectetur adipiscing elit.
    - Sed nec purus eget ipsum elementum aliquam. Nullam nec nunc et nisl ultricies tincidunt.
    ### See more complex demos
    - Lorem ipsum dolor sit amet, consectetur adipiscing elit.
    - Sed nec purus eget ipsum elementum aliquam. Nullam nec nunc et nisl ultricies tincidunt.
"""
)