import fitz
import streamlit as st
import pandas as pd
import os 

st.set_page_config(page_title="PDF Table Extractor", layout="wide")
st.title("Extracting Tables from PDF")


pdf_path = "files"
pdf_list = os.listdir(pdf_path)
option = st.selectbox("Select a pdf file", pdf_list , index = 2)
doc = fitz.open("files/" + option)

pages = st.slider(
    "Select a range of pages",
    1, len(doc), (1, 2)
)

for i in range(pages[0]-1, pages[1]):
    st.subheader(f"Page {i + 1}")
    with st.container(border=True):
        tables = doc[i].find_tables()
        st.write(f"{len(tables.tables)} table(s) found on page {i + 1}")
        
        for table in tables:
            data = table.extract()
            
            columns = []
            for idx, col in enumerate(data[0]):
                if col is None or col == "":
                    columns.append(f"Unnamed: {idx}")
                else:
                    columns.append(col)
            
            df = pd.DataFrame(data[1:], columns=columns)
            st.dataframe(df, use_container_width=True)