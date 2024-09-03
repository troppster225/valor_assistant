import streamlit as st

st.title("Valor Assistant")

st.file_uploader("Upload PDF", type=".pdf", accept_multiple_files=True, 
                 key=None, help=None, on_change=None, args=None, kwargs=None, 
                 disabled=False, label_visibility="visible")




