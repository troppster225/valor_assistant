import streamlit as st

st.title("Valor Assistant")

uploaded_files = st.file_uploader("Upload PDF", type=".pdf", accept_multiple_files=True, 
                 key=None, help=None, on_change=None, args=None, kwargs=None, 
                 disabled=False, label_visibility="visible")

for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)
    st.write(bytes_data)




