import streamlit as st
import os
from google.cloud import storage

def upload_blob(bucket_name, destination_blob_name, file):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_file(file)

    print(f"File {destination_blob_name} uploaded to {bucket_name}.")
    st.write("File: ", destination_blob_name, "uploaded. ")

st.title("Valor Assistant")

uploaded_files = st.file_uploader("Upload PDF", type=".pdf", accept_multiple_files=True, 
                 key=None, help=None, on_change=None, args=None, kwargs=None, 
                 disabled=False, label_visibility="visible")

bucket_name = st.secrets["default"]["bucket_name"]

for uploaded_file in uploaded_files:
    upload_blob(bucket_name, uploaded_file.name, uploaded_file)



