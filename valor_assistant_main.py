import streamlit as st
import os
import io
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.api_core.exceptions import GoogleAPIError
from llama_cpp import Llama
import pandas as pd
from typing import Dict, List
import json

# Page configuration
st.set_page_config(page_title="Valor Assistant", page_icon="📁", layout="wide")

# Constants
ALLOWED_EXTENSIONS = {'.pdf'}

# Utility Functions
def is_valid_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

@st.cache_resource
def get_storage_client():
    try:
        return storage.Client()
    except Exception as e:
        st.error(f"Failed to initialize Google Cloud Storage client: {str(e)}")
        return None

def get_bucket_name():
    try:
        if "bucket_name" in os.environ:
            return os.getenv("bucket_name")
        return st.secrets["default"]["bucket_name"]
    except KeyError:
        st.error("Bucket name not found in environment variables or secrets.")
        return None

def get_file_icon(file_name):
    extension = os.path.splitext(file_name)[1].lower()
    if extension == '.pdf':
        return "📄"
    elif extension in ['.jpg', '.jpeg', '.png', '.gif']:
        return "🖼️"
    elif extension in ['.doc', '.docx']:
        return "📝"
    elif extension in ['.xls', '.xlsx']:
        return "📊"
    elif extension in ['.ppt', '.pptx']:
        return "📽️"
    else:
        return "📁"

def get_file_size(bucket_name, blob_name):
    storage_client = get_storage_client()
    if not storage_client:
        return "N/A"

    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        size = blob.size
        
        if size is None:
            return "N/A"
        
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.2f} KB"
        else:
            return f"{size / (1024 * 1024):.2f} MB"
    except GoogleAPIError as e:
        st.error(f"Error getting size for {blob_name}: {str(e)}")
        return "N/A"
    except Exception as e:
        st.error(f"Unexpected error getting size for {blob_name}: {str(e)}")
        return "N/A"

# File Upload Functions
@st.cache_data
def upload_blob(_bucket_name, destination_blob_name, file):
    if not is_valid_file(destination_blob_name):
        return f"Error: {destination_blob_name} is not a valid file type. Only PDF files are allowed."

    storage_client = get_storage_client()
    if not storage_client:
        return None

    try:
        bucket = storage_client.bucket(_bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_file(file)
        return f"File: {destination_blob_name} uploaded to {_bucket_name}."
    except GoogleAPIError as e:
        return f"Error uploading {destination_blob_name}: {str(e)}"
    except Exception as e:
        return f"Unexpected error uploading {destination_blob_name}: {str(e)}"

def reset_upload_session():
    st.session_state.upload_complete = False
    st.session_state.upload_summary = []
    st.session_state.uploaded_files = []

# File Explorer Functions
@st.cache_data(ttl=60)  # Cache the result for 1 minute
def list_bucket_files(bucket_name):
    """List files in the specified bucket."""
    storage_client = get_storage_client()
    if not storage_client:
        return None

    try:
        blobs = storage_client.list_blobs(bucket_name)
        return [blob.name for blob in blobs]
    except GoogleAPIError as e:
        st.error(f"Error listing files in bucket: {str(e)}")
        return None

def download_blob(bucket_name, source_blob_name):
    """Download a blob from the bucket."""
    storage_client = get_storage_client()
    if not storage_client:
        return None

    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        contents = blob.download_as_bytes()
        return contents
    except GoogleAPIError as e:
        st.error(f"Error downloading {source_blob_name}: {str(e)}")
        return None

def delete_blob(bucket_name, blob_name):
    """Delete a blob from the bucket."""
    storage_client = get_storage_client()
    if not storage_client:
        return False

    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.delete()
        return True
    except GoogleAPIError as e:
        st.error(f"Error deleting {blob_name}: {str(e)}")
        return False

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
for key in ['uploaded_files', 'upload_complete', 'upload_summary']:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ['uploaded_files', 'upload_summary'] else False

# Navigation
st.title("Valor Assistant")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Home"):
        st.session_state.current_page = 'home'
with col2:
    if st.button("Upload Files"):
        st.session_state.current_page = 'upload'
with col3:
    if st.button("File Explorer"):
        st.session_state.current_page = 'explorer'

st.divider()

# Page Content
if st.session_state.current_page == 'home':
    st.header("Welcome to Valor Assistant")
    st.write("Use the navigation buttons above to upload files or explore the file system.")

elif st.session_state.current_page == 'upload':
    st.header("File Upload")
    if not st.session_state.upload_complete:
        uploaded_files = st.file_uploader("Upload PDF", type=list(ALLOWED_EXTENSIONS), accept_multiple_files=True)

        if uploaded_files:
            new_files = [file for file in uploaded_files if file not in st.session_state.uploaded_files]
            valid_files = [file for file in new_files if is_valid_file(file.name)]
            invalid_files = [file for file in new_files if file not in valid_files]
            
            st.session_state.uploaded_files.extend(valid_files)
            
            if valid_files:
                st.write(f"{len(valid_files)} new valid files added.")
            if invalid_files:
                st.warning(f"{len(invalid_files)} files were not added due to invalid file type. Only PDF files are allowed.")

        if st.session_state.uploaded_files:
            st.write("Files ready for upload:")
            for i, file in enumerate(st.session_state.uploaded_files, 1):
                st.write(f"{i}. {file.name}")

        if st.button("Submit"):
            bucket_name = get_bucket_name()
            if not bucket_name:
                st.error("Cannot proceed without a valid bucket name.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()

                def upload_file(file):
                    return file, upload_blob(bucket_name, file.name, file)

                successful_uploads = 0
                failed_uploads = 0

                with ThreadPoolExecutor() as executor:
                    future_to_file = {executor.submit(upload_file, file): file for file in st.session_state.uploaded_files}
                    for i, future in enumerate(as_completed(future_to_file)):
                        file, result = future.result()
                        if result and not result.startswith("Error"):
                            successful_uploads += 1
                            st.session_state.upload_summary.append(result)
                        else:
                            failed_uploads += 1
                            st.session_state.upload_summary.append(f"Failed to upload {file.name}: {result}")
                        
                        progress = (i + 1) / len(future_to_file)
                        progress_bar.progress(progress)
                        status_text.text(f"Uploading... {i+1}/{len(future_to_file)}")

                if failed_uploads > 0:
                    st.warning(f"{successful_uploads} files uploaded successfully. {failed_uploads} files failed to upload.")
                else:
                    st.success(f"All {successful_uploads} files uploaded successfully!")

                st.session_state.upload_complete = True
                st.session_state.uploaded_files = []
                st.rerun()

    else:
        if any(summary.startswith("Failed") for summary in st.session_state.upload_summary):
            st.warning("Some files failed to upload. Check the summary below for details.")
        else:
            st.success("All files uploaded successfully!")
        
        for summary in st.session_state.upload_summary:
            if summary.startswith("Failed"):
                st.error(summary)
            else:
                st.write(summary)
        
        if st.button("Upload More Files"):
            reset_upload_session()
            st.rerun()

elif st.session_state.current_page == 'explorer':
    st.header("File Explorer")
    bucket_name = get_bucket_name()
    if bucket_name:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"Files in '{bucket_name}'")
        with col2:
            if st.button("🔄 Refresh", key="refresh_button"):
                st.cache_data.clear()
                st.rerun()
        
        files = list_bucket_files(bucket_name)
        if files:
            # Create a container for the file explorer
            file_explorer = st.container()
            
            with file_explorer:
                # Create a table header
                header_cols = st.columns([0.5, 2, 0.5, 0.5, 1])
                header_cols[0].markdown("**Type**")
                header_cols[1].markdown("**File Name**")
                header_cols[2].markdown("**Download**")
                header_cols[3].markdown("**Delete**")
                header_cols[4].markdown("**Size**")
                
                st.markdown("<hr>", unsafe_allow_html=True)
                
                # Display files in a table-like format
                for file in files:
                    col1, col2, col3, col4, col5 = st.columns([0.5, 2, 0.5, 0.5, 1])
                    
                    # File type icon
                    col1.markdown(get_file_icon(file))
                    
                    # File name
                    col2.markdown(file)
                    
                    # Download button
                    if col3.button("⬇️", key=f"download_{file}"):
                        file_contents = download_blob(bucket_name, file)
                        if file_contents:
                            st.download_button(
                                label="Save file",
                                data=file_contents,
                                file_name=file,
                                mime="application/octet-stream"
                            )
                    
                    # Delete button
                    if col4.button("🗑️", key=f"delete_{file}"):
                        if delete_blob(bucket_name, file):
                            st.success(f"{file} deleted successfully.")
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            st.error(f"Failed to delete {file}.")
                    
                    # File size
                    col5.markdown(get_file_size(bucket_name, file))
                    
                    st.markdown("<hr>", unsafe_allow_html=True)
            
            # Add some custom CSS to make it look more like a file explorer and handle responsiveness
            st.markdown("""
            <style>
            .stButton>button {
                width: 100%;
                border: none;
                background-color: transparent;
                padding: 2px;
            }
            .stButton>button:hover {
                background-color: #f0f0f0;
            }
            /* Responsive design */
            @media (max-width: 1200px) {
                .stButton>button {
                    font-size: 12px;
                }
            }
            @media (max-width: 992px) {
                .stButton>button {
                    font-size: 10px;
                }
            }
            @media (max-width: 768px) {
                .stButton>button {
                    font-size: 8px;
                }
            }
            /* Ensure the file name doesn't overflow */
            .element-container:nth-child(2) div {
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            </style>
            """, unsafe_allow_html=True)
            
        else:
            st.write("No files found in the bucket or error occurred.")
    else:
        st.error("Unable to retrieve bucket name.")

# Add Llama model initialization
@st.cache_resource
def initialize_llama():
    try:
        # Update the path to your Llama model
        model_path = "path/to/your/llama/model.gguf"
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,  # Context window
            n_threads=4   # Number of CPU threads to use
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize Llama model: {str(e)}")
        return None

# Add business matching function
def match_business_parameters(llm: Llama, parameters: Dict, one_pagers: List[Dict]) -> str:
    # Create prompt
    prompt = f"""You are a business matching expert. Analyze these business parameters and compare them with private equity one-pagers to find the best matches.

Business Parameters:
{json.dumps(parameters, indent=2)}

Available One-Pagers:
{json.dumps(one_pagers, indent=2)}

Please provide the top 3 matches with explanations for why they are good fits. Consider factors like industry alignment, financial compatibility, and growth potential.
"""
    
    # Get response from Llama
    response = llm(
        prompt,
        max_tokens=1024,
        temperature=0.1,
        top_p=0.95,
        stop=["</s>", "\n\n\n"]
    )
    
    return response['choices'][0]['text']

# Modify your homepage section to include the Llama-based matching system
if st.session_state.current_page == 'home':
    st.header("Business Matching Assistant")
    
    # Initialize Llama
    llm = initialize_llama()
    if not llm:
        st.error("Failed to initialize Llama model. Please check your model configuration.")
    else:
        # Create input form for business parameters
        with st.form("business_parameters"):
            st.subheader("Enter Business Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                industry = st.selectbox("Industry", [
                    "Manufacturing", "Technology", "Healthcare", "Retail", 
                    "Services", "Construction", "Other"
                ])
                revenue = st.number_input("Annual Revenue ($M)", min_value=0.0)
                ebitda = st.number_input("EBITDA ($M)", min_value=0.0)
            
            with col2:
                employees = st.number_input("Number of Employees", min_value=0)
                location = st.text_input("Location")
                growth_rate = st.slider("Growth Rate (%)", -20, 100, 0)
            
            additional_notes = st.text_area("Additional Notes", height=100)
            
            submitted = st.form_submit_button("Find Matches")
        
        if submitted:
            # Create parameters dictionary
            parameters = {
                "industry": industry,
                "revenue_millions": revenue,
                "ebitda_millions": ebitda,
                "employees": employees,
                "location": location,
                "growth_rate_percentage": growth_rate,
                "additional_notes": additional_notes
            }
            
            # Get list of one-pagers from storage
            bucket_name = get_bucket_name()
            files = list_bucket_files(bucket_name) if bucket_name else []
            
            if files:
                with st.spinner("Analyzing matches..."):
                    # Create a placeholder for the one-pagers data
                    # You'll need to implement PDF text extraction and parsing
                    one_pagers = []
                    
                    for file in files:
                        # Download and process each one-pager
                        content = download_blob(bucket_name, file)
                        if content:
                            # Parse PDF content (you'll need to implement this)
                            parsed_content = {
                                "file_name": file,
                                "industry": "Sample Industry",
                                "revenue": "Sample Revenue",
                                "ebitda": "Sample EBITDA",
                                # Add more fields as needed
                            }
                            one_pagers.append(parsed_content)
                    
                    # Get matches using Llama
                    matches = match_business_parameters(llm, parameters, one_pagers)
                    
                    # Display results in a nice format
                    st.success("Analysis complete!")
                    
                    # Create tabs for different views
                    tab1, tab2 = st.tabs(["Matches", "Raw Analysis"])
                    
                    with tab1:
                        st.markdown("### Top Matches")
                        st.markdown(matches)
                    
                    with tab2:
                        st.markdown("### Raw Analysis")
                        st.json(parameters)
                        st.json(one_pagers)
            else:
                st.error("No one-pagers found in storage.")

# Add CSS to improve the appearance
st.markdown("""
<style>
.stForm {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.stTextInput>div>div>input {
    background-color: white;
}
.stSelectbox>div>div>select {
    background-color: white;
}
.stNumberInput>div>div>input {
    background-color: white;
}
</style>
""", unsafe_allow_html=True)