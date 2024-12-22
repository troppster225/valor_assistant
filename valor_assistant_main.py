import streamlit as st
import os
from io import BytesIO
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.api_core.exceptions import GoogleAPIError
from llama_cpp import Llama
import pandas as pd
from typing import Dict, List
import json
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
import tempfile
from PyPDF2 import PdfReader
import textwrap
import re
import traceback
from typing import Dict
import time
# Page configuration
st.set_page_config(page_title="Valor Assistant", 
                   page_icon="📁", 
                   layout="wide",
                   initial_sidebar_state="expanded"
                   )

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

def parse_and_store_pdf(file_content: bytes, filename: str, bucket_name: str) -> bool:
    """
    Parse PDF content and store both raw PDF and parsed text in GCS.
    Returns True if successful, False otherwise.
    """
    try:
        storage_client = get_storage_client()
        bucket = storage_client.bucket(bucket_name)
        
        # Store original PDF
        pdf_blob = bucket.blob(f"pdfs/{filename}")
        pdf_blob.upload_from_string(file_content)
        
        # Parse PDF content
        pdf_file = BytesIO(file_content)
        pdf_reader = PdfReader(pdf_file)
        
        full_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n\n"  # Add page breaks for clarity
        
        # Store parsed text
        text_filename = f"parsed/{os.path.splitext(filename)[0]}.txt"
        text_blob = bucket.blob(text_filename)
        text_blob.upload_from_string(full_text)
        
        return True
        
    except Exception as e:
        st.error(f"Error processing {filename}: {str(e)}")
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

if st.session_state.current_page == 'upload':
    if not st.session_state.upload_complete:
        uploaded_files = st.file_uploader("Upload PDF", type=list(ALLOWED_EXTENSIONS), accept_multiple_files=True)

        if uploaded_files:
            new_files = [file for file in uploaded_files if file not in st.session_state.uploaded_files]
            valid_files = [file for file in new_files if is_valid_file(file.name)]
            
            if valid_files:
                st.session_state.uploaded_files.extend(valid_files)
                st.write(f"{len(valid_files)} new valid files added.")

        if st.session_state.uploaded_files and st.button("Submit"):
            bucket_name = get_bucket_name()
            if bucket_name:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                successful_uploads = 0
                failed_uploads = 0
                
                for i, file in enumerate(st.session_state.uploaded_files):
                    try:
                        file_content = file.read()
                        # Parse and store both PDF and text content
                        if parse_and_store_pdf(file_content, file.name, bucket_name):
                            successful_uploads += 1
                            st.session_state.upload_summary.append(f"Successfully processed: {file.name}")
                        else:
                            failed_uploads += 1
                            st.session_state.upload_summary.append(f"Failed to process: {file.name}")
                            
                        progress = (i + 1) / len(st.session_state.uploaded_files)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing... {i+1}/{len(st.session_state.uploaded_files)}")
                        
                    except Exception as e:
                        failed_uploads += 1
                        st.session_state.upload_summary.append(f"Error processing {file.name}: {str(e)}")
                
                st.session_state.upload_complete = True
                if failed_uploads > 0:
                    st.warning(f"{successful_uploads} files processed successfully. {failed_uploads} files failed.")
                else:
                    st.success(f"All {successful_uploads} files processed successfully!")

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
        model_path = "/Users/tommyropp/Desktop/Valor_BD_Project/ValorAssistant/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf"
        
        llm = Llama(
            model_path=model_path,
            n_ctx=3000,  # Reduced context window for faster processing
            n_gpu_layers=-1,
            n_threads=12,  # Increased threads
            n_batch=512,
            f16_kv=True,
            verbose=False
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize Llama model: {str(e)}")
        return None

def get_parsed_documents(bucket_name: str) -> List[str]:
    """
    Get list of all parsed document names from the 'parsed/' directory.
    Returns a list of filenames.
    """
    try:
        storage_client = get_storage_client()
        bucket = storage_client.bucket(bucket_name)
        # Only list blobs in the parsed/ directory
        blobs = bucket.list_blobs(prefix="parsed/")
        return [blob.name for blob in blobs if blob.name.endswith('.txt')]
    except Exception as e:
        st.error(f"Error listing parsed documents: {str(e)}")
        return []

def get_parsed_document(bucket_name: str, filename: str) -> str:
    """
    Retrieve pre-parsed document text content from storage.
    Returns the actual text content of a single document.
    """
    try:
        storage_client = get_storage_client()
        bucket = storage_client.bucket(bucket_name)
        text_filename = f"parsed/{os.path.splitext(filename)[0]}.txt"
        blob = bucket.blob(text_filename)
        return blob.download_as_text()
    except Exception as e:
        st.error(f"Error retrieving parsed text for {filename}: {str(e)}")
        return None

def analyze_document(llm: Llama, document_text: str, parameters: Dict) -> str:
    """
    Analyze document text with partial matching ranges for all criteria.
    """
    try:
        prompt = f""" You are a deal screening analyst. You will be given data on investment criteria
        that different private equity firms have. The user has input information on a company that is for sale.
        Your job is to score how well the inputted information matches the criteria from the private equity
        firms. Return ONLY valid JSON with the following format:

{{
"score": <total points from 0-100>,
"matching_points": [
    "FINANCE: [exact values] - [match type]",
    "INDUSTRY: [specific description] - [match type]",
    "GEO: [specific regions] - [match type]"
],
"concerns": [
    "specific gaps or mismatches in provided criteria"
],
"summary": "brief analysis focusing on match quality"
}}

CRITERIA PROVIDED:
• Enterprise Value Target: ${parameters['enterprise_value']}M
• Revenue Target: ${parameters['revenue']}M
• EBITDA Target: ${parameters['ebitda']}M
• Industry Focus: {parameters['industry']}
• Geographic Focus: {parameters['geography']}

SCORING RULES (Total score must be between 0-100):
Each criterion is worth up to 33.33 points:

Financial Metrics (Combined worth 33.33 points):
• Within ±20%: 33.33 points (Full match)
• Within ±35%: 25 points (Strong match)
• Within ±50%: 15 points (Partial match)
• Outside ranges: 5 points
• No data: 0 points

Industry Match (worth 33.33 points):
• Direct match: 33.33 points
• Closely related: 25 points
• Partial overlap: 15 points
• Minor overlap: 5 points
• No match: 0 points

Geography Match (worth 33.33 points):
• Direct match: 33.33 points
• Major overlap: 25 points
• Partial overlap: 15 points
• Minor overlap: 5 points
• No match: 0 points

IMPORTANT:
- Total score MUST be between 0 and 100
- Calculate percentage differences for financials
- Strong matches should result in high scores (80+)
- Moderate matches should be around 60-79
- Weak matches should be below 60

DOCUMENT TO ANALYZE:
{document_text}

Return ONLY valid JSON with scores normalized to 100-point scale.
"""

        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a deal screening analyst. Score matches fairly based on actual alignment with criteria. Strong matches should receive high scores."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0
        )
        
        if not response or "choices" not in response or not response["choices"]:
            return """Match Score: 0/100
Key Points: LLM processing error
Concerns: Failed to get response
Summary: Unable to complete analysis"""

        content = response["choices"][0]["message"]["content"].strip()
        
        # Clean and parse JSON
        content = content.replace('```json', '').replace('```', '').strip()
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        
        if start_idx == -1 or end_idx == -1:
            return """Match Score: 0/100
Key Points: Invalid format
Concerns: No JSON found
Summary: Failed to parse response"""
            
        json_str = content[start_idx:end_idx + 1]
        analysis = json.loads(json_str)
        
        # Ensure score is between 0-100
        raw_score = int(analysis.get('score', 0))
        adjusted_score = max(0, min(100, raw_score))  # Clamp between 0-100
        
        matching_points = analysis.get('matching_points', [])
        concerns = analysis.get('concerns', [])
        summary = analysis.get('summary', 'No summary provided')
        
        if isinstance(matching_points, str):
            matching_points = [matching_points]
        if isinstance(concerns, str):
            concerns = [concerns]
        
        # Clean up formatting
        matching_points = [point.replace('+', ' plus ').replace('-', ' - ').replace('  ', ' ') for point in matching_points]
        concerns = [concern.replace('+', ' plus ').replace('-', ' - ').replace('  ', ' ') for concern in concerns]
            
        return f"""Match Score: {adjusted_score}/100

Key Matches:
{chr(10).join('• ' + point for point in matching_points) if matching_points else '• None identified'}

Concerns:
{chr(10).join('• ' + concern for concern in concerns) if concerns else '• None identified'}

Summary:
{summary}"""
            
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        return """Match Score: 0/100
Key Points: Error occurred
Concerns: Analysis failed
Summary: Unable to process document"""

def match_business_parameters(llm: Llama, parameters: Dict, bucket_name: str) -> Dict:
    results = {}
    error_log = []
    
    parsed_files = get_parsed_documents(bucket_name)
    if not parsed_files:
        return {}
        
    with st.spinner("Analyzing documents..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        documents_processed = st.empty()
        avg_score = st.empty()
        scores = []
        
        for idx, parsed_file in enumerate(parsed_files):
            try:
                status_text.text(f"Analyzing {parsed_file}...")
                document_text = get_parsed_document(bucket_name, parsed_file)
                
                if document_text:
                    display_name = os.path.basename(parsed_file).replace('.txt', '.pdf')
                    
                    print(f"\n=== Processing document: {display_name} ===")
                    result = analyze_document(llm, document_text, parameters)
                    results[display_name] = result
                    
                    # Extract score
                    try:
                        score_line = result.split('\n')[0]
                        score = int(score_line.split(':')[1].strip().replace('/100', ''))
                        scores.append(score)
                    except (ValueError, IndexError) as e:
                        error_log.append(f"Score extraction failed for {display_name}: {str(e)}")
                    
                    # Update progress
                    progress = (idx + 1) / len(parsed_files)
                    progress_bar.progress(progress)
                    documents_processed.metric("Documents Processed", f"{idx + 1}/{len(parsed_files)}")
                    if scores:
                        avg_score.metric("Average Match Score", f"{sum(scores)/len(scores):.1f}%")
                
            except Exception as e:
                display_name = os.path.basename(parsed_file).replace('.txt', '.pdf')
                error_msg = f"Error analyzing {display_name}: {str(e)}"
                error_log.append(error_msg)
                results[display_name] = error_msg
                
        progress_bar.empty()
        status_text.empty()
        
        # Display error log if there were any errors
        if error_log:
            print("\n=== Error Log ===")
            for error in error_log:
                print(error)
            print("================\n")
    
    return results

def get_parsed_document(bucket_name: str, parsed_file: str) -> str:
    """
    Retrieve pre-parsed document text content from storage.
    Returns the actual text content of a single document.
    """
    try:
        storage_client = get_storage_client()
        bucket = storage_client.bucket(bucket_name)
        # Use the parsed file path directly
        blob = bucket.blob(parsed_file)
        return blob.download_as_text()
    except Exception as e:
        st.error(f"Error retrieving parsed text for {parsed_file}: {str(e)}")
        return None

if st.session_state.current_page == 'home':
    st.title("Private Equity Deal Matching Assistant")
    
    st.markdown("""
    ### Deal Screening Tool
    This tool matches potential investment opportunities against your deal criteria by analyzing our database of company profiles and deal memos.
    
    #### Key Criteria Analyzed:
    - Company financials and metrics
    - Industry and sector focus
    - Geographic preferences
    - Deal size and structure
    - Growth potential and market position
    """)
    
    st.divider()
    
    with st.form("deal_criteria"):
        # Deal Size & Company Metrics Section
        st.subheader("1. Deal Size & Company Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            enterprise_value = st.number_input(
                "Enterprise Value Range ($M)",
                min_value=0.0,
                help="Target enterprise value in millions of dollars"
            )
            
            revenue = st.number_input(
                "Annual Revenue ($M)",
                min_value=0.0,
                help="Last twelve months (LTM) revenue in millions"
            )
        
        with col2:
            ebitda = st.number_input(
                "EBITDA ($M)",
                min_value=0.0,
                help="Last twelve months (LTM) EBITDA in millions"
            )
            
            ebitda_margin = st.slider(
                "EBITDA Margin (%)",
                min_value=0,
                max_value=100,
                value=15,
                help="EBITDA as a percentage of revenue"
            )

        # Industry & Geography Section
        st.subheader("2. Industry & Market Focus")
        col3, col4 = st.columns(2)
        
        with col3:
            industry = st.selectbox(
                "Target Industry",
                [
                    "Select primary industry...",
                    "Technology & Software",
                    "Healthcare & Life Sciences",
                    "Industrial & Manufacturing",
                    "Consumer & Retail",
                    "Business Services",
                    "Financial Services",
                    "Energy & Resources",
                    "Media & Telecommunications",
                    "Other"
                ],
                help="Select the primary industry focus"
            )
            
            sub_industry = st.text_input(
                "Sub-Industry/Sector",
                placeholder="e.g., SaaS, Healthcare Services, etc.",
                help="Specific sector or sub-industry focus"
            )
        
        with col4:
            geography = st.selectbox(
                "Geographic Focus",
                [
                    "Select region...",
                    "North America",
                    "Europe",
                    "Asia Pacific",
                    "Latin America",
                    "Global"
                ],
                help="Primary geographic focus"
            )
            
            specific_region = st.text_input(
                "Specific Region/Market",
                placeholder="e.g., Northeast US, Western Europe, etc.",
                help="Target specific regions or markets"
            )

        # Growth & Operations Section
        st.subheader("3. Growth & Operational Metrics")
        col5, col6 = st.columns(2)
        
        with col5:
            revenue_growth = st.slider(
                "Revenue Growth Rate (%)",
                min_value=-20,
                max_value=200,
                value=10,
                help="Historical annual revenue growth rate"
            )
            
            employee_count = st.number_input(
                "Employee Count",
                min_value=0,
                help="Current number of full-time employees"
            )
        
        with col6:
            gross_margin = st.slider(
                "Gross Margin (%)",
                min_value=0,
                max_value=100,
                value=40,
                help="Gross profit as a percentage of revenue"
            )
            
            recurring_revenue = st.slider(
                "Recurring Revenue (%)",
                min_value=0,
                max_value=100,
                value=0,
                help="Percentage of revenue that is recurring"
            )

        # Additional Criteria Section
        st.subheader("4. Additional Investment Criteria")
        additional_notes = st.text_area(
            "Additional Requirements",
            placeholder="""Specify any additional criteria such as:
- Minimum cash flow requirements
- Customer concentration limits
- Management team preferences
- Preferred deal structure
- Industry-specific metrics
- Exit strategy considerations""",
            height=100,
            help="Include any specific requirements or preferences not covered above"
        )
        
        # Submit section
        st.divider()
        col7, col8 = st.columns([3, 1])
        with col8:
            submitted = st.form_submit_button("🔍 Screen Opportunities", use_container_width=True)
        with col7:
            st.markdown("*Click to analyze and find matching opportunities in our database*")

    if submitted:
        proceed_with_analysis = True
        
        if industry == "Select primary industry..." or geography == "Select region...":
            st.error("Please select both an industry and geographic region.")
            proceed_with_analysis = False
            
        if proceed_with_analysis:
            # Initialize Llama
            llm = initialize_llama()
            if not llm:
                st.error("Failed to initialize Llama model. Please check your model configuration.")
                proceed_with_analysis = False
            
            if proceed_with_analysis:
                # Prepare parameters dictionary
                parameters = {
                    "enterprise_value": enterprise_value,
                    "revenue": revenue,
                    "ebitda": ebitda,
                    "ebitda_margin": ebitda_margin,
                    "industry": industry,
                    "sub_industry": sub_industry,
                    "geography": geography,
                    "specific_region": specific_region,
                    "revenue_growth": revenue_growth,
                    "employee_count": employee_count,
                    "gross_margin": gross_margin,
                    "recurring_revenue": recurring_revenue,
                    "additional_notes": additional_notes
                }

                bucket_name = get_bucket_name()
                if not bucket_name:
                    st.error("Unable to access document storage.")
                    proceed_with_analysis = False
                
                if proceed_with_analysis:
                    # Create tabs first so we can update them during analysis
                    tab1, tab2 = st.tabs(["💼 Matches", "📊 Analysis Progress"])
                    
                    with tab2:
                        progress_container = st.container()
                        status_text = st.empty()
                        progress_bar = st.progress(0)
                        metrics_col1, metrics_col2 = st.columns(2)
                        documents_processed = metrics_col1.empty()
                        avg_score = metrics_col2.empty()

                    try:
                        # Use the match_business_parameters function to analyze documents
                        results = match_business_parameters(llm, parameters, bucket_name)
                        
                        if not results:
                            st.warning("No documents found in database. Please upload some documents first.")
                        else:
                            # Sort results by match score
                            sorted_results = dict(sorted(
                                results.items(),
                                key=lambda x: float(x[1].split('/100')[0].split(':')[-1].strip() if '/100' in x[1] else 0),
                                reverse=True
                            ))

                            # Extract scores for summary metrics
                            scores = []
                            for analysis in results.values():
                                try:
                                    score = int(analysis.split('/100')[0].split(':')[-1].strip())
                                    scores.append(score)
                                except:
                                    pass

                            # Display final results
                            with tab1:
                                st.success("Analysis complete!")
                                
                                # Display top matches
                                st.markdown("### Top Matching Opportunities")
                                for filename, analysis in sorted_results.items():
                                    try:
                                        score = float(analysis.split('/100')[0].split(':')[-1].strip())
                                        score_color = (
                                            "🟢" if score >= 80 else
                                            "🟡" if score >= 60 else
                                            "🔴"
                                        )
                                        with st.expander(f"{score_color} {filename} (Match: {score:.1f}%)"):
                                            st.markdown(analysis)
                                    except:
                                        with st.expander(f"⚠️ {filename}"):
                                            st.markdown(analysis)
                                
                                # Display analysis summary
                                st.markdown("### Analysis Summary")
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Documents Analyzed", len(results))
                                if scores:
                                    col2.metric("Average Match Score", f"{sum(scores)/len(scores):.1f}%")
                                    col3.metric("Top Match Score", f"{max(scores):.1f}%")
                                
                                # Display search parameters
                                with st.expander("🎯 Search Parameters"):
                                    st.json(parameters)

                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")

st.markdown("""
<style>
    /* Form background and text colors */
    .stForm {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #0f1116;
    }
    
    /* Input fields styling */
    .stTextInput>div>div>input, 
    .stSelectbox>div>div>select, 
    .stNumberInput>div>div>input {
        background-color: white;
        color: #0f1116;
    }
    
    /* Text area styling */
    .stTextArea>div>div>textarea {
        background-color: white;
        color: #0f1116;
    }
    
    /* Labels and headers */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        color: #0f1116;
    }
    
    /* General text color */
    .stMarkdown {
        color: white;
    }
    
    /* Help text styling */
    .stMarkdown small {
        color: #white;
    }
    
    /* Slider text */
    .stSlider {
        color: #0f1116;
    }
    
    /* Number input text */
    .stNumberInput {
        color: #0f1116;
    }
    
    /* Selectbox text */
    .stSelectbox {
        color: black;
    }
    
    /* Section headers */
    .main .block-container {
        color: #0f1116;
    }
    
    /* Make dark theme text visible */
    @media (prefers-color-scheme: dark) {
        .stTextInput>div>div>input,
        .stSelectbox>div>div>select,
        .stNumberInput>div>div>input,
        .stTextArea>div>div>textarea {
            color: black;
        }
    }
</style>
""", unsafe_allow_html=True)
