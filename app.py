import streamlit as st
# This MUST be the first Streamlit command
st.set_page_config(
    page_title="Valor Assistant", 
    page_icon="📁", 
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import sys
from pathlib import Path
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.api_core.exceptions import GoogleAPIError
from llama_cpp import Llama
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
from io import BytesIO
from PyPDF2 import PdfReader
import base64
import re
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import io
import tempfile
from src.models.industry_classification import IndustryClassification, IndustryEncoder
from src.matching.matcher import BusinessMatcher
from src.utils.similarity import (
    calculate_industry_similarity, 
    calculate_hierarchical_similarity,
    calculate_financial_similarity
)
from src.utils.pdf_processor import PDFProcessor
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def create_embedding(text: str, model) -> np.ndarray:
    return model.encode(text, convert_to_tensor=False)

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

def create_criteria_embedding(parameters: Dict) -> np.ndarray:
    """Create embedding for search criteria"""
    criteria_text = f"""
    Investment Criteria:
    Enterprise Value: ${parameters.get('enterprise_value', '')}M
    Revenue: ${parameters.get('revenue', '')}M
    EBITDA: ${parameters.get('ebitda', '')}M
    Industry: {parameters.get('industry', '')}
    Sub-industry: {parameters.get('sub_industry', '')}
    Geography: {parameters.get('geography', '')}
    Region: {parameters.get('specific_region', '')}
    Growth Rate: {parameters.get('revenue_growth', '')}%
    EBITDA Margin: {parameters.get('ebitda_margin', '')}%
    """
    
    model = load_embedding_model()
    return create_embedding(criteria_text, model)

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
                
                # Initialize PDF processor
                pdf_processor = PDFProcessor()
                
                for i, file in enumerate(st.session_state.uploaded_files):
                    try:
                        file_content = file.read()
                        
                        # Process the PDF
                        if pdf_processor.parse_and_store_pdf(file_content, file.name, bucket_name):
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
    """Get list of all parsed document names"""
    storage_client = get_storage_client()
    if not storage_client:
        return []
    try:
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix="parsed/")
        return [blob.name for blob in blobs if blob.name.endswith('.txt')]
    except Exception as e:
        st.error(f"Error listing parsed documents: {str(e)}")
        return []

def get_parsed_document(bucket_name: str, parsed_file: str) -> Optional[str]:
    """Get content of a parsed document"""
    storage_client = get_storage_client()
    if not storage_client:
        return None
    try:
        bucket = storage_client.bucket(bucket_name)
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
        
        # Add debug prints
        bucket_name = get_bucket_name()
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        print("\nChecking bucket contents:")
        all_blobs = list(bucket.list_blobs())
        print(f"Total files in bucket: {len(all_blobs)}")
        for blob in all_blobs:
            print(f"Found file: {blob.name}")
            
        parsed_blobs = list(bucket.list_blobs(prefix="parsed/"))
        print(f"\nFiles in parsed/ directory: {len(parsed_blobs)}")
        for blob in parsed_blobs:
            print(f"Found parsed file: {blob.name}")
        
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
                try:
                    # Initialize BusinessMatcher
                    matcher = BusinessMatcher()
                    
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
                    else:
                        # Create tabs for results display
                        tab1, tab2 = st.tabs(["💼 Matches", "📊 Analysis Progress"])
                        
                        with tab2:
                            progress_container = st.container()
                            status_text = st.empty()
                            progress_bar = st.progress(0)
                            metrics_col1, metrics_col2 = st.columns(2)
                            documents_processed = metrics_col1.empty()
                            avg_score = metrics_col2.empty()

                        # Use BusinessMatcher to analyze documents
                        results = matcher.match_business_parameters(llm, parameters, bucket_name)
                        
                        # Display results (keep your existing results display code)
                        if not results:
                            st.warning("No documents found in database. Please upload some documents first.")
                        else:
                            # Display results in the Matches tab
                            with tab1:
                                if results:
                                    st.subheader("🎯 Matching Opportunities")
                                    
                                    # Sort results by match score
                                    sorted_results = sorted(
                                        results.items(), 
                                        key=lambda x: x[1]['match_score'], 
                                        reverse=True
                                    )
                                    
                                    # Create metrics summary
                                    metrics_cols = st.columns(3)
                                    metrics_cols[0].metric(
                                        "Total Matches",
                                        len(results),
                                        help="Total number of opportunities analyzed"
                                    )
                                    metrics_cols[1].metric(
                                        "Best Match Score",
                                        f"{max(r['match_score'] for r in results.values()):.1%}",
                                        help="Highest matching score found"
                                    )
                                    metrics_cols[2].metric(
                                        "Average Score",
                                        f"{sum(r['match_score'] for r in results.values()) / len(results):.1%}",
                                        help="Average matching score across all opportunities"
                                    )
                                    
                                    st.divider()
                                    
                                    # Display each match
                                    for doc_name, match_data in sorted_results:
                                        # Determine emoji based on match score
                                        match_score = match_data['match_score']
                                        if match_score >= 0.75:
                                            match_emoji = "🟢"  # Green for good match
                                        elif match_score >= 0.50:
                                            match_emoji = "🟡"  # Yellow for moderate match
                                        else:
                                            match_emoji = "🔴"  # Red for poor match
                                            
                                        # Display result with colored indicator
                                        with st.expander(f"{match_emoji} {os.path.basename(doc_name)} - Match: {match_score:.1%}"):
        # Create columns for key metrics
                                            col1, col2, col3, col4 = st.columns(4)  # Changed to 4 columns
                                            
                                            # Match Score
                                            score_color = "green" if match_score >= 0.75 else "orange" if match_score >= 0.50 else "red"
                                            col1.markdown(f"""
                                                <div style='color: {score_color}'>
                                                    <h4>Match Score</h4>
                                                    <h2>{match_score:.1%}</h2>
                                                </div>
                                            """, unsafe_allow_html=True)
                                            
                                            # Industry Match
                                            industry_score = match_data['industry_similarity']
                                            industry_color = "green" if industry_score >= 0.75 else "orange" if industry_score >= 0.50 else "red"
                                            col2.markdown(f"""
                                                <div style='color: {industry_color}'>
                                                    <h4>Industry Similarity</h4>
                                                    <h2>{industry_score:.1%}</h2>
                                                </div>
                                            """, unsafe_allow_html=True)
                                            
                                            # Financial Similarity
                                            financial_score = calculate_financial_similarity(
                                                parameters,  # Your input criteria
                                                match_data['company_data']['financials']  # Company's financial data
                                            )
                                            if financial_score is not None:
                                                financial_color = "green" if financial_score >= 0.75 else "orange" if financial_score >= 0.50 else "red"
                                                financial_display = f"{financial_score:.1%}"
                                            else:
                                                financial_color = "gray"
                                                financial_display = "N/A"
                                                
                                            col3.markdown(f"""
                                                <div style='color: {financial_color}'>
                                                    <h4>Financial Similarity</h4>
                                                    <h2>{financial_display}</h2>
                                                </div>
                                            """, unsafe_allow_html=True)
                                            
                                            # Industry
                                            col4.markdown(f"""
                                                <h4>Industry</h4>
                                                <h2>{match_data['company_data']['industry']}</h2>
                                            """, unsafe_allow_html=True)
                                            
                                            # Detailed company information
                                            st.markdown("### Firm Criteria")
                                            company_data = match_data['company_data']
                                            
                                            # Create two columns for company details
                                            detail_col1, detail_col2 = st.columns(2)
                                            
                                            with detail_col1:
                                                st.markdown("**Financial Metrics:**")
                                                financials = company_data['financials']
                                                st.write(f"• Revenue: ${financials['revenue']}M")
                                                st.write(f"• EBITDA: ${financials['ebitda']}M")
                                                st.write(f"• Enterprise Value: ${financials['enterprise_value']}M")
                                            
                                            with detail_col2:
                                                st.markdown("**Location & Market:**")
                                                geography = company_data['geography']
                                                st.write(f"• Primary Region: {geography['primary_region']}")
                                                if geography['countries']:
                                                    st.write(f"• Countries: {', '.join(geography['countries'])}")
                                                if geography['specific_regions']:
                                                    st.write(f"• Specific Regions: {', '.join(geography['specific_regions'])}")
                                            
                                            st.markdown("### Document Content")

                                            doc_base_name = os.path.splitext(os.path.basename(doc_name))[0]  # Remove any extension
                                            pdf_path = f"pdfs/{doc_base_name}.pdf"
                                            try:
                                                storage_client = get_storage_client()
                                                if storage_client:
                                                    bucket = storage_client.bucket(bucket_name)
                                                    blob = bucket.blob(pdf_path)
                                                    
                                                    if blob.exists():
                                                        pdf_content = blob.download_as_bytes()
                                                        
                                                        # Add download button for PDF
                                                        st.download_button(
                                                            label="📥 Download PDF",
                                                            data=pdf_content,
                                                            file_name=os.path.basename(doc_name),
                                                            mime="application/pdf",
                                                            key=f"download_pdf_{doc_name}"
                                                        )
                                                        
                                                        # Create a base64 encoded version of the PDF for display
                                                        base64_pdf = base64.b64encode(pdf_content).decode('utf-8')
                                                        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" type="application/pdf"></iframe>'
                                                        st.markdown(pdf_display, unsafe_allow_html=True)
                                                        
                                                    else:
                                                        st.error(f"PDF not found: {pdf_path}")
                                            except Exception as e:
                                                st.error(f"Error loading PDF: {str(e)}")

                                            st.divider()
                                else:
                                    st.warning("No matching opportunities found. Try adjusting your criteria.")

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