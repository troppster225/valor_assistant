from setuptools import setup, find_packages

setup(
    name="business_matcher",
    version="1.0.0",
    packages=find_packages(include=['src', 'src.*']),
    install_requires=[
        "streamlit>=1.24.0",
        "google-cloud-storage>=2.10.0",
        "llama-cpp-python>=0.2.0",
        "pandas>=2.0.0",
        "langchain-community>=0.0.10",
        "langchain>=0.1.0",
        "unstructured>=0.10.0",
        "PyPDF2>=3.0.0",
        "sentence-transformers>=2.2.0",
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "google-api-core>=2.11.0",
        "textwrap3>=0.9.2",
    ],
    python_requires='>=3.8',
)