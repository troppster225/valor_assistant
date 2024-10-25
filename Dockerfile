# Use an official Python runtime as a parent image
FROM python:3.10-slim

RUN pip install --upgrade pip

# Set the working directory in the container
WORKDIR /opt

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose port 80 to the outside world
EXPOSE 80

# Create the .streamlit directory
RUN mkdir -p ~/.streamlit

# Copy the Streamlit configuration and credentials files into the .streamlit directory
COPY config.toml ~/.streamlit/config.toml
COPY credentials.toml ~/.streamlit/credentials.toml

# Set the default command to run the Streamlit app
ENTRYPOINT ["streamlit", "run"]

# Command to run the Streamlit app
CMD ["valor_assistant_main.py"]
