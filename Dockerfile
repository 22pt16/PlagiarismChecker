# Use lightweight python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Run Streamlit app
EXPOSE 8501
CMD ["streamlit", "run", "app/interface.py", "--server.port=8501", "--server.address=0.0.0.0"]
