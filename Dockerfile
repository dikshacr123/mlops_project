# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install OS-level dependencies (optional: for streamlit + pandas + scikit-learn)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Command to run Streamlit app
CMD ["streamlit", "run", "Scripts/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
