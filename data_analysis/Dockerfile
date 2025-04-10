FROM python:3.11-slim

# Update and install essential packages including libGL for OpenCV
RUN apt-get update && apt-get install -y \ 
    build-essential \
    git \
    curl \
    wget \
    libgl1-mesa-glx \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
COPY src .
RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 8888

CMD ["streamlit", "run", "src/dashboard.py", "--server.address=0.0.0.0", "--server.port=8888", "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]

