FROM python:3.10-slim

# Set working dir
WORKDIR /app

# Install system dependencies for Pillow & Tesseract
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
 && apt-get clean

# Copy and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app + data
COPY app.py .
COPY tds_combined_data.json .

# Expose HF default port
EXPOSE 7860

# Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
