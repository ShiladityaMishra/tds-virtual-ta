# Use official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create static directory if it doesn't exist
RUN mkdir -p /app/static

# Copy the entire app
COPY . .

# Expose FastAPI port
EXPOSE 7860

# Start the FastAPI app (change to app.py or main.py as needed)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
