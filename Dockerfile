FROM python:3.12-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY data/raw/wesad/model.pkl ./data/raw/wesad/
COPY data/raw/wesad/scaler.pkl ./data/raw/wesad/
COPY data/raw/wesad/feature_schema.json ./data/raw/wesad/

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
