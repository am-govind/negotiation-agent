FROM python:3.10

# Install Node.js for frontend build
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs

WORKDIR /app
COPY . .

# Build frontend
RUN cd frontend && npm install && npm run build

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create HF-required temp directories and permissions if needed
RUN mkdir -p /app/outputs/negotiations && chmod -R 777 /app/outputs

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
