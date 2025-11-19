# --- Base image ---
FROM python:3.12-slim

# Prevent buffer issues
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install OS dependencies (for psycopg2 + sentence-transformers)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port (FastAPI will run here)
EXPOSE 8000

# Start your main app
CMD ["python", "main.py"]
