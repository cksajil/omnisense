FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces requires a non-root user with UID 1000
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package itself
RUN pip install --no-cache-dir -e .

# Give ownership to the non-root user
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

CMD ["python", "-m", "omnisense.app"]
