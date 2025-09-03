FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user with UID 1000 (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user

# Set up user environment
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8

# Set working directory
WORKDIR $HOME/app

# Copy requirements first to leverage Docker cache
COPY --chown=user requirements.txt .

# Install Python dependencies
RUN pip install --user --no-cache-dir --upgrade pip && \
    pip install --user --no-cache-dir -r requirements.txt && \
    pip install --user --no-cache-dir --upgrade gradio>=4.44.1

# Pre-download the sentence transformer model to speed up startup
RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('thenlper/gte-large'); print('Model downloaded successfully')"

# Copy the application code
COPY --chown=user . .

# Create necessary directories with proper permissions
RUN mkdir -p $HOME/app/temp && \
    chmod -R 755 $HOME/app

# Expose port 7860 (required by Hugging Face Spaces)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run the Gradio application
CMD ["python", "app.py"]