# Riva Speech Recognition MCP Server Dockerfile
# Multi-stage build for optimized production image

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files and source code
COPY pyproject.toml ./
COPY src/ ./src/

# Create virtual environment and install Python dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/venv/bin:$PATH"

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Create non-root user for security
RUN groupadd --gid 1000 riva && \
    useradd --uid 1000 --gid riva --shell /bin/bash --create-home riva

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    # Audio processing libraries (for audio file handling)
    libsndfile1 \
    # Network tools for debugging and health checks
    curl \
    netcat-openbsd \
    # gRPC dependencies for NVIDIA Riva client
    libssl-dev \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy application code and configuration files
COPY --chown=riva:riva src/ ./src/
COPY --chown=riva:riva pyproject.toml ./
COPY --chown=riva:riva run_server.py ./
COPY --chown=riva:riva config.example ./

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data && \
    chown -R riva:riva /app

# Switch to non-root user
USER riva

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD sh -c 'curl -f http://localhost:$RIVA_PORT/health || exit 1'

# Default environment variables
ENV RIVA_HOST=0.0.0.0
ENV RIVA_PORT=8080
ENV RIVA_LOG_LEVEL=INFO
ENV RIVA_URI=localhost:50051
ENV RIVA_LANGUAGE_CODE=en-US
ENV RIVA_ASR_MODE=offline
ENV RIVA_MAX_ALTERNATIVES=3
ENV RIVA_ENABLE_PUNCTUATION=true
ENV RIVA_VERBATIM_TRANSCRIPTS=false

# Default command
CMD ["sh", "-c", "python -m src.riva_speech_recognition_mcp.server --host $RIVA_HOST --port $RIVA_PORT"]