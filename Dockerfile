# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# git is often needed for pip installing from git repos or for other tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements (if you have them separately) or setup files first to leverage Docker cache
COPY pyproject.toml setup.py README.md ./
# If you have a requirements.txt, uncomment the next line
# COPY requirements.txt ./

# Install dependencies
# We install the package itself, but not in editable mode for production
RUN pip install --no-cache-dir --upgrade pip && \
    pip install .

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=file:///app/mlruns

# Define the entrypoint or default command
# Since this seems to be a script-based project, we can default to help or a shell
CMD ["python"]
