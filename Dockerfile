FROM ubuntu:22.04

# Avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /sts_solver

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        curl \
        python3.10 \
        python3-pip \
        wget \
        git \
        nano \
        bash \
        build-essential \
        glpk-utils \
        coinor-cbc && \
    
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install wrapt --upgrade --ignore-installed
RUN pip install gdown

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir -U git+https://github.com/coin-or/pulp

# Default command
CMD ["/bin/bash"]

# Back to default frontend
ENV DEBIAN_FRONTEND=dialog