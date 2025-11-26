FROM minizinc/minizinc:2.8.5

# Avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        curl \
        python3.12 \
        python3-pip \
        wget \
        git \
        nano \
        bash \
        build-essential \
        glpk-utils \
        coinor-cbc \
        z3 \
        libgmp-dev \
        dos2unix && \
    \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Install Optimathsat
RUN wget https://optimathsat.disi.unitn.it/releases/optimathsat-1.7.3/optimathsat-1.7.3-linux-64-bit.tar.gz && \
    tar xzf optimathsat-1.7.3-linux-64-bit.tar.gz && \
    mv optimathsat-1.7.3-linux-64-bit /opt/optimathsat && \
    rm optimathsat-1.7.3-linux-64-bit.tar.gz

RUN pip install wrapt --upgrade --ignore-installed
RUN pip install gdown
    
# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir -U git+https://github.com/coin-or/pulp

COPY . .

# Back to default frontend
ENV DEBIAN_FRONTEND=dialog
ENV PATH="/opt/optimathsat/bin:${PATH}"

# Default command
CMD ["/bin/bash"]