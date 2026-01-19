FROM python:3.11-slim

# Set environment variables
ENV TZ=Europe/Brussels

# Install build dependencies for pycairo and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libcairo2-dev \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY scripts/start.sh /start.sh

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x /start.sh

ENTRYPOINT ["/start.sh"]
