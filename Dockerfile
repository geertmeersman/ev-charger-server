FROM python:3.11-slim

# Set environment variables
ENV TZ=Europe/Brussels

COPY scripts/start.sh /start.sh

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x /start.sh

ENTRYPOINT ["/start.sh"]