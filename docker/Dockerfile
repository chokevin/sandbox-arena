FROM python:3.12-slim

WORKDIR /app

# Install Node.js for Pokemon Showdown
RUN apt-get update -qq && \
    apt-get install -y -qq curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y -qq nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Pokemon Showdown
RUN npm install -g pokemon-showdown

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy opponent implementations
COPY opponents/ /app/opponents/

EXPOSE 8888

# Start the showdown server and keep the container alive
CMD ["sh", "-c", "pokemon-showdown start --no-security --port 8888 & sleep infinity"]
