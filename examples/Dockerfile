FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir k8s-agent-sandbox

COPY arenas/ arenas/
COPY agents/ agents/
COPY run.py train.py ./

ENTRYPOINT ["python3"]
CMD ["train.py", "--arena", "trading", "--generations", "20", "--population", "20", "--mode", "cluster"]
