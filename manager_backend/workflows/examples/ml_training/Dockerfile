FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

COPY train_on_shard.py .

# Le volontaire monte /input (ro) et /output (rw)
ENTRYPOINT ["python", "train_on_shard.py"]
