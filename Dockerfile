FROM pytorch/pytorch:2.12.1-cuda12.6-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    git vim nano wget curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

CMD ["bash"]
