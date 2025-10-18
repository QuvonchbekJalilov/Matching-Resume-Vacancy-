# ============================
# 1️⃣ Base image — PyTorch (CPU/GPU auto)
# ============================
FROM pytorch/pytorch:2.5.1

# ============================
# 2️⃣ Environment setup
# ============================
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/app/cache \
    MODEL_NAME=BAAI/bge-m3

# ============================
# 3️⃣ Install dependencies
# ============================
RUN apt-get update && apt-get install -y --no-install-recommends \
    git nginx python3-dev build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ============================
# 4️⃣ Copy app files
# ============================
COPY . /python

# ============================
# 5️⃣ Expose FastAPI port
# ============================
EXPOSE 8000

# ============================
# 6️⃣ Run app via Gunicorn + Uvicorn worker
# ============================
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "main:app"]
