# CricketIQ live demo — serves the precomputed replay UI (read-only: no training, no LLM).
FROM python:3.12-slim

# libgomp1 is required by LightGBM at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# application source + the demo data subset (produced by deploy/subset_for_deploy.py)
COPY src /app/src
COPY data /app/data

# core/config.py computes ROOT as parents[3] of src/cricketiq/core/config.py -> /app,
# so PROCESSED_DIR = /app/data/processed. We run FROM SOURCE (PYTHONPATH), and deliberately
# do NOT `pip install` the package — installing it would move config.py into site-packages
# and that path would then point away from /app/data.
ENV PYTHONPATH=/app/src

EXPOSE 7860
CMD ["uvicorn", "cricketiq.serve.api:app", "--host", "0.0.0.0", "--port", "7860"]
