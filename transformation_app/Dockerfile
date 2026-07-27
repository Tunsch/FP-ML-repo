FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Absicherung gegen CPU-spezifische Segfaults in libarrow.so (SIMD/AVX-
# Dispatch-Probleme auf manchen CPUs/Virtualisierungsumgebungen) -- erzwingt
# den portablen Ausführungspfad ohne CPU-Feature-Erkennung.
ENV ARROW_USE_SIMD=0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY core.py cli.py app.py .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
