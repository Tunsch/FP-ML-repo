FROM python:3.11-slim
WORKDIR /app

# 1. Schritt: Anforderungen kopieren (ändert sich fast nie) -> KOMMT AUS DEM CACHE
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt  # -> KOMMT AUS DEM CACHE

# 2. Schritt: Quellcode kopieren (ändert sich bei jedem Update)
# -> HIER merkt Docker die Änderung! Der Cache bricht ab hier ab.
COPY app.py .

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
