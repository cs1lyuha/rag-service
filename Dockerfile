FROM python:3.12-slim

RUN useradd --create-home --uid 10001 rag
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir fastapi numpy uvicorn
COPY ragsvc ./ragsvc
COPY kb ./kb

USER rag
ENV RAG_KB_DIR=/app/kb PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
CMD ["uvicorn", "ragsvc.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
