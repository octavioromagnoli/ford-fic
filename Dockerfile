# Demo de producto (Ford DPF) para Railway: solo la app y la capa de agentes, sin el stack de modelado.
#
#   docker build -t radar-dpf .
#   docker run -p 8501:8501 -e WANDB_API_KEY=... -e OPENAI_API_KEY=... -e DEMO_PASSWORD=... radar-dpf
#
# Al arrancar baja el bundle (el wandb Artifact de `bundle.artifact` en configs/demo.yaml, hoy `demo-bundle-gru`,
# en la versión fijada ahí) y levanta Streamlit en $PORT (Railway lo define; 8501 en local).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEMO_BUNDLE_DIR=/app/bundle

WORKDIR /app
COPY requirements-demo.txt .
RUN pip install -r requirements-demo.txt

COPY src/ src/
COPY scripts/__init__.py scripts/__init__.py
COPY scripts/demo_app/ scripts/demo_app/
COPY .streamlit/ .streamlit/
COPY configs/demo.yaml configs/agents.yaml configs/

EXPOSE 8501
CMD ["sh", "-c", "python scripts/demo_app/fetch_bundle.py && exec streamlit run scripts/demo_app/app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false"]
