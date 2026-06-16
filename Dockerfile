# Imagem do app Streamlit (para Render / Railway / Fly.io / Hugging Face Spaces).
# Vercel NÃO serve (é serverless; Streamlit precisa de servidor persistente + WebSocket).
FROM python:3.12-slim

# libgomp1 é exigido pelo LightGBM em runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o projeto (o .dockerignore exclui os dados pesados não usados em runtime).
COPY . .

# A maioria dos hosts injeta $PORT; default 8501 p/ rodar local.
ENV PORT=8501
EXPOSE 8501
CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true"]
