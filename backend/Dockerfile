# MaskWall firewall API.
# Build:  docker build -t maskwall-api backend        (context = backend/; needs backend/model/<name>/ from scripts/prepare_model.py)
# Run:    docker run -p 7860:7860 --env-file .env maskwall-api
# Port 7860 and the non-root uid 1000 are what Hugging Face Docker Spaces expect. API keys come from the environment at run
# time (never baked into the image). On the Space the model folder will be fetched from the Hugging Face Hub (Day 4).
# NOTE: written without Docker available on the development machine, so this file has not been built or run yet.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY model ./model
RUN mkdir -p /home/user/data && chown -R user:user /home/user
USER user

ENV MASKWALL_MODEL_DIR=/home/user/app/model \
    MASKWALL_DB_PATH=/home/user/data/maskwall.sqlite \
    MASKWALL_SERVER_MODEL=minilm

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "7860"]
