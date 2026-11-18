# MaskWall firewall API (Hugging Face Docker Space; also runs anywhere with Docker).
# Build:  docker build -t maskwall-api backend        (context = backend/; no local model needed, it is downloaded from the Hub)
# Run:    docker run -p 7860:7860 --env-file .env maskwall-api
#
# Port 7860 and the non-root uid 1000 are what Hugging Face Docker Spaces expect. API keys come from the environment at run time
# (Space secrets); they are never baked into the image.
#
# The model is fetched at BUILD time from the public Hub repo, pinned to a commit (MODEL_REVISION), so a cold start needs no
# network and the image is reproducible. To use a newer model: upload it, then change MODEL_REVISION.
# NOTE: this file has never been built on the development machine (no Docker there); CI builds it on every push and the Space
# build log is the real test.

# ---- stage 1: download the model files (huggingface_hub stays out of the final image) ----
FROM python:3.13-slim AS model
ARG MODEL_REPO=namya06/maskwall-injection-minilm
ARG MODEL_REVISION=78085d047839aef62e6bc4bad01dad4b4e9cf370
RUN pip install --no-cache-dir "huggingface_hub>=0.30"
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_REPO}', revision='${MODEL_REVISION}', local_dir='/model/minilm', allow_patterns=['model.int8.onnx','tokenizer.json','model_meta.json'])" \
 && ls -l /model/minilm && rm -rf /model/minilm/.cache

# ---- stage 2: the app ----
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# exact tested versions (requirements.lock.txt); requirements.txt only holds the minimums
COPY requirements.lock.txt .
RUN pip install -r requirements.lock.txt

COPY app ./app
COPY --from=model /model ./model
RUN mkdir -p /home/user/data && chown -R user:user /home/user
USER user

ENV MASKWALL_MODEL_DIR=/home/user/app/model \
    MASKWALL_DB_PATH=/home/user/data/maskwall.sqlite \
    MASKWALL_SERVER_MODEL=minilm

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "7860"]
