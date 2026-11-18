# THROWAWAY benchmark service (see bench_service.py). Context: backend/. Built by Render from render.bench.yaml; delete it afterwards.
FROM python:3.13-slim AS model
RUN pip install --no-cache-dir "huggingface_hub>=0.30"
COPY scripts/fetch_model.py /tmp/fetch_model.py
RUN python /tmp/fetch_model.py /model/minilm
# DeBERTa is NOT served; it is downloaded only so it can be measured on request (/run_deberta). Unpinned: throwaway.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('namya06/maskwall-injection-deberta', local_dir='/model/deberta', allow_patterns=['model.int8.onnx','tokenizer.json','model_meta.json'])" && rm -rf /model/deberta/.cache

FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY requirements.lock.txt .
RUN pip install -r requirements.lock.txt
COPY app ./app
COPY bench ./bench
COPY --from=model /model ./model
RUN chown -R user:user /home/user
USER user
ENV MASKWALL_MODEL_DIR=/home/user/app/model
CMD ["python", "bench/bench_service.py"]
