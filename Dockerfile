FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY src/ src/
COPY models/model.skops models/model_info.json models/
RUN useradd --create-home appuser && chmod -R a-w /app/models
USER appuser
ARG BUILD_SHA=unknown
ENV SERVICE_GIT_COMMIT=${BUILD_SHA}
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','8000')+'/health',timeout=3)"
CMD ["sh", "-c", "exec uvicorn src.api.main:app --host 0.0.0.0 --port \"${PORT:-8000}\""]
