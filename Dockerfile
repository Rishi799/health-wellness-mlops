FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY src/ src/
COPY models/model.skops models/model_info.json models/
RUN useradd --create-home appuser && chmod -R a-w /app/models
# Create /app/logs owned by appuser so the prediction logger can write even
# when the directory is bind-mounted from the host (the entrypoint below
# handles the bind-mount case at runtime).
RUN mkdir -p /app/logs && chown appuser:appuser /app/logs
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
USER appuser
ARG BUILD_SHA=unknown
ENV SERVICE_GIT_COMMIT=${BUILD_SHA}
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','8000')+'/health',timeout=3)"
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
