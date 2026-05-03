FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project (coordinator package needs to be importable from /app)
COPY coordinator/ coordinator/
COPY worker/ worker/

# Railway provides $PORT at runtime
ENV PORT=8000
EXPOSE ${PORT}

CMD ["sh", "-c", "uvicorn coordinator.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
