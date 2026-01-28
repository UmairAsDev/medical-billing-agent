# STAGE 1: Builder (Compiles dependencies)
FROM python:3.11-slim as builder

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev

# Install dependencies into a virtual environment
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# STAGE 2: Runner (Production Image)
FROM python:3.11-slim as runner

WORKDIR /app
# Create a non-root user for security
RUN addgroup --system app && adduser --system --group app

# Copy compiled wheels from builder
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache /wheels/*

# Copy Application Code
COPY ./src /app/src
COPY ./config /app/config

# Permissions
chown -R app:app /app
USER app

# Expose Port
EXPOSE 8000

# Start Command (Uses Gunicorn for Process Management + Uvicorn Workers)
CMD ["gunicorn", "-c", "config/gunicorn_conf.py", "src.app.main:app"]