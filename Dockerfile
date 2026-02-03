
FROM python:3.11-slim AS builder

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev


COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt


FROM python:3.11-slim AS runner

WORKDIR /app

RUN addgroup --system app && adduser --system --group app


COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache /wheels/*


COPY ./src /app/src
COPY ./config /app/config


RUN chown -R app:app /app
USER app


EXPOSE 8000


CMD ["gunicorn", "-c", "config/gunicorn_conf.py", "src.app.main:app"]