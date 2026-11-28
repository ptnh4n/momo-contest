FROM python:3.12-slim AS backend
WORKDIR /srv/safepay
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 SAFEPAY_DB_PATH=/data/safepay.sqlite
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/app app
RUN mkdir -p /data
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.wallet.api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM node:24-bookworm-slim AS frontend
WORKDIR /srv/safepay
ENV NEXT_TELEMETRY_DISABLED=1
ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL}
COPY frontend/package*.json ./
RUN npm ci --no-audit --no-fund
COPY frontend .
RUN npm run build
EXPOSE 3000
CMD ["npm", "run", "start", "--", "--hostname", "0.0.0.0"]
