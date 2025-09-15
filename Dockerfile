FROM python:3.10-slim
LABEL authors="KYDN"

WORKDIR /app

COPY requirements.txt .

# Cài đặt dependencies hệ thống + Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        unzip \
        libgl1 \
        libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && pip install --no-cache-dir -r requirements.txt 

COPY . .

ENV GDRIVE_ID=1WPbJ3N2PEWTexsqqvX_GOd6-1qUILra4


RUN gdown --id $GDRIVE_ID -O /app/models/weights.zip && \
    unzip -o /app/models/weights.zip -d /app/models/ && \
    rm /app/models/weights.zip


# Chạy Gunicorn khi container start
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:$PORT interface.app:app --workers 1 --timeout 120"]

