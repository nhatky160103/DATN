FROM python:3.10-slim
LABEL authors="KYDN"

WORKDIR /app

COPY requirements.txt .

# Cài đặt dependencies hệ thống + Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        libopencv-dev \
        python3-opencv \
        unzip \
        libgl1 \
        libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gdown

COPY . .

EXPOSE 5000

ENV GDRIVE_ID=1vWL4xmAyK-in6jCo0196ZbmIoAprg6ST

COPY setup_weights.sh /app/setup_weights.sh
RUN chmod +x /app/setup_weights.sh

# Khi container start → tải weights rồi khởi chạy app
CMD ["/app/setup_weights.sh", "gunicorn", "-b", "0.0.0.0:5000", "interface.app:app"]
