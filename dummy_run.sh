docker run \
  -p 5000:5000 \
  --device=/dev/video0:/dev/video0 \
  -e PORT=5000 \
  -e PYTHONUNBUFFERED=1 \
  --env-file .env \
  -v $(pwd)/database/ServiceAccountKey.json:/app/database/ServiceAccountKey.json \
  face_recognition_kydn 
  
docker build -t face_recognition_kydn .


