docker run \
  -p 5000:5000 \
  -e PORT=5000 \
  -e PYTHONUNBUFFERED=1 \
  --env-file .env \
  face_recognition_kydn
  
docker build -t face_recognition_kydn .


