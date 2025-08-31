docker run -it \
  --device /dev/video0:/dev/video0 \
  -p 5000:5000 \
  --env-file .env \
  -v $(pwd)/database/serviceAccountKey.json:/app/database/serviceAccountKey.json \
  face_recognition_kydn

docker build -t face_recognition_kydn .


