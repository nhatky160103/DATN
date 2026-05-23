# Future Work

## Runtime Hardening

- Add authentication and role-based access control to the Admin Dashboard.
- Persist enrollment job status in PostgreSQL.
- Add audit logs for camera, identity, and threshold changes.
- Add request validation and rate limiting to the API.
- Add backup and restore scripts for PostgreSQL and Qdrant volumes.

## Recognition Improvements

- Calibrate Qdrant thresholds on a production validation set.
- Add model versioning to Qdrant payloads and API responses.
- Support side-by-side evaluation of multiple embedding models.
- Add automatic identity quality reports after enrollment.
- Add duplicate identity detection during enrollment.

## Camera and Streaming

- Add camera health metrics and reconnect counters.
- Support explicit RTSP transport configuration.
- Add per-camera crop/rotation/debug settings.
- Add stream preview in the Admin Dashboard.

## Deployment

- Add TLS termination and reverse proxy configuration.
- Add separate production Compose files.
- Add GPU-enabled Triton deployment profile.
- Add CI checks for linting, type checks, and smoke tests.

## Model Research

- Export and evaluate the best CDML lightweight model in the production Triton stack.
- Compare latency and accuracy against the current ArcFace-compatible baseline.
- Evaluate anti-spoofing robustness on real presentation attacks.
- Run threshold calibration with production-like camera data.
