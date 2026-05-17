CREATE TABLE IF NOT EXISTS cameras (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    rtsp_url TEXT NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    location TEXT,
    sample_interval_ms INTEGER NOT NULL DEFAULT 500,
    jpeg_quality INTEGER NOT NULL DEFAULT 85,
    reconnect_delay_sec DOUBLE PRECISION NOT NULL DEFAULT 3,
    status TEXT NOT NULL DEFAULT 'unknown',
    last_seen_at TIMESTAMPTZ,
    last_error TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO cameras (
    id,
    name,
    rtsp_url,
    enabled,
    location,
    sample_interval_ms,
    jpeg_quality,
    reconnect_delay_sec
)
VALUES (
    'camera-01',
    'Main camera',
    'rtsp://user:password@camera-ip:554/stream1',
    TRUE,
    'default',
    500,
    85,
    3
)
ON CONFLICT (id) DO NOTHING;
