CREATE TABLE IF NOT EXISTS attendance_events (
    id BIGSERIAL PRIMARY KEY,
    bucket_name TEXT NOT NULL,
    employee_id TEXT NOT NULL,
    camera_id TEXT,
    track_id INTEGER,
    frame_id TEXT,
    status TEXT NOT NULL,
    score DOUBLE PRECISION,
    quality_score DOUBLE PRECISION,
    det_score DOUBLE PRECISION,
    liveness_score DOUBLE PRECISION,
    bbox JSONB,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_attendance_events_occurred_at
    ON attendance_events (occurred_at DESC);

CREATE INDEX IF NOT EXISTS idx_attendance_events_employee_id
    ON attendance_events (employee_id);

CREATE INDEX IF NOT EXISTS idx_attendance_events_camera_id
    ON attendance_events (camera_id);
