from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Register or update a camera in PostgreSQL camera registry.")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://camera_app:camera_app_dev_password@localhost:5432/camera_db"))
    parser.add_argument("--id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source", required=True, help="RTSP URL or camera source")
    parser.add_argument("--location", default="")
    parser.add_argument("--sample-interval-ms", type=int, default=500)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--reconnect-delay-sec", type=float, default=3.0)
    parser.add_argument("--disabled", action="store_true")
    args = parser.parse_args()

    import psycopg

    with psycopg.connect(args.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO cameras (
                    id,
                    name,
                    rtsp_url,
                    enabled,
                    location,
                    sample_interval_ms,
                    jpeg_quality,
                    reconnect_delay_sec,
                    updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    rtsp_url = EXCLUDED.rtsp_url,
                    enabled = EXCLUDED.enabled,
                    location = EXCLUDED.location,
                    sample_interval_ms = EXCLUDED.sample_interval_ms,
                    jpeg_quality = EXCLUDED.jpeg_quality,
                    reconnect_delay_sec = EXCLUDED.reconnect_delay_sec,
                    updated_at = now()
                """,
                (
                    args.id,
                    args.name,
                    args.source,
                    not args.disabled,
                    args.location,
                    args.sample_interval_ms,
                    args.jpeg_quality,
                    args.reconnect_delay_sec,
                ),
            )

    print(f"Registered camera {args.id}")


if __name__ == "__main__":
    main()
