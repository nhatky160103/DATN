from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.redis_queue import RedisStreamQueue


def _payload(camera_id: str) -> str:
    return json.dumps(
        {
            "status": "test",
            "camera_id": camera_id,
            "frame_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "message": "hello redis stream",
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test Redis Stream push/read/ack.")
    parser.add_argument("--redis-url", default="redis://localhost:6379/0")
    parser.add_argument("--stream", default="attendance:frames")
    parser.add_argument("--group", default="attendance-workers")
    parser.add_argument("--consumer", default="manual-test")
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--camera-id", default="test-camera")
    parser.add_argument("--timeout-ms", type=int, default=1000)
    args = parser.parse_args()

    producer = RedisStreamQueue(args.redis_url, args.stream, args.max_len)
    producer.push(_payload(args.camera_id))
    print(f"OK push stream={args.stream}")

    latest = producer.peek_latest()
    print("latest:")
    print(json.dumps(latest, ensure_ascii=False, indent=2))

    consumer = RedisStreamQueue(
        args.redis_url,
        args.stream,
        args.max_len,
        group=args.group,
        consumer=args.consumer,
    )
    message = consumer.pop(timeout_ms=args.timeout_ms)
    if message is None:
        raise SystemExit("No message received from Redis Stream")

    print(f"OK read stream_id={message.stream_id}")
    print("payload:")
    print(json.dumps(json.loads(message.payload), ensure_ascii=False, indent=2))

    consumer.ack(message.stream_id)
    print(f"OK ack stream_id={message.stream_id}")


if __name__ == "__main__":
    main()

