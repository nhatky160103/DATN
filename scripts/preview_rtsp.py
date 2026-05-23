from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview an RTSP/video source with OpenCV.")
    parser.add_argument("source", help="RTSP URL, video file path, or camera index such as 0")
    parser.add_argument("--window", default="RTSP Preview")
    parser.add_argument("--snapshot", default="", help="Save one frame to this image path and exit.")
    parser.add_argument("--reconnect-delay-sec", type=float, default=2.0)
    args = parser.parse_args()

    source: str | int = int(args.source) if args.source.isdigit() else args.source

    while True:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Cannot open source: {args.source}. Retrying...")
            cap.release()
            time.sleep(args.reconnect_delay_sec)
            continue

        print("Opened source. Press q or Esc to quit.")
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Cannot read frame. Reconnecting...")
                break

            if args.snapshot:
                output_path = Path(args.snapshot)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), frame)
                print(f"Saved snapshot: {output_path}")
                cap.release()
                return

            try:
                cv2.imshow(args.window, frame)
            except cv2.error as exc:
                output_path = Path("rtsp_preview_snapshot.jpg")
                cv2.imwrite(str(output_path), frame)
                print(f"Cannot open GUI preview window: {exc}")
                print(f"Saved one frame instead: {output_path}")
                cap.release()
                return

            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), 27}:
                cap.release()
                cv2.destroyAllWindows()
                return

        cap.release()
        time.sleep(args.reconnect_delay_sec)


if __name__ == "__main__":
    main()
