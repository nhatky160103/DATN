from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Face attendance system entrypoint")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("frame-reader")
    sub.add_parser("worker")
    sub.add_parser("api")
    args = parser.parse_args()

    if args.command == "frame-reader":
        from attendance_pipeline.frame_reader import run_frame_reader

        run_frame_reader()
    elif args.command == "worker":
        from attendance_pipeline.config import load_pipeline_config
        from attendance_pipeline.worker import AttendancePipelineWorker

        AttendancePipelineWorker(load_pipeline_config()).run_forever()
    elif args.command == "api":
        from attendance_pipeline.api import app

        app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
