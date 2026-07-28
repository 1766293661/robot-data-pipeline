from __future__ import annotations

import argparse
import json

from core.config import load_config
from core.export import export_jsonl
from core.pipeline import Pipeline, SimulatedInterruption


def main() -> None:
    parser = argparse.ArgumentParser(description="Robot data pipeline launcher")
    parser.add_argument("--config", default="config.yaml", help="path to pipeline YAML configuration")
    parser.add_argument("--check-config", action="store_true", help="validate and print resolved configuration")
    parser.add_argument("--export-jsonl", action="store_true", help="export accepted data from the existing SQLite database")
    parser.add_argument("--frame-budget", type=int, help="override export.frame_budget")
    parser.add_argument("--clip-length", type=int, help="override export.clip_length")
    parser.add_argument("--interrupt-after-stage", choices=["normalized", "quality_checked"], help="test only: raise after the selected committed stage")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.check_config:
        print(json.dumps(config.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return
    if args.export_jsonl:
        result = export_jsonl(
            config.paths.database_path,
            config.paths.output_root / config.export.output_filename,
            args.frame_budget or config.export.frame_budget,
            args.clip_length or config.export.clip_length,
            config.export.include_needs_review,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    try:
        report = Pipeline(config).run(args.interrupt_after_stage)
    except SimulatedInterruption as exc:
        print(json.dumps({"interrupted": str(exc)}, ensure_ascii=False))
        raise SystemExit(2) from exc
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
