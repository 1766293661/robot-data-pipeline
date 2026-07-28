from __future__ import annotations

from typing import Any

from core.adapters import make_adapter
from core.config import PipelineConfig
from core.normalize import normalize_episode
from core.quality import evaluate_episode
from core.store import Store


class SimulatedInterruption(RuntimeError):
    pass


class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, interrupt_after_stage: str | None = None) -> dict[str, Any]:
        store = Store(self.config.paths.database_path)
        run_id = store.begin_run()
        counts = {"discovered": 0, "new_episodes": 0, "skipped": 0, "normalized": 0, "quality_checked": 0, "stored": 0, "failed": 0}
        try:
            for source in self.config.sources:
                if not source.enabled:
                    continue
                store.register_source(source, source.model_dump(mode="json"))
                stored_episode_ids = store.stored_episode_ids(source.source_id, source.source_revision)
                for raw in make_adapter(source).discover(stored_episode_ids):
                    counts["discovered"] += 1
                    episode, created = store.ensure_episode(raw)
                    if created:
                        counts["new_episodes"] += 1
                    episode_pk = int(episode["episode_pk"])
                    if episode["stage"] == "stored":
                        counts["skipped"] += 1
                        continue
                    try:
                        if episode["stage"] == "discovered":
                            store.save_normalized(episode_pk, normalize_episode(raw, source))
                            counts["normalized"] += 1
                            if interrupt_after_stage == "normalized":
                                raise SimulatedInterruption("interrupted after normalized transaction")
                        episode = store.get_episode(raw)  # Refresh last committed stage.
                        if episode["stage"] == "normalized":
                            frames = store.load_frames(episode_pk)
                            status, results = evaluate_episode(frames, raw.native_fps, raw.capabilities)
                            store.save_quality(episode_pk, status, results)
                            counts["quality_checked"] += 1
                            if interrupt_after_stage == "quality_checked":
                                raise SimulatedInterruption("interrupted after quality transaction")
                        episode = store.get_episode(raw)
                        if episode["stage"] == "quality_checked":
                            store.mark_stored(episode_pk)
                            counts["stored"] += 1
                    except SimulatedInterruption:
                        raise
                    except Exception as exc:
                        store.record_error(episode_pk, str(exc))
                        counts["failed"] += 1
        finally:
            report = store.report(run_id, counts)
            store.close()
        return report
