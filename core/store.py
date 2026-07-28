from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.models import NormalizedFrame, QualityResult, RawEpisode


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript("""
        CREATE TABLE IF NOT EXISTS sources (
          source_id TEXT NOT NULL, source_revision TEXT NOT NULL, source_uri TEXT NOT NULL,
          format TEXT NOT NULL, manifest_json TEXT NOT NULL, updated_at TEXT NOT NULL,
          PRIMARY KEY (source_id, source_revision)
        );
        CREATE TABLE IF NOT EXISTS episodes (
          episode_pk INTEGER PRIMARY KEY AUTOINCREMENT,
          source_id TEXT NOT NULL, source_revision TEXT NOT NULL, native_episode_id TEXT NOT NULL,
          source_format TEXT NOT NULL, robot_type TEXT, native_fps REAL, capabilities_json TEXT NOT NULL,
          task_json TEXT NOT NULL, locator_json TEXT NOT NULL, artifact_fingerprint TEXT NOT NULL,
          frame_count INTEGER NOT NULL DEFAULT 0, stage TEXT NOT NULL DEFAULT 'discovered',
          quality_status TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
          UNIQUE (source_id, source_revision, native_episode_id),
          FOREIGN KEY (source_id, source_revision) REFERENCES sources(source_id, source_revision)
        );
        CREATE TABLE IF NOT EXISTS checkpoints (
          episode_pk INTEGER PRIMARY KEY, stage TEXT NOT NULL, attempt INTEGER NOT NULL DEFAULT 0,
          last_error TEXT, updated_at TEXT NOT NULL,
          FOREIGN KEY (episode_pk) REFERENCES episodes(episode_pk) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS frames (
          episode_pk INTEGER NOT NULL, frame_index INTEGER NOT NULL, native_timestamp_sec REAL,
          derived_timestamp_sec REAL, time_basis TEXT NOT NULL, action_json TEXT, state_json TEXT,
          camera_refs_json TEXT NOT NULL, extra_json TEXT NOT NULL,
          PRIMARY KEY (episode_pk, frame_index),
          FOREIGN KEY (episode_pk) REFERENCES episodes(episode_pk) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS quality_results (
          episode_pk INTEGER NOT NULL, rule_id TEXT NOT NULL, severity TEXT NOT NULL, passed INTEGER NOT NULL,
          evidence_json TEXT NOT NULL, rule_version TEXT NOT NULL,
          PRIMARY KEY (episode_pk, rule_id),
          FOREIGN KEY (episode_pk) REFERENCES episodes(episode_pk) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS runs (
          run_id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT NOT NULL, finished_at TEXT, report_json TEXT
        );
        """)
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def register_source(self, source, manifest_json: dict[str, Any]) -> None:
        with self.connection:
            self.connection.execute("""
            INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id, source_revision) DO UPDATE SET source_uri=excluded.source_uri,
            format=excluded.format, manifest_json=excluded.manifest_json, updated_at=excluded.updated_at
            """, (source.source_id, source.source_revision, source.source_uri, source.format, json.dumps(manifest_json), _now()))

    def begin_run(self) -> int:
        cursor = self.connection.execute("INSERT INTO runs(started_at) VALUES (?)", (_now(),))
        self.connection.commit()
        return int(cursor.lastrowid)

    def ensure_episode(self, raw: RawEpisode) -> tuple[sqlite3.Row, bool]:
        existing = self.get_episode(raw)
        if existing:
            return existing, False
        with self.connection:
            cursor = self.connection.execute("""
            INSERT INTO episodes(source_id,source_revision,native_episode_id,source_format,robot_type,native_fps,
              capabilities_json,task_json,locator_json,artifact_fingerprint,created_at,updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (raw.source_id, raw.source_revision, raw.native_episode_id, raw.source_format, raw.robot_type,
                  raw.native_fps, json.dumps(raw.capabilities), json.dumps(raw.task), json.dumps(raw.locator), raw.fingerprint, _now(), _now()))
            episode_pk = int(cursor.lastrowid)
            self.connection.execute("INSERT INTO checkpoints(episode_pk,stage,updated_at) VALUES (?, 'discovered', ?)", (episode_pk, _now()))
        return self.get_episode(raw), True  # type: ignore[return-value]

    def save_normalized(self, episode_pk: int, frames: list[NormalizedFrame]) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM frames WHERE episode_pk=?", (episode_pk,))
            self.connection.executemany("""
            INSERT INTO frames VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [(episode_pk, frame.frame_index, frame.native_timestamp_sec, frame.derived_timestamp_sec, frame.time_basis,
                    json.dumps(frame.action), json.dumps(frame.state), json.dumps(frame.camera_refs), json.dumps(frame.extra)) for frame in frames])
            self._advance(episode_pk, "normalized", frame_count=len(frames))

    def load_frames(self, episode_pk: int) -> list[NormalizedFrame]:
        rows = self.connection.execute("SELECT * FROM frames WHERE episode_pk=? ORDER BY frame_index", (episode_pk,)).fetchall()
        return [NormalizedFrame(row["frame_index"], row["native_timestamp_sec"], row["derived_timestamp_sec"], row["time_basis"],
                                json.loads(row["action_json"]) if row["action_json"] else None,
                                json.loads(row["state_json"]) if row["state_json"] else None,
                                json.loads(row["camera_refs_json"]), json.loads(row["extra_json"])) for row in rows]

    def save_quality(self, episode_pk: int, status: str, results: list[QualityResult]) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM quality_results WHERE episode_pk=?", (episode_pk,))
            self.connection.executemany("INSERT INTO quality_results VALUES (?, ?, ?, ?, ?, ?)", [
                (episode_pk, result.rule_id, result.severity, int(result.passed), json.dumps(result.evidence), result.rule_version) for result in results])
            self._advance(episode_pk, "quality_checked", quality_status=status)

    def mark_stored(self, episode_pk: int) -> None:
        with self.connection:
            self._advance(episode_pk, "stored")

    def record_error(self, episode_pk: int, error: str) -> None:
        with self.connection:
            self.connection.execute("UPDATE checkpoints SET attempt=attempt+1,last_error=?,updated_at=? WHERE episode_pk=?", (error[:2000], _now(), episode_pk))

    def report(self, run_id: int, run: dict[str, Any]) -> dict[str, Any]:
        cumulative = {
            "episodes": self.connection.execute("SELECT COUNT(*) FROM episodes WHERE stage='stored'").fetchone()[0],
            "frames": self.connection.execute("SELECT COALESCE(SUM(frame_count),0) FROM episodes WHERE stage='stored'").fetchone()[0],
            "by_source": [dict(row) for row in self.connection.execute("SELECT source_id,COUNT(*) episodes,SUM(frame_count) frames FROM episodes WHERE stage='stored' GROUP BY source_id")],
            "by_robot_type": [dict(row) for row in self.connection.execute("SELECT robot_type,COUNT(*) episodes,SUM(frame_count) frames FROM episodes WHERE stage='stored' GROUP BY robot_type")],
            "quality_rules": [dict(row) for row in self.connection.execute("SELECT rule_id,COUNT(*) checks,SUM(CASE WHEN passed=0 THEN 1 ELSE 0 END) failures FROM quality_results GROUP BY rule_id")],
        }
        report = {"run_id": run_id, "run": run, "cumulative": cumulative}
        with self.connection:
            self.connection.execute("UPDATE runs SET finished_at=?,report_json=? WHERE run_id=?", (_now(), json.dumps(report), run_id))
        return report

    def accepted_episode_rows(self, include_needs_review: bool = False) -> list[sqlite3.Row]:
        statuses = ("accepted", "needs_review") if include_needs_review else ("accepted",)
        placeholders = ",".join("?" for _ in statuses)
        return self.connection.execute(f"SELECT * FROM episodes WHERE stage='stored' AND quality_status IN ({placeholders}) ORDER BY source_id,native_episode_id", statuses).fetchall()

    def stored_episode_ids(self, source_id: str, source_revision: str) -> set[str]:
        rows = self.connection.execute("SELECT native_episode_id FROM episodes WHERE source_id=? AND source_revision=? AND stage='stored'", (source_id, source_revision))
        return {row[0] for row in rows}

    def get_episode(self, raw: RawEpisode) -> sqlite3.Row | None:
        return self.connection.execute("SELECT * FROM episodes WHERE source_id=? AND source_revision=? AND native_episode_id=?", (raw.source_id, raw.source_revision, raw.native_episode_id)).fetchone()

    def _advance(self, episode_pk: int, stage: str, *, frame_count: int | None = None, quality_status: str | None = None) -> None:
        assignments = ["stage=?", "updated_at=?"]
        values: list[Any] = [stage, _now()]
        if frame_count is not None:
            assignments.append("frame_count=?")
            values.append(frame_count)
        if quality_status is not None:
            assignments.append("quality_status=?")
            values.append(quality_status)
        values.append(episode_pk)
        self.connection.execute(f"UPDATE episodes SET {','.join(assignments)} WHERE episode_pk=?", values)
        self.connection.execute("UPDATE checkpoints SET stage=?,last_error=NULL,updated_at=? WHERE episode_pk=?", (stage, _now(), episode_pk))
