"""
FRAMEWORM DEPLOY model registry — unified API.
Supports both:
  • Steps 1–5 manifest-based API  (register(model_name, manifest, ...) → DeploymentRecord)
  • Steps 6–10 flat API           (register(name, version, type, path, stage) → dict)

Windows-safe: journal_mode=DELETE + explicit conn.close() in a real contextmanager.
"""

import sqlite3
import logging
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict

logger = logging.getLogger("frameworm.deploy")

REGISTRY_DB = Path("experiments/deploy_registry/registry.db")

VALID_STAGES = {"dev", "staging", "production", "archived"}


class ModelStage(Enum):
    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"

    @classmethod
    def transitions(cls) -> Dict[str, list]:
        return {
            cls.DEV.value: [cls.STAGING.value, cls.ARCHIVED.value],
            cls.STAGING.value: [cls.PRODUCTION.value, cls.DEV.value, cls.ARCHIVED.value],
            cls.PRODUCTION.value: [cls.ARCHIVED.value, cls.STAGING.value],
            cls.ARCHIVED.value: [],
        }


@dataclass
class DeploymentRecord:
    """Steps 1–5 record object returned by the manifest-based API."""

    id: str
    model_name: str
    version: str
    stage: str
    export_dir: str
    manifest_path: str
    checkpoint_path: str
    checkpoint_hash: str
    model_class: str
    experiment_id: Optional[str]
    git_hash: Optional[str]
    config_snapshot: Optional[str]
    training_metrics: Optional[str]
    deployed_at: Optional[str]
    archived_at: Optional[str]
    created_at: str
    notes: str = ""
    tags: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_row(cls, row, columns: list) -> "DeploymentRecord":
        return cls(**dict(zip(columns, row)))


class ModelRegistry:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else REGISTRY_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ─────────────────────────────────────────────────────────────────────────
    # Connection management
    # ─────────────────────────────────────────────────────────────────────────

    @contextmanager
    def _conn(self):
        """
        The root fix for WinError 32.

        sqlite3.connect() used as a plain context manager (with sqlite3.connect(...) as conn)
        only handles transactions — its __exit__ calls commit() or rollback() but NEVER
        close(). The OS file handle stays open, so Windows refuses to delete the file.

        This @contextmanager explicitly calls conn.close() in the finally block,
        guaranteeing the handle is released before TemporaryDirectory.__exit__ runs.

        journal_mode=DELETE prevents .db-wal/.db-shm sidecar files (WAL mode default)
        that Windows also locks separately.
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.commit()
            conn.row_factory = sqlite3.Row
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()  # ← releases the OS file handle immediately

    # ─────────────────────────────────────────────────────────────────────────
    # UNIFIED register()
    # ─────────────────────────────────────────────────────────────────────────

    def register(
        self,
        name_or_model_name,
        version_or_manifest=None,
        model_type_or_experiment_id=None,
        checkpoint_path=None,
        stage=None,
        notes="",
        tags=None,
    ):
        if isinstance(version_or_manifest, str):
            return self._register_flat(
                name=name_or_model_name,
                version=version_or_manifest,
                model_type=model_type_or_experiment_id or "generic",
                checkpoint_path=checkpoint_path or "",
                stage=stage or "dev",
                notes=notes,
            )
        else:
            return self._register_manifest(
                model_name=name_or_model_name,
                manifest=version_or_manifest,
                experiment_id=model_type_or_experiment_id,
                notes=notes,
                tags=tags,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Steps 1–5 manifest-based API
    # ─────────────────────────────────────────────────────────────────────────

    def _register_manifest(
        self, model_name, manifest, experiment_id=None, notes="", tags=None
    ) -> DeploymentRecord:
        git_hash, config_snapshot, training_metrics = self._pull_experiment_context(experiment_id)

        version = self._next_semver(model_name)
        record_id = str(uuid.uuid4())[:8]
        now = datetime.utcnow().isoformat()

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO deployments (
                    id, model_name, version, stage,
                    export_dir, manifest_path, checkpoint_path, checkpoint_hash,
                    model_class, experiment_id, git_hash,
                    config_snapshot, training_metrics,
                    deployed_at, archived_at, created_at, notes, tags
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
                (
                    record_id,
                    model_name,
                    version,
                    ModelStage.DEV.value,
                    getattr(manifest, "export_dir", ""),
                    str(Path(getattr(manifest, "export_dir", "")) / "manifest.json"),
                    getattr(manifest, "checkpoint_path", ""),
                    getattr(manifest, "checkpoint_hash", ""),
                    getattr(manifest, "model_class", "unknown"),
                    experiment_id,
                    git_hash,
                    config_snapshot,
                    training_metrics,
                    None,
                    None,
                    now,
                    notes,
                    ",".join(tags) if tags else "",
                ),
            )

        print(f"[DEPLOY] Registered '{model_name}' v{version} " f"(id={record_id}, stage=DEV)")
        return self.get(record_id)

    def transition(
        self, record_id: str, new_stage: ModelStage, notes: str = ""
    ) -> DeploymentRecord:
        record = self.get(record_id)
        if record is None:
            raise ValueError(f"[DEPLOY] No record with id={record_id}")

        valid = ModelStage.transitions().get(record.stage, [])
        if new_stage.value not in valid:
            raise ValueError(
                f"[DEPLOY] Invalid transition: {record.stage} → {new_stage.value}. "
                f"Valid: {valid}"
            )

        if new_stage == ModelStage.PRODUCTION:
            current_prod = self.get_production(record.model_name)
            if current_prod and current_prod.id != record_id:
                self._update_fields(
                    current_prod.id,
                    {
                        "stage": ModelStage.ARCHIVED.value,
                        "archived_at": datetime.utcnow().isoformat(),
                    },
                )
                print(
                    f"[DEPLOY] Archived previous production "
                    f"v{current_prod.version} (id={current_prod.id})"
                )

        updates: dict = {"stage": new_stage.value}
        if new_stage == ModelStage.PRODUCTION:
            updates["deployed_at"] = datetime.utcnow().isoformat()
        if new_stage == ModelStage.ARCHIVED:
            updates["archived_at"] = datetime.utcnow().isoformat()
        if notes:
            updates["notes"] = (record.notes or "") + f"\n[{new_stage.value}] {notes}"

        self._update_fields(record_id, updates)
        print(f"[DEPLOY] '{record.model_name}' v{record.version} → {new_stage.value}")
        return self.get(record_id)

    def get(self, record_id: str) -> Optional[DeploymentRecord]:
        with self._conn() as conn:
            cur = conn.execute("SELECT * FROM deployments WHERE id=?", (record_id,))
            row = cur.fetchone()
            if row is None:
                return None
            cols = [d[0] for d in cur.description]
            return DeploymentRecord.from_row(tuple(row), cols)

    def get_production(self, model_name: str) -> Optional[DeploymentRecord]:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT * FROM deployments WHERE model_name=? AND stage=? "
                "ORDER BY created_at DESC LIMIT 1",
                (model_name, ModelStage.PRODUCTION.value),
            )
            row = cur.fetchone()
            if row is None:
                return None
            cols = [d[0] for d in cur.description]
            return DeploymentRecord.from_row(tuple(row), cols)

    def get_staging(self, model_name: str) -> Optional[DeploymentRecord]:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT * FROM deployments WHERE model_name=? AND stage=? "
                "ORDER BY created_at DESC LIMIT 1",
                (model_name, ModelStage.STAGING.value),
            )
            row = cur.fetchone()
            if row is None:
                return None
            cols = [d[0] for d in cur.description]
            return DeploymentRecord.from_row(tuple(row), cols)

    def history(self, model_name: str, limit: int = 20) -> List[DeploymentRecord]:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT * FROM deployments WHERE model_name=? " "ORDER BY created_at DESC LIMIT ?",
                (model_name, limit),
            )
            cols = [d[0] for d in cur.description]
            return [DeploymentRecord.from_row(tuple(r), cols) for r in cur.fetchall()]

    def list_models(self) -> List[str]:
        with self._conn() as conn:
            cur = conn.execute("SELECT DISTINCT model_name FROM deployments ORDER BY model_name")
            return [r[0] for r in cur.fetchall()]

    def print_status(self, model_name: Optional[str] = None):
        models = [model_name] if model_name else self.list_models()
        if not models:
            print("[DEPLOY] No models registered yet.")
            return
        colours = {
            "dev": "\033[94m",
            "staging": "\033[93m",
            "production": "\033[92m",
            "archived": "\033[90m",
        }
        reset = "\033[0m"
        print(f"\n  {'Model':<25} {'Version':<10} {'Stage':<12} " f"{'Class':<12} {'Deployed'}")
        print(f"  {'─'*25} {'─'*10} {'─'*12} {'─'*12} {'─'*20}")
        for name in models:
            for rec in self.history(name, limit=5):
                c = colours.get(rec.stage, "")
                dep = rec.deployed_at[:10] if rec.deployed_at else "—"
                print(
                    f"  {rec.model_name:<25} {rec.version:<10} "
                    f"{c}{rec.stage:<12}{reset} {rec.model_class:<12} {dep}"
                )
        print()

    def close(self):
        """No-op — every _conn() call closes itself in its finally block."""
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # Steps 6–10 flat API
    # ─────────────────────────────────────────────────────────────────────────

    def _register_flat(
        self,
        name: str,
        version: str,
        model_type: str,
        checkpoint_path: str,
        stage: str = "dev",
        notes: str = "",
    ) -> dict:
        if stage not in VALID_STAGES:
            raise ValueError(f"Invalid stage '{stage}'. Valid: {VALID_STAGES}")

        now = datetime.utcnow().isoformat()
        record_id = str(uuid.uuid4())[:8]

        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO deployments (
                    id, model_name, version, stage,
                    export_dir, manifest_path, checkpoint_path, checkpoint_hash,
                    model_class, experiment_id, git_hash,
                    config_snapshot, training_metrics,
                    deployed_at, archived_at, created_at, notes, tags
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
                (
                    record_id,
                    name,
                    version,
                    stage,
                    "",
                    "",
                    checkpoint_path,
                    "",
                    model_type,
                    None,
                    None,
                    None,
                    None,
                    now if stage == "production" else None,
                    now if stage == "archived" else None,
                    now,
                    notes,
                    "",
                ),
            )

        print(f"[DEPLOY] Registered '{name}' {version} (stage={stage})")
        return self._get_dict(name, version)

    def promote(self, name: str, version: str, new_stage: str, note: str = "") -> dict:
        if new_stage not in VALID_STAGES:
            raise ValueError(f"Invalid stage '{new_stage}'")
        updates: dict = {"stage": new_stage}
        if new_stage == "production":
            updates["deployed_at"] = datetime.utcnow().isoformat()
        if new_stage == "archived":
            updates["archived_at"] = datetime.utcnow().isoformat()
        self._update_by_name_version(name, version, updates)
        print(f"[DEPLOY] '{name}' {version} → {new_stage}")
        return self._get_dict(name, version)

    def demote(self, name: str, version: str, new_stage: str, note: str = "") -> dict:
        return self.promote(name, version, new_stage, note)

    def get_by_name(self, name: str) -> List[dict]:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT * FROM deployments WHERE model_name=? " "ORDER BY created_at DESC",
                (name,),
            )
            return self._rows(cur)

    def get_current_production(self, name: str) -> Optional[dict]:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT * FROM deployments WHERE model_name=? AND stage='production' "
                "ORDER BY deployed_at DESC LIMIT 1",
                (name,),
            )
            rows = self._rows(cur)
            return rows[0] if rows else None

    def get_production_history(self, name: str) -> List[dict]:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT * FROM deployments "
                "WHERE model_name=? AND (stage='production' OR deployed_at IS NOT NULL) "
                "ORDER BY deployed_at DESC",
                (name,),
            )
            return self._rows(cur)

    def list_all(self):
        with self._conn() as conn:
            cur = conn.execute("SELECT * FROM deployments ORDER BY model_name, created_at DESC")
            rows = self._rows(cur)

        if not rows:
            print("[DEPLOY] No deployments registered.")
            return

        colours = {
            "dev": "\033[94m",
            "staging": "\033[93m",
            "production": "\033[92m",
            "archived": "\033[90m",
        }
        reset = "\033[0m"
        print(f"\n  {'Name':<20} {'Version':<10} {'Stage':<12} " f"{'Type':<12} {'Deployed'}")
        print(f"  {'─'*20} {'─'*10} {'─'*12} {'─'*12} {'─'*20}")
        for r in rows:
            c = colours.get(r["stage"], "")
            dep = (r["deployed_at"] or "—")[:10]
            marker = "▶ " if r["stage"] == "production" else "  "
            print(
                f"  {marker}{r['model_name']:<18} {r['version']:<10} "
                f"{c}{r['stage']:<12}{reset} {r['model_class']:<12} {dep}"
            )
        print()

    # ─────────────────────────────────────────────────────────────────────────
    # DB internals
    # ─────────────────────────────────────────────────────────────────────────

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS deployments (
                    id               TEXT PRIMARY KEY,
                    model_name       TEXT NOT NULL,
                    version          TEXT NOT NULL,
                    stage            TEXT NOT NULL DEFAULT 'dev',
                    export_dir       TEXT DEFAULT '',
                    manifest_path    TEXT DEFAULT '',
                    checkpoint_path  TEXT DEFAULT '',
                    checkpoint_hash  TEXT DEFAULT '',
                    model_class      TEXT DEFAULT '',
                    experiment_id    TEXT,
                    git_hash         TEXT,
                    config_snapshot  TEXT,
                    training_metrics TEXT,
                    deployed_at      TEXT,
                    archived_at      TEXT,
                    created_at       TEXT NOT NULL,
                    notes            TEXT DEFAULT '',
                    tags             TEXT DEFAULT ''
                )
            """)

    def _get_dict(self, name: str, version: str) -> Optional[dict]:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT * FROM deployments WHERE model_name=? AND version=?",
                (name, version),
            )
            rows = self._rows(cur)
            return rows[0] if rows else None

    def _update_fields(self, record_id: str, fields: dict):
        sets = ", ".join(f"{k}=?" for k in fields)
        vals = list(fields.values()) + [record_id]
        with self._conn() as conn:
            conn.execute(f"UPDATE deployments SET {sets} WHERE id=?", vals)

    def _update_by_name_version(self, name: str, version: str, fields: dict):
        sets = ", ".join(f"{k}=?" for k in fields)
        vals = list(fields.values()) + [name, version]
        with self._conn() as conn:
            conn.execute(
                f"UPDATE deployments SET {sets} WHERE model_name=? AND version=?",
                vals,
            )

    def _next_semver(self, model_name: str) -> str:
        recs = self.history(model_name, limit=1)
        if not recs:
            return "1.0.0"
        last = recs[0].version
        try:
            major, minor, patch = map(int, last.split("."))
            return f"{major}.{minor}.{patch + 1}"
        except Exception:
            return "1.0.0"

    @staticmethod
    def _rows(cursor) -> List[dict]:
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    # ─────────────────────────────────────────────────────────────────────────
    # Experiment lineage (Steps 1–5)
    # ─────────────────────────────────────────────────────────────────────────

    def _pull_experiment_context(self, experiment_id: Optional[str]):
        if experiment_id is None:
            return self._current_git_hash(), None, None
        try:
            exp_db = Path("experiments/experiments.db")
            if not exp_db.exists():
                return self._current_git_hash(), None, None
            conn = sqlite3.connect(str(exp_db))
            try:
                conn.execute("PRAGMA journal_mode=DELETE")
                conn.commit()
                cur = conn.execute(
                    "SELECT * FROM experiments WHERE id=? LIMIT 1",
                    (experiment_id,),
                )
                row = cur.fetchone()
                if row:
                    cols = [d[0] for d in cur.description]
                    d = dict(zip(cols, row))
                    return (
                        d.get("git_hash"),
                        d.get("config_snapshot"),
                        d.get("final_metrics"),
                    )
            finally:
                conn.close()
        except Exception as e:
            logger.warning(f"[DEPLOY] Could not pull experiment context: {e}")
        return self._current_git_hash(), None, None

    def _current_git_hash(self) -> Optional[str]:
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None
