"""
Experiment tracking and versioning.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import time
import subprocess
from datetime import datetime
import sqlite3
import uuid
import shutil


class Experiment:
    """
    Track and version experiments.

    Automatically logs config, metrics, code version, and artifacts.

    Args:
        name: Experiment name
        config: Configuration object or dict
        description: Optional description
        tags: List of tags for categorization
        root_dir: Root directory for experiments

    Example:
        >>> exp = Experiment("vae-baseline", config, tags=["vae", "mnist"])
        >>> with exp:
        ...     trainer.set_experiment(exp)
        ...     trainer.train(train_loader, val_loader)
    """

    def __init__(
        self,
        name: str,
        config: Any = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        root_dir: str = "experiments",
    ):
        self.name = name
        self.config = config
        self.description = description
        self.tags = tags or []
        self.root_dir = Path(root_dir)

        # Generate unique ID
        self.experiment_id = f"{name}_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        # Create experiment directory
        self.exp_dir = self.root_dir / self.experiment_id
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.log_dir = self.exp_dir / "logs"
        self.artifact_dir = self.exp_dir / "artifacts"

        for d in [self.checkpoint_dir, self.log_dir, self.artifact_dir]:
            d.mkdir(exist_ok=True)

        # State
        self.status = "pending"
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

        # Database
        self.db_path = self.root_dir / "experiments.db"
        self._init_database()

        # Metrics cache
        self._metrics_buffer: List[Dict] = []
        self._buffer_size = 100  # Flush every 100 metrics

    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables (idempotent)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'pending',
                git_hash TEXT,
                git_dirty BOOLEAN,
                path TEXT NOT NULL,
                tags TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                epoch INTEGER,
                step INTEGER,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_type TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                config_key TEXT NOT NULL,
                config_value TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                artifact_path TEXT NOT NULL,
                artifact_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_metrics_experiment ON metrics(experiment_id)"
        )

        conn.commit()
        conn.close()

    def _get_git_info(self) -> Dict[str, Any]:
        """Get current git commit hash and dirty status"""
        try:
            # Get commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            )
            git_hash = result.stdout.strip()

            # Check if dirty
            result = subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            )
            git_dirty = len(result.stdout.strip()) > 0

            return {"git_hash": git_hash, "git_dirty": git_dirty}
        except:
            return {"git_hash": None, "git_dirty": None}

    def start(self):
        """Start tracking experiment"""
        self.status = "running"
        self.start_time = time.time()

        # Get git info
        git_info = self._get_git_info()

        # Save experiment to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO experiments 
            (experiment_id, name, description, status, git_hash, git_dirty, path, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                self.experiment_id,
                self.name,
                self.description,
                self.status,
                git_info["git_hash"],
                git_info["git_dirty"],
                str(self.exp_dir),
                json.dumps(self.tags),
            ),
        )

        conn.commit()
        conn.close()

        # Save config
        if self.config is not None:
            self._save_config()

        # Save metadata
        metadata = {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "start_time": self.start_time,
            "git_hash": git_info["git_hash"],
            "git_dirty": git_info["git_dirty"],
            "status": self.status,
        }

        with open(self.exp_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _save_config(self):
        """Save configuration"""
        # Save as YAML if Config object
        if hasattr(self.config, "to_dict"):
            config_dict = self.config.to_dict()
        elif isinstance(self.config, dict):
            config_dict = self.config
        else:
            config_dict = {"config": str(self.config)}

        # Save to file
        import yaml

        with open(self.exp_dir / "config.yaml", "w") as f:
            yaml.dump(config_dict, f)

        # Save to database (flattened)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        def flatten_dict(d, parent_key="", sep="."):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, json.dumps(v)))
            return dict(items)

        flat_config = flatten_dict(config_dict)

        for key, value in flat_config.items():
            cursor.execute(
                """
                INSERT INTO configs (experiment_id, config_key, config_value)
                VALUES (?, ?, ?)
            """,
                (self.experiment_id, key, value),
            )

        conn.commit()
        conn.close()

    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        metric_type: str = "train",
    ):
        """
        Log a metric.

        Args:
            name: Metric name
            value: Metric value
            step: Training step
            epoch: Training epoch
            metric_type: Type (train, val, test)
        """
        # Add to buffer
        self._metrics_buffer.append(
            {
                "experiment_id": self.experiment_id,
                "metric_name": name,
                "metric_value": float(value),
                "step": step,
                "epoch": epoch,
                "metric_type": metric_type,
            }
        )

        # Flush if buffer full
        if len(self._metrics_buffer) >= self._buffer_size:
            self._flush_metrics()

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        metric_type: str = "train",
    ):
        """Log multiple metrics at once"""
        for name, value in metrics.items():
            self.log_metric(name, value, step, epoch, metric_type)

    def _flush_metrics(self):
        """Flush metrics buffer to database"""
        if not self._metrics_buffer:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.executemany(
            """
            INSERT INTO metrics 
            (experiment_id, metric_name, metric_value, step, epoch, metric_type)
            VALUES (:experiment_id, :metric_name, :metric_value, :step, :epoch, :metric_type)
        """,
            self._metrics_buffer,
        )

        conn.commit()
        conn.close()

        self._metrics_buffer.clear()

    def log_artifact(
        self, artifact_path: str, artifact_type: str = "file", artifact_name: Optional[str] = None
    ):
        """
        Log an artifact (file, checkpoint, image, etc.)

        Args:
            artifact_path: Path to artifact
            artifact_type: Type of artifact
            artifact_name: Optional name
        """
        artifact_path = Path(artifact_path)

        # Copy to artifact directory
        dest_path = self.artifact_dir / artifact_path.name
        if artifact_path.exists():
            shutil.copy2(artifact_path, dest_path)

        # Log to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO artifacts 
            (experiment_id, artifact_type, artifact_path, artifact_name)
            VALUES (?, ?, ?, ?)
        """,
            (
                self.experiment_id,
                artifact_type,
                str(dest_path),
                artifact_name or artifact_path.name,
            ),
        )

        conn.commit()
        conn.close()

    def end(self, status: str = "completed"):
        """End experiment tracking"""
        self.status = status
        self.end_time = time.time()

        # Flush remaining metrics
        self._flush_metrics()

        # Update database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE experiments 
            SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE experiment_id = ?
        """,
            (status, self.experiment_id),
        )

        conn.commit()
        conn.close()

        # Update metadata
        metadata_path = self.exp_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            metadata["status"] = status
            metadata["end_time"] = self.end_time
            metadata["duration"] = self.end_time - self.start_time

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

    def get_metrics(self, metric_name: Optional[str] = None) -> List[Dict]:
        """
        Get logged metrics.

        Args:
            metric_name: Filter by metric name

        Returns:
            List of metric dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if metric_name:
            cursor.execute(
                """
                SELECT * FROM metrics 
                WHERE experiment_id = ? AND metric_name = ?
                ORDER BY step, epoch
            """,
                (self.experiment_id, metric_name),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM metrics 
                WHERE experiment_id = ?
                ORDER BY step, epoch
            """,
                (self.experiment_id,),
            )

        metrics = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return metrics

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if exc_type is not None:
            self.end(status="failed")
        else:
            self.end(status="completed")
        return False

    def __repr__(self):
        return f"Experiment(id='{self.experiment_id}', name='{self.name}', status='{self.status}')"
