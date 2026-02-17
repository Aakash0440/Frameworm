"""
Experiment management and comparison tools.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import sqlite3
import json
from tabulate import tabulate
import pandas as pd


class ExperimentManager:
    """
    Manage and compare experiments.

    Args:
        root_dir: Root directory for experiments
    """

    def __init__(self, root_dir: str = "experiments"):
        self.root_dir = Path(root_dir)
        self.db_path = self.root_dir / "experiments.db"

        if not self.db_path.exists():
            raise FileNotFoundError(f"No experiments database found at {self.db_path}")

    def list_experiments(
        self,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        List experiments with filters.

        Args:
            status: Filter by status (running, completed, failed)
            tags: Filter by tags
            limit: Maximum number of results

        Returns:
            DataFrame with experiment info
        """
        conn = sqlite3.connect(self.db_path)

        query = "SELECT * FROM experiments WHERE 1=1"
        params = []

        if status:
            query += " AND status = ?"
            params.append(status)

        if tags:
            # Filter by tags (stored as JSON array)
            for tag in tags:
                query += " AND tags LIKE ?"
                params.append(f'%"{tag}"%')

        query += " ORDER BY created_at DESC"

        if limit:
            query += f" LIMIT {limit}"

        df = pd.read_sql_query(query, conn, params=params if params else None)
        conn.close()

        return df

    def get_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get experiment details.

        Args:
            experiment_id: Experiment ID

        Returns:
            Dictionary with experiment info
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,))

        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = dict(row)

        # Get config
        cursor.execute(
            "SELECT config_key, config_value FROM configs WHERE experiment_id = ?", (experiment_id,)
        )
        config = {row["config_key"]: json.loads(row["config_value"]) for row in cursor.fetchall()}
        experiment["config"] = config

        # Get metric summary
        cursor.execute(
            """
            SELECT metric_name, 
                   COUNT(*) as count,
                   MIN(metric_value) as min_value,
                   MAX(metric_value) as max_value,
                   AVG(metric_value) as avg_value
            FROM metrics 
            WHERE experiment_id = ?
            GROUP BY metric_name
        """,
            (experiment_id,),
        )

        metrics_summary = {row["metric_name"]: dict(row) for row in cursor.fetchall()}
        experiment["metrics_summary"] = metrics_summary

        conn.close()
        return experiment

    def compare_experiments(
        self, experiment_ids: List[str], metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple experiments.

        Args:
            experiment_ids: List of experiment IDs to compare
            metrics: Specific metrics to compare (if None, compare all)

        Returns:
            DataFrame with comparison
        """
        conn = sqlite3.connect(self.db_path)

        # Get final metrics for each experiment
        comparisons = []

        for exp_id in experiment_ids:
            # Get experiment info
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name, status, created_at FROM experiments WHERE experiment_id = ?",
                (exp_id,),
            )
            exp_info = cursor.fetchone()

            if not exp_info:
                continue

            comparison = {
                "experiment_id": exp_id,
                "name": exp_info[0],
                "status": exp_info[1],
                "created_at": exp_info[2],
            }

            # Get final metrics (last epoch for each metric)
            if metrics:
                metric_filter = " AND metric_name IN ({})".format(",".join(["?" for _ in metrics]))
                params = [exp_id] + metrics
            else:
                metric_filter = ""
                params = [exp_id]

            query = f"""
                SELECT metric_name, metric_value, MAX(epoch) as epoch
                FROM metrics
                WHERE experiment_id = ? {metric_filter}
                GROUP BY metric_name
            """

            cursor.execute(query, params)

            for row in cursor.fetchall():
                metric_name = row[0]
                metric_value = row[1]
                comparison[metric_name] = metric_value

            comparisons.append(comparison)

        conn.close()

        df = pd.DataFrame(comparisons)
        return df

    def get_metric_history(self, experiment_id: str, metric_name: str) -> pd.DataFrame:
        """
        Get metric history for plotting.

        Args:
            experiment_id: Experiment ID
            metric_name: Metric name

        Returns:
            DataFrame with metric history
        """
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT epoch, step, metric_value, metric_type, timestamp
            FROM metrics
            WHERE experiment_id = ? AND metric_name = ?
            ORDER BY step
        """

        df = pd.read_sql_query(query, conn, params=(experiment_id, metric_name))
        conn.close()

        return df

    def delete_experiment(self, experiment_id: str):
        """
        Delete an experiment.

        Args:
            experiment_id: Experiment ID to delete
        """
        # Get experiment path
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT path FROM experiments WHERE experiment_id = ?", (experiment_id,))

        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Experiment {experiment_id} not found")

        exp_path = Path(row[0])

        # Delete from database
        cursor.execute("DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,))
        cursor.execute("DELETE FROM metrics WHERE experiment_id = ?", (experiment_id,))
        cursor.execute("DELETE FROM configs WHERE experiment_id = ?", (experiment_id,))
        cursor.execute("DELETE FROM artifacts WHERE experiment_id = ?", (experiment_id,))

        conn.commit()
        conn.close()

        # Delete directory
        import shutil

        if exp_path.exists():
            shutil.rmtree(exp_path)

    def search_experiments(
        self,
        config_filter: Optional[Dict[str, Any]] = None,
        metric_filter: Optional[Dict[str, tuple]] = None,
    ) -> pd.DataFrame:
        """
        Search experiments by config or metrics.

        Args:
            config_filter: Dict of config_key: value to filter
            metric_filter: Dict of metric_name: (operator, value)
                          e.g., {'val_loss': ('<=', 0.5)}

        Returns:
            DataFrame with matching experiments
        """
        conn = sqlite3.connect(self.db_path)

        # Start with all experiments
        query = "SELECT DISTINCT e.* FROM experiments e"
        joins = []
        where_clauses = []
        params = []

        # Filter by config
        if config_filter:
            for i, (key, value) in enumerate(config_filter.items()):
                alias = f"c{i}"
                joins.append(f"JOIN configs {alias} ON e.experiment_id = {alias}.experiment_id")
                where_clauses.append(f"({alias}.config_key = ? AND {alias}.config_value = ?)")
                params.extend([key, json.dumps(value)])

        # Filter by metrics
        if metric_filter:
            for i, (metric_name, (operator, value)) in enumerate(metric_filter.items()):
                alias = f"m{i}"
                joins.append(f"JOIN metrics {alias} ON e.experiment_id = {alias}.experiment_id")
                where_clauses.append(
                    f"({alias}.metric_name = ? AND {alias}.metric_value {operator} ?)"
                )
                params.extend([metric_name, value])

        # Build query
        if joins:
            query += " " + " ".join(joins)

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        query += " ORDER BY e.created_at DESC"

        df = pd.read_sql_query(query, conn, params=params if params else None)
        conn.close()

        return df

    def print_summary(self):
        """Print summary of all experiments"""
        df = self.list_experiments()

        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)

        print(f"\nTotal Experiments: {len(df)}")

        if len(df) > 0:
            # Status breakdown
            status_counts = df["status"].value_counts()
            print(f"\nBy Status:")
            for status, count in status_counts.items():
                print(f"  {status}: {count}")

            # Recent experiments
            print(f"\nRecent Experiments:")
            recent = df.head(5)[["experiment_id", "name", "status", "created_at"]]
            print(tabulate(recent, headers="keys", tablefmt="grid", showindex=False))

        print("=" * 60 + "\n")
