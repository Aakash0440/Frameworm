"""
Database integrations for experiment storage.

Alternative to SQLite for multi-user scenarios.

Example:
    >>> from frameworm.integrations import PostgresBackend
    >>> 
    >>> db = PostgresBackend(
    ...     host='localhost',
    ...     database='frameworm_experiments'
    ... )
    >>> manager = ExperimentManager(backend=db)
"""

from typing import Optional, Dict, List, Any
from abc import ABC, abstractmethod


class DatabaseBackend(ABC):
    """Abstract base for database backends"""
    
    @abstractmethod
    def insert_experiment(self, experiment_data: Dict) -> str:
        """Insert experiment, return ID"""
        pass
    
    @abstractmethod
    def get_experiment(self, experiment_id: str) -> Dict:
        """Get experiment by ID"""
        pass
    
    @abstractmethod
    def list_experiments(self, filters: Optional[Dict] = None) -> List[Dict]:
        """List experiments with optional filters"""
        pass
    
    @abstractmethod
    def update_experiment(self, experiment_id: str, updates: Dict):
        """Update experiment"""
        pass


class PostgresBackend(DatabaseBackend):
    """
    PostgreSQL backend for experiment tracking.
    
    Better than SQLite for multi-user production environments.
    
    Args:
        host: Database host
        database: Database name
        user: Username
        password: Password
        port: Port (default: 5432)
        
    Example:
        >>> backend = PostgresBackend(
        ...     host='localhost',
        ...     database='ml_experiments',
        ...     user='mluser',
        ...     password='secret'
        ... )
        >>> # Use with ExperimentManager
        >>> from frameworm.experiment import ExperimentManager
        >>> manager = ExperimentManager(backend=backend)
    """
    
    def __init__(
        self,
        host: str,
        database: str,
        user: str,
        password: str,
        port: int = 5432
    ):
        try:
            import psycopg2
            from psycopg2 import pool
        except ImportError:
            raise ImportError("psycopg2 not installed. Install: pip install psycopg2-binary")
        
        self.pool = pool.SimpleConnectionPool(
            1, 10,  # min/max connections
            host=host,
            database=database,
            user=user,
            password=password,
            port=port
        )
        
        self._create_tables()
    
    def _create_tables(self):
        """Create tables if they don't exist"""
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS experiments (
                        experiment_id VARCHAR(255) PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        status VARCHAR(50),
                        created_at TIMESTAMP,
                        config JSONB,
                        metrics JSONB,
                        tags TEXT[]
                    )
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_name ON experiments(name);
                    CREATE INDEX IF NOT EXISTS idx_status ON experiments(status);
                    CREATE INDEX IF NOT EXISTS idx_tags ON experiments USING GIN(tags);
                """)
                
                conn.commit()
        finally:
            self.pool.putconn(conn)
    
    def insert_experiment(self, experiment_data: Dict) -> str:
        """Insert new experiment"""
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                import json
                cur.execute("""
                    INSERT INTO experiments (experiment_id, name, status, created_at, config, metrics, tags)
                    VALUES (%(experiment_id)s, %(name)s, %(status)s, %(created_at)s, %(config)s, %(metrics)s, %(tags)s)
                """, {
                    'experiment_id': experiment_data['experiment_id'],
                    'name': experiment_data['name'],
                    'status': experiment_data.get('status', 'running'),
                    'created_at': experiment_data['created_at'],
                    'config': json.dumps(experiment_data.get('config', {})),
                    'metrics': json.dumps(experiment_data.get('metrics', {})),
                    'tags': experiment_data.get('tags', [])
                })
                conn.commit()
            return experiment_data['experiment_id']
        finally:
            self.pool.putconn(conn)
    
    def get_experiment(self, experiment_id: str) -> Dict:
        """Get experiment by ID"""
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM experiments WHERE experiment_id = %s",
                    (experiment_id,)
                )
                row = cur.fetchone()
                if not row:
                    raise ValueError(f"Experiment {experiment_id} not found")
                
                import json
                return {
                    'experiment_id': row[0],
                    'name': row[1],
                    'status': row[2],
                    'created_at': str(row[3]),
                    'config': json.loads(row[4]) if row[4] else {},
                    'metrics': json.loads(row[5]) if row[5] else {},
                    'tags': row[6] or []
                }
        finally:
            self.pool.putconn(conn)
    
    def list_experiments(self, filters: Optional[Dict] = None) -> List[Dict]:
        """List experiments"""
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                query = "SELECT * FROM experiments"
                params = []
                
                if filters:
                    conditions = []
                    if 'status' in filters:
                        conditions.append("status = %s")
                        params.append(filters['status'])
                    if 'name' in filters:
                        conditions.append("name LIKE %s")
                        params.append(f"%{filters['name']}%")
                    
                    if conditions:
                        query += " WHERE " + " AND ".join(conditions)
                
                query += " ORDER BY created_at DESC"
                
                cur.execute(query, params)
                rows = cur.fetchall()
                
                import json
                return [
                    {
                        'experiment_id': row[0],
                        'name': row[1],
                        'status': row[2],
                        'created_at': str(row[3]),
                        'config': json.loads(row[4]) if row[4] else {},
                        'metrics': json.loads(row[5]) if row[5] else {},
                        'tags': row[6] or []
                    }
                    for row in rows
                ]
        finally:
            self.pool.putconn(conn)
    
    def update_experiment(self, experiment_id: str, updates: Dict):
        """Update experiment"""
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                import json
                set_clauses = []
                params = []
                
                if 'status' in updates:
                    set_clauses.append("status = %s")
                    params.append(updates['status'])
                if 'metrics' in updates:
                    set_clauses.append("metrics = %s")
                    params.append(json.dumps(updates['metrics']))
                
                if set_clauses:
                    query = f"UPDATE experiments SET {', '.join(set_clauses)} WHERE experiment_id = %s"
                    params.append(experiment_id)
                    cur.execute(query, params)
                    conn.commit()
        finally:
            self.pool.putconn(conn)