-- Experiments table
CREATE TABLE experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'running',  -- running, completed, failed, stopped
    git_hash TEXT,
    git_dirty BOOLEAN,
    path TEXT NOT NULL,
    tags TEXT  -- JSON array
);

-- Metrics table
CREATE TABLE metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    epoch INTEGER,
    step INTEGER,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    metric_type TEXT,  -- train, val, test
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

-- Config table
CREATE TABLE configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    config_key TEXT NOT NULL,
    config_value TEXT NOT NULL,  -- JSON
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

-- Artifacts table
CREATE TABLE artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,  -- checkpoint, image, log, etc.
    artifact_path TEXT NOT NULL,
    artifact_name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

-- Indexes
CREATE INDEX idx_experiments_status ON experiments(status);
CREATE INDEX idx_experiments_created ON experiments(created_at);
CREATE INDEX idx_metrics_experiment ON metrics(experiment_id);
CREATE INDEX idx_metrics_name ON metrics(metric_name);