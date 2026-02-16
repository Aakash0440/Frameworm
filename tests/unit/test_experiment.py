"""Tests for experiment tracking"""

import pytest
import tempfile
import shutil
from pathlib import Path
from experiment import Experiment, ExperimentManager


class TestExperiment:
    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_experiment_creation(self):
        exp = Experiment(
            "test",
            config={'lr': 0.001},
            root_dir=str(self.temp_dir)
        )
        
        assert exp.name == "test"
        assert exp.status == "pending"
        assert exp.exp_dir.exists()
    
    def test_experiment_context_manager(self):
        exp = Experiment("test", root_dir=str(self.temp_dir))
        
        with exp:
            assert exp.status == "running"
            exp.log_metric("loss", 0.5, step=0)
        
        assert exp.status == "completed"
    
    def test_metric_logging(self):
        exp = Experiment("test", root_dir=str(self.temp_dir))
        
        with exp:
            for i in range(10):
                exp.log_metric("loss", 1.0/(i+1), step=i)
        
        metrics = exp.get_metrics("loss")
        assert len(metrics) == 10
    
    def test_artifact_logging(self):
        exp = Experiment("test", root_dir=str(self.temp_dir))
        
        # Create temp file
        temp_file = self.temp_dir / "test.txt"
        temp_file.write_text("test")
        
        with exp:
            exp.log_artifact(str(temp_file), artifact_type="test")
        
        # Check artifact copied
        artifact_path = exp.artifact_dir / "test.txt"
        assert artifact_path.exists()


class TestExperimentManager:
    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_list_experiments(self):
        # Create experiments
        for i in range(3):
            exp = Experiment(f"test-{i}", root_dir=str(self.temp_dir))
            with exp:
                exp.log_metric("loss", float(i), step=0)
        
        manager = ExperimentManager(str(self.temp_dir))
        df = manager.list_experiments()
        
        assert len(df) == 3
    
    def test_compare_experiments(self):
        exp_ids = []
        
        for i in range(3):
            exp = Experiment(f"test-{i}", root_dir=str(self.temp_dir))
            with exp:
                exp.log_metric("loss", float(i), step=0)
            exp_ids.append(exp.experiment_id)
        
        manager = ExperimentManager(str(self.temp_dir))
        comparison = manager.compare_experiments(exp_ids)
        
        assert len(comparison) == 3
        assert 'loss' in comparison.columns