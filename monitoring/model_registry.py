"""
Model version registry with lineage tracking.

Manages model versions, promotions, and metadata.
Supports: dev → staging → production lifecycle.

Example:
    >>> registry = ModelRegistry('models/')
    >>> 
    >>> # Register a new model version
    >>> version = registry.register(
    ...     model_path='checkpoints/best.pt',
    ...     name='vae-mnist',
    ...     version='1.2.0',
    ...     metrics={'fid': 18.3, 'is': 9.4},
    ...     config={'lr': 0.001, 'epochs': 100}
    ... )
    >>> 
    >>> # Promote to production
    >>> registry.promote('vae-mnist', '1.2.0', stage='production')
    >>> 
    >>> # Get production model
    >>> prod = registry.get_production('vae-mnist')
"""

import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any


class ModelVersion:
    """Represents a single registered model version"""
    
    def __init__(
        self,
        name: str,
        version: str,
        path: str,
        stage: str = 'dev',
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict] = None,
        description: str = '',
        tags: Optional[List[str]] = None
    ):
        self.name = name
        self.version = version
        self.path = path
        self.stage = stage
        self.metrics = metrics or {}
        self.config = config or {}
        self.description = description
        self.tags = tags or []
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.file_hash = None
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'version': self.version,
            'path': self.path,
            'stage': self.stage,
            'metrics': self.metrics,
            'config': self.config,
            'description': self.description,
            'tags': self.tags,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'file_hash': self.file_hash
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ModelVersion':
        mv = cls(
            name=d['name'], version=d['version'], path=d['path'],
            stage=d.get('stage', 'dev'), metrics=d.get('metrics', {}),
            config=d.get('config', {}), description=d.get('description', ''),
            tags=d.get('tags', [])
        )
        mv.created_at = d.get('created_at', mv.created_at)
        mv.updated_at = d.get('updated_at', mv.updated_at)
        mv.file_hash = d.get('file_hash')
        return mv
    
    def __repr__(self):
        return f"ModelVersion({self.name}=={self.version}, stage={self.stage})"


class ModelRegistry:
    """
    Local model registry for versioning and lifecycle management.
    
    Stores model metadata in JSON. Copies model files to managed storage.
    
    Args:
        registry_dir: Directory to store registry data and model files
        
    Stages:
        dev       → Initial registration
        staging   → Ready for integration testing
        production→ Live serving
        archived  → Replaced, kept for rollback
        
    Example:
        >>> registry = ModelRegistry('model_registry/')
        >>> 
        >>> # Register new version
        >>> mv = registry.register(
        ...     model_path='checkpoints/epoch_100.pt',
        ...     name='vae-mnist',
        ...     version='1.0.0',
        ...     metrics={'fid': 22.1}
        ... )
        >>> print(mv)
        ModelVersion(vae-mnist==1.0.0, stage=dev)
        >>> 
        >>> # Promote through pipeline
        >>> registry.promote('vae-mnist', '1.0.0', 'staging')
        >>> registry.promote('vae-mnist', '1.0.0', 'production')
        >>> 
        >>> # Get production model path
        >>> prod = registry.get_production('vae-mnist')
        >>> model = torch.load(prod.path)
    """
    
    VALID_STAGES = ['dev', 'staging', 'production', 'archived']
    STAGE_ORDER = {'dev': 0, 'staging': 1, 'production': 2, 'archived': -1}
    
    def __init__(self, registry_dir: str = 'model_registry'):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.registry_dir / 'models'
        self.models_dir.mkdir(exist_ok=True)
        self.index_path = self.registry_dir / 'registry.json'
        self._index = self._load_index()
    
    def _load_index(self) -> Dict:
        if self.index_path.exists():
            with open(self.index_path) as f:
                return json.load(f)
        return {'versions': {}}
    
    def _save_index(self):
        with open(self.index_path, 'w') as f:
            json.dump(self._index, f, indent=2)
    
    def _compute_hash(self, path: str) -> str:
        """Compute SHA256 hash of model file"""
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()[:12]
    
    def _get_key(self, name: str, version: str) -> str:
        return f"{name}/{version}"
    
    def register(
        self,
        model_path: str,
        name: str,
        version: str,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict] = None,
        description: str = '',
        tags: Optional[List[str]] = None,
        copy_file: bool = True
    ) -> ModelVersion:
        """
        Register a model version.
        
        Args:
            model_path: Path to model file
            name: Model name (e.g., 'vae-mnist')
            version: Semantic version (e.g., '1.2.3')
            metrics: Evaluation metrics dict
            config: Training config dict
            description: Human-readable description
            tags: List of tags
            copy_file: Copy model file to registry storage
            
        Returns:
            ModelVersion object
        """
        key = self._get_key(name, version)
        
        if key in self._index['versions']:
            raise ValueError(f"Version {name}=={version} already registered")
        
        # Copy model file to registry
        stored_path = model_path
        if copy_file and Path(model_path).exists():
            dest = self.models_dir / name / version
            dest.mkdir(parents=True, exist_ok=True)
            dest_file = dest / Path(model_path).name
            shutil.copy2(model_path, dest_file)
            stored_path = str(dest_file)
        
        # Create version record
        mv = ModelVersion(
            name=name,
            version=version,
            path=stored_path,
            stage='dev',
            metrics=metrics or {},
            config=config or {},
            description=description,
            tags=tags or []
        )
        
        if Path(stored_path).exists():
            mv.file_hash = self._compute_hash(stored_path)
        
        self._index['versions'][key] = mv.to_dict()
        self._save_index()
        
        print(f"✓ Registered: {name}=={version} (stage=dev)")
        return mv
    
    def promote(
        self,
        name: str,
        version: str,
        stage: str
    ) -> ModelVersion:
        """
        Promote a model version to a new stage.
        
        Validates stage progression (dev→staging→production).
        Automatically archives previous production model.
        
        Args:
            name: Model name
            version: Version string
            stage: Target stage
        """
        if stage not in self.VALID_STAGES:
            raise ValueError(f"Invalid stage: {stage}. Use: {self.VALID_STAGES}")
        
        key = self._get_key(name, version)
        if key not in self._index['versions']:
            raise ValueError(f"Version {name}=={version} not found")
        
        mv_dict = self._index['versions'][key]
        current_stage = mv_dict['stage']
        
        # Validate progression
        if stage != 'archived':
            current_order = self.STAGE_ORDER[current_stage]
            target_order = self.STAGE_ORDER[stage]
            if target_order != current_order + 1:
                raise ValueError(
                    f"Invalid promotion: {current_stage} → {stage}. "
                    f"Must follow dev→staging→production."
                )
        
        # Archive existing production model
        if stage == 'production':
            for k, v in self._index['versions'].items():
                if v['name'] == name and v['stage'] == 'production':
                    v['stage'] = 'archived'
                    v['updated_at'] = datetime.now().isoformat()
                    print(f"  Archived: {name}=={v['version']}")
        
        mv_dict['stage'] = stage
        mv_dict['updated_at'] = datetime.now().isoformat()
        self._save_index()
        
        print(f"✓ Promoted: {name}=={version} → {stage}")
        return ModelVersion.from_dict(mv_dict)
    
    def get_production(self, name: str) -> Optional[ModelVersion]:
        """Get the current production model version"""
        for v in self._index['versions'].values():
            if v['name'] == name and v['stage'] == 'production':
                return ModelVersion.from_dict(v)
        return None
    
    def list_versions(self, name: str) -> List[ModelVersion]:
        """List all versions of a model"""
        versions = [
            ModelVersion.from_dict(v)
            for v in self._index['versions'].values()
            if v['name'] == name
        ]
        return sorted(versions, key=lambda v: v.created_at)
    
    def compare(self, name: str, v1: str, v2: str) -> Dict:
        """Compare two model versions side-by-side"""
        key1, key2 = self._get_key(name, v1), self._get_key(name, v2)
        
        mv1 = ModelVersion.from_dict(self._index['versions'][key1])
        mv2 = ModelVersion.from_dict(self._index['versions'][key2])
        
        comparison = {'v1': v1, 'v2': v2, 'metrics': {}, 'winner': {}}
        
        all_metrics = set(mv1.metrics) | set(mv2.metrics)
        for metric in all_metrics:
            val1 = mv1.metrics.get(metric)
            val2 = mv2.metrics.get(metric)
            delta = (val2 - val1) if (val1 is not None and val2 is not None) else None
            comparison['metrics'][metric] = {
                'v1': val1, 'v2': val2,
                'delta': delta,
                'delta_pct': f"{delta / val1 * 100:+.1f}%" if (delta and val1) else 'N/A'
            }
        
        return comparison
    
    def print_registry(self, name: Optional[str] = None):
        """Pretty-print registry contents"""
        print(f"\n📦 Model Registry")
        print("─" * 70)
        print(f"{'Name':<20} {'Version':<12} {'Stage':<12} {'Metrics'}")
        print("─" * 70)
        
        for v in self._index['versions'].values():
            if name and v['name'] != name:
                continue
            metrics_str = ', '.join(f"{k}={val:.3f}" for k, val in list(v['metrics'].items())[:2])
            stage_color = {'dev': '🟡', 'staging': '🔵', 'production': '🟢', 'archived': '⚫'}
            icon = stage_color.get(v['stage'], '')
            print(f"{v['name']:<20} {v['version']:<12} {icon} {v['stage']:<10} {metrics_str}")
        print("─" * 70)