# API Reference

## frameworm.core

### Config
```python
class Config:
    def __init__(self, config_path: Optional[Union[str, Path]] = None)
    def load(self, config_path: Union[str, Path]) -> 'Config'
    def freeze(self)
    def to_dict(self) -> Dict[str, Any]
    def dump(self, output_path: Union[str, Path])
    def validate(self, schema: Type[BaseModel]) -> BaseModel
    @staticmethod
    def from_cli_args(base_config, overrides: List[str]) -> 'Config'
    @classmethod
    def from_template(cls, template_name: str, **overrides) -> 'Config'
    def diff(self, other: 'Config') -> Dict[str, tuple]
    def to_json(self, output_path: Optional[Union[str, Path]] = None) -> str
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'Config'
```

See [Config Documentation](user_guide/configuration.md) for details.

### Types

See [Type System Documentation](architecture/type_system.md) for full reference.

## frameworm.models

### BaseModel
```python
class BaseModel(nn.Module, ABC):
    def __init__(self, config: Config)
    @abstractmethod
    def forward(self, *args, **kwargs) -> Any
    def to_device(self, device: Optional[str] = None)
    def init_weights(self, init_fn: Optional[Callable] = None)
    def freeze(self, module_names: Optional[List[str]] = None)
    def unfreeze(self, module_names: Optional[List[str]] = None)
    def summary(self, input_size: Optional[tuple] = None, detailed: bool = False)
    def save(self, path: str, **extra_metadata)
    def load(self, path: str, strict: bool = True)
```

## frameworm.pipelines

### BasePipeline
```python
class BasePipeline(ABC):
    def __init__(self, config)
    @abstractmethod
    def run(self, *args, **kwargs) -> Any
    def add_step(self, name: str, fn: Callable, depends_on: Optional[List[str]] = None)
    def execute_steps(self, progress_callback: Optional[Callable] = None)
    def save_state(self, path: str)
    def load_state(self, path: str)
```

## frameworm.trainers

### BaseTrainer
```python
class BaseTrainer(ABC):
    def __init__(self, model, config: Config)
    @abstractmethod
    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]
    @abstractmethod
    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]
    def fit(self, train_loader, val_loader=None, epochs: Optional[int] = None)
    def save_checkpoint(self, path: str, optimizer=None, scheduler=None, **extra_info)
    def load_checkpoint(self, path: str, optimizer=None, scheduler=None)
```

---

*Full API documentation will be auto-generated from docstrings in Week 4*