"""Comprehensive tests for config system"""
import sys
import pytest
import yaml
from pathlib import Path
import tempfile
import os
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.config import Config, ConfigNode


class TestConfigNode:
    """Test ConfigNode functionality"""
    
    def test_dot_notation_access(self):
        """Should access nested values via dot notation"""
        node = ConfigNode({'model': {'name': 'gan', 'dim': 128}})
        assert node.model.name == 'gan'
        assert node.model.dim == 128
    
    def test_dict_access(self):
        """Should work with dict-style access"""
        node = ConfigNode({'key': 'value'})
        assert node['key'] == 'value'
    
    def test_nested_lists(self):
        """Should handle lists of dicts"""
        node = ConfigNode({
            'layers': [
                {'type': 'conv', 'filters': 64},
                {'type': 'conv', 'filters': 128}
            ]
        })
        assert node.layers[0].type == 'conv'
        assert node.layers[1].filters == 128
    
    def test_to_dict(self):
        """Should convert back to regular dict"""
        node = ConfigNode({'model': {'name': 'gan'}})
        d = node.to_dict()
        assert isinstance(d, dict)
        assert isinstance(d['model'], dict)
        assert d['model']['name'] == 'gan'
    
    def test_attribute_error(self):
        """Should raise AttributeError for missing keys"""
        node = ConfigNode({'key': 'value'})
        with pytest.raises(AttributeError):
            _ = node.missing_key


class TestConfig:
    """Test Config functionality"""
    
    def test_load_simple_config(self, tmp_path):
        """Should load basic YAML config"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
model:
  name: dcgan
  latent_dim: 128
training:
  epochs: 100
""")
        
        cfg = Config(config_file)
        assert cfg.model.name == 'dcgan'
        assert cfg.model.latent_dim == 128
        assert cfg.training.epochs == 100
    
    def test_single_inheritance(self, tmp_path):
        """Should merge base config with child config"""
        # Create base config
        base_config = tmp_path / "base.yaml"
        base_config.write_text("""
training:
  epochs: 100
  batch_size: 32
optimizer:
  type: adam
""")
        
        # Create child config
        child_config = tmp_path / "child.yaml"
        child_config.write_text(f"""
_base_: base.yaml
training:
  epochs: 200
model:
  name: gan
""")
        
        cfg = Config(child_config)
        
        # Should inherit from base
        assert cfg.training.batch_size == 32
        assert cfg.optimizer.type == 'adam'
        
        # Should override base
        assert cfg.training.epochs == 200
        
        # Should add new values
        assert cfg.model.name == 'gan'
    
    def test_file_not_found(self):
        """Should raise FileNotFoundError for missing file"""
        with pytest.raises(FileNotFoundError):
            Config('nonexistent.yaml')
    
    def test_freeze(self, tmp_path):
        """Should prevent loading after freeze"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("key: value")
        
        cfg = Config(config_file)
        cfg.freeze()
        assert cfg.is_frozen()
        
        with pytest.raises(RuntimeError):
            cfg.load(config_file)
    
    def test_dump(self, tmp_path):
        """Should save merged config to file"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
model:
  name: test
training:
  epochs: 100
""")
        
        cfg = Config(config_file)
        
        output_file = tmp_path / "output.yaml"
        cfg.dump(output_file)
        
        # Load dumped file
        cfg2 = Config(output_file)
        assert cfg2.model.name == 'test'
        assert cfg2.training.epochs == 100


# Run tests immediately
if __name__ == '__main__':
    pytest.main([__file__, '-v'])