"""Tests for ModelRegistry lifecycle management."""

import sys
import tempfile
import os
sys.path.insert(0, ".")


def test_register_and_retrieve():
    from deploy.core.registry import ModelRegistry
    with tempfile.TemporaryDirectory() as d:
        reg = ModelRegistry(db_path=os.path.join(d, "registry.db"))
        reg.register("fraud", "v1.0", "dcgan", "/path/to/model.pt", "staging")
        versions = reg.get_by_name("fraud")
        assert len(versions) == 1
        assert versions[0]["version"] == "v1.0"
        assert versions[0]["stage"] == "staging"

def test_promote_to_production():
    from deploy.core.registry import ModelRegistry
    with tempfile.TemporaryDirectory() as d:
        reg = ModelRegistry(db_path=os.path.join(d, "r.db"))
        reg.register("model", "v1.0", "vae", "/m.pt", "staging")
        reg.promote("model", "v1.0", "production")
        current = reg.get_current_production("model")
        assert current is not None
        assert current["version"] == "v1.0"
        assert current["stage"] == "production"

def test_full_lifecycle():
    from deploy.core.registry import ModelRegistry
    with tempfile.TemporaryDirectory() as d:
        reg = ModelRegistry(db_path=os.path.join(d, "r.db"))
        reg.register("m", "v1.0", "ddpm", "/a.pt", "dev")
        reg.promote("m", "v1.0", "staging")
        reg.promote("m", "v1.0", "production")
        reg.register("m", "v2.0", "ddpm", "/b.pt", "staging")
        reg.promote("m", "v2.0", "production")
        reg.demote("m", "v1.0", "archived")
        history = reg.get_production_history("m")
        versions_in_history = [h["version"] for h in history]
        assert "v1.0" in versions_in_history
        assert "v2.0" in versions_in_history

def test_list_all_doesnt_crash():
    from deploy.core.registry import ModelRegistry
    with tempfile.TemporaryDirectory() as d:
        reg = ModelRegistry(db_path=os.path.join(d, "r.db"))
        reg.register("a", "v1", "vae",   "/a.pt", "production")
        reg.register("b", "v1", "dcgan", "/b.pt", "staging")
        reg.list_all()   # just check it doesn't throw