"""
Plugin loader and discovery system for FRAMEWORM.

Discovers plugins from three sources (in order):
  1. ~/.frameworm/plugins/       — user-global plugins
  2. ./frameworm_plugins/        — project-local plugins
  3. Installed packages with     — pip-installed plugins
     names starting with
     "frameworm_"

Each plugin directory must contain a ``plugin.yaml`` describing the plugin.
Each plugin package must expose a ``frameworm_plugin`` entry point group.

plugin.yaml schema:
    name:         my-plugin           # required, unique identifier
    version:      1.0.0               # required
    description:  Does something cool  # optional
    author:       Your Name           # optional
    entry_point:  my_module:register  # required — "module:function"
    dependencies:                     # optional list of pip packages
      - numpy>=1.20
      - torch>=1.10
    frameworm_min_version: 1.0.0      # optional

The ``entry_point`` function receives the HookRegistry and ModelRegistry
and should register hooks/models:

    # my_module.py
    def register(hook_registry, model_registry):
        @hook_registry.register("on_epoch_end")
        def log(epoch, metrics, **kwargs):
            ...
        model_registry.register("my-model", MyModel)

Usage:
    from plugins.loader import PluginLoader
    from plugins.hooks import HookRegistry
    from core.registry import _MODEL_REGISTRY

    loader = PluginLoader(hook_registry=HookRegistry())
    result = loader.discover()
    print(result)  # {'loaded': [...], 'failed': [...], 'skipped': [...]}
"""

import importlib
import importlib.util
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plugin metadata
# ---------------------------------------------------------------------------

@dataclass
class PluginMeta:
    """Parsed metadata from plugin.yaml."""
    name: str
    version: str
    entry_point: str                       # "module:function"
    description: str = ""
    author: str = ""
    dependencies: List[str] = field(default_factory=list)
    frameworm_min_version: Optional[str] = None
    source_path: Optional[Path] = None    # directory containing plugin.yaml

    def __str__(self):
        return f"{self.name} v{self.version}"


@dataclass
class LoadResult:
    """Result of a single plugin load attempt."""
    meta: PluginMeta
    success: bool
    error: Optional[str] = None

    def __str__(self):
        status = "✓" if self.success else f"✗ {self.error}"
        return f"{self.meta.name} v{self.meta.version}: {status}"


# ---------------------------------------------------------------------------
# PluginLoader
# ---------------------------------------------------------------------------

class PluginLoader:
    """
    Discovers and loads FRAMEWORM plugins from configured locations.

    Example:
        >>> from plugins.loader import PluginLoader
        >>> from plugins.hooks import HookRegistry

        >>> loader = PluginLoader(hook_registry=HookRegistry())
        >>> result = loader.discover()
        >>> print(f"Loaded {len(result['loaded'])} plugins")

    Attributes:
        search_paths: List of directories to scan for plugins.
        hook_registry: HookRegistry instance passed to each plugin's register().
        model_registry: Optional registry passed to each plugin's register().
    """

    def __init__(
        self,
        hook_registry=None,
        model_registry=None,
        extra_paths: Optional[List[Path]] = None,
        ignore_missing_deps: bool = False,
    ):
        """
        Args:
            hook_registry:       HookRegistry to pass to plugin register().
            model_registry:      ModelRegistry to pass to plugin register().
            extra_paths:         Additional directories to scan.
            ignore_missing_deps: If True, load plugins even if pip dependencies
                                 are not installed (may cause ImportErrors later).
        """
        self.hook_registry = hook_registry
        self.model_registry = model_registry
        self.ignore_missing_deps = ignore_missing_deps

        # Build default search paths
        self.search_paths: List[Path] = []
        self._add_default_paths()
        if extra_paths:
            for p in extra_paths:
                self.add_path(p)

        # Track what we've loaded to prevent double-loading
        self._loaded: Dict[str, PluginMeta] = {}  # name → meta
        self._failed: Dict[str, str] = {}          # name → error message

    # ------------------------------------------------------------------
    # Path management
    # ------------------------------------------------------------------

    def add_path(self, path) -> None:
        """Add a directory to the plugin search path."""
        resolved = Path(path).resolve()
        if resolved not in self.search_paths:
            self.search_paths.append(resolved)

    def remove_path(self, path) -> None:
        """Remove a directory from the search path."""
        resolved = Path(path).resolve()
        if resolved in self.search_paths:
            self.search_paths.remove(resolved)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(self, force: bool = False) -> Dict[str, List]:
        """
        Scan all search paths and load any discovered plugins.

        Args:
            force: If True, reload plugins that were already loaded.

        Returns:
            Dict with keys:
              'loaded'  — list of LoadResult (success=True)
              'failed'  — list of LoadResult (success=False)
              'skipped' — list of plugin names skipped (already loaded)
        """
        loaded = []
        failed = []
        skipped = []

        # Collect all plugin metadata from search paths
        candidates = self._collect_candidates()

        # Also collect from installed packages
        candidates.extend(self._collect_installed())

        for meta in candidates:
            if meta.name in self._loaded and not force:
                skipped.append(meta.name)
                continue

            result = self._load_one(meta)
            if result.success:
                self._loaded[meta.name] = meta
                loaded.append(result)
                logger.info("Loaded plugin: %s", meta)
            else:
                self._failed[meta.name] = result.error
                failed.append(result)
                logger.warning("Failed to load plugin %s: %s", meta.name, result.error)

        return {"loaded": loaded, "failed": failed, "skipped": skipped}

    def load_path(self, path, force: bool = False) -> LoadResult:
        """
        Load a single plugin from a specific directory.

        Args:
            path:  Directory containing plugin.yaml.
            force: Reload even if already loaded.

        Returns:
            LoadResult
        """
        meta = self._read_meta(Path(path))
        if meta is None:
            dummy = PluginMeta(name=str(path), version="?", entry_point="")
            return LoadResult(meta=dummy, success=False, error="plugin.yaml not found or invalid")

        if meta.name in self._loaded and not force:
            return LoadResult(meta=meta, success=True, error="already loaded (use force=True to reload)")

        result = self._load_one(meta)
        if result.success:
            self._loaded[meta.name] = meta
        return result

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_loaded(self) -> List[PluginMeta]:
        """Return metadata for all successfully loaded plugins."""
        return list(self._loaded.values())

    def list_failed(self) -> Dict[str, str]:
        """Return {name: error_message} for all plugins that failed to load."""
        return dict(self._failed)

    def is_loaded(self, name: str) -> bool:
        """Return True if a plugin with the given name is loaded."""
        return name in self._loaded

    def summary(self) -> None:
        """Print a human-readable summary of loaded/failed plugins."""
        print(f"PluginLoader — search paths:")
        for p in self.search_paths:
            exists = "✓" if p.exists() else "✗ (not found)"
            print(f"  {p}  {exists}")

        print(f"\nLoaded ({len(self._loaded)}):")
        for meta in self._loaded.values():
            print(f"  ✓ {meta.name} v{meta.version}  — {meta.description or '(no description)'}")

        if self._failed:
            print(f"\nFailed ({len(self._failed)}):")
            for name, error in self._failed.items():
                print(f"  ✗ {name}: {error}")

    # ------------------------------------------------------------------
    # CLI-style helpers (called by frameworm CLI)
    # ------------------------------------------------------------------

    def create_template(self, name: str, output_dir: Optional[Path] = None) -> Path:
        """
        Generate a plugin template directory.

        Args:
            name:       Plugin name (used in plugin.yaml and file names).
            output_dir: Where to create the template. Defaults to ./frameworm_plugins/<name>.

        Returns:
            Path to the created directory.
        """
        if output_dir is None:
            output_dir = Path.cwd() / "frameworm_plugins" / name
        output_dir.mkdir(parents=True, exist_ok=True)

        # plugin.yaml
        yaml_content = f"""\
name: {name}
version: 0.1.0
description: My FRAMEWORM plugin
author: Your Name
entry_point: plugin:register
dependencies: []
"""
        (output_dir / "plugin.yaml").write_text(yaml_content)

        # plugin.py
        py_content = f'''\
"""
{name} — FRAMEWORM plugin.

Generated by: frameworm plugins create {name}

Edit this file to implement your plugin.
"""


def register(hook_registry, model_registry=None):
    """Called by PluginLoader. Register hooks and/or models here."""

    @hook_registry.register("on_train_begin")
    def on_train_begin(config, **kwargs):
        print(f"[{name}] Training started!")

    @hook_registry.register("on_epoch_end")
    def on_epoch_end(epoch, metrics, **kwargs):
        print(f"[{name}] Epoch {{epoch}}: {{metrics}}")

    @hook_registry.register("on_train_end")
    def on_train_end(**kwargs):
        print(f"[{name}] Training complete!")

    # To register a custom model:
    # from my_model import MyModel
    # if model_registry is not None:
    #     model_registry.register("my-model", MyModel)
'''
        (output_dir / "plugin.py").write_text(py_content)

        # README
        readme = f"""\
# {name}

A FRAMEWORM plugin.

## Installation

Copy this directory to `~/.frameworm/plugins/{name}` or `./frameworm_plugins/{name}`.

## Usage

```python
from plugins.loader import PluginLoader
from plugins.hooks import HookRegistry

registry = HookRegistry()
loader = PluginLoader(hook_registry=registry)
loader.discover()
```
"""
        (output_dir / "README.md").write_text(readme)

        logger.info("Plugin template created at: %s", output_dir)
        return output_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_default_paths(self) -> None:
        """Add the three default search paths."""
        # 1. User-global
        user_path = Path.home() / ".frameworm" / "plugins"
        self.search_paths.append(user_path)

        # 2. Project-local
        local_path = Path.cwd() / "frameworm_plugins"
        self.search_paths.append(local_path)

        # 3. Built-in plugins directory (next to this file)
        builtin_path = Path(__file__).parent
        self.search_paths.append(builtin_path)

    def _collect_candidates(self) -> List[PluginMeta]:
        """Scan all search paths and return list of PluginMeta."""
        candidates = []
        for search_path in self.search_paths:
            if not search_path.exists():
                continue
            # Each sub-directory may be a plugin
            for item in search_path.iterdir():
                if item.is_dir():
                    meta = self._read_meta(item)
                    if meta is not None:
                        candidates.append(meta)
        return candidates

    def _collect_installed(self) -> List[PluginMeta]:
        """Discover installed packages that declare frameworm_plugin entry points."""
        candidates = []
        try:
            import importlib.metadata as importlib_metadata
            eps = importlib_metadata.entry_points(group="frameworm_plugin")
            for ep in eps:
                meta = PluginMeta(
                    name=ep.name,
                    version=ep.dist.version if ep.dist else "?",
                    entry_point=f"{ep.value}",
                    description=f"Installed package: {ep.dist.name if ep.dist else ep.name}",
                )
                candidates.append(meta)
        except Exception as exc:
            logger.debug("Entry point discovery failed: %s", exc)
        return candidates

    def _read_meta(self, directory: Path) -> Optional[PluginMeta]:
        """Parse plugin.yaml from a directory. Returns None if not found/invalid."""
        yaml_path = directory / "plugin.yaml"
        if not yaml_path.exists():
            return None

        try:
            import yaml
        except ImportError:
            # Fallback: minimal YAML parser for simple plugin.yaml files
            return self._read_meta_fallback(yaml_path, directory)

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict):
                return None

            name = data.get("name")
            version = data.get("version")
            entry_point = data.get("entry_point")

            if not all([name, version, entry_point]):
                logger.warning(
                    "plugin.yaml at %s missing required fields (name, version, entry_point)",
                    yaml_path,
                )
                return None

            return PluginMeta(
                name=str(name),
                version=str(version),
                entry_point=str(entry_point),
                description=str(data.get("description", "")),
                author=str(data.get("author", "")),
                dependencies=list(data.get("dependencies", [])),
                frameworm_min_version=data.get("frameworm_min_version"),
                source_path=directory,
            )
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", yaml_path, exc)
            return None

    def _read_meta_fallback(self, yaml_path: Path, directory: Path) -> Optional[PluginMeta]:
        """Minimal YAML parser when PyYAML is not installed."""
        data = {}
        try:
            for line in yaml_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("-"):
                    continue
                if ":" in line:
                    key, _, value = line.partition(":")
                    data[key.strip()] = value.strip()
        except Exception:
            return None

        name = data.get("name")
        version = data.get("version")
        entry_point = data.get("entry_point")

        if not all([name, version, entry_point]):
            return None

        return PluginMeta(
            name=name, version=version, entry_point=entry_point,
            description=data.get("description", ""),
            author=data.get("author", ""),
            source_path=directory,
        )

    def _load_one(self, meta: PluginMeta) -> LoadResult:
        """Attempt to load a single plugin."""
        # 1. Check dependencies
        if not self.ignore_missing_deps:
            missing = self._check_dependencies(meta.dependencies)
            if missing:
                return LoadResult(
                    meta=meta,
                    success=False,
                    error=f"Missing dependencies: {', '.join(missing)}",
                )

        # 2. Resolve entry point
        register_fn = self._resolve_entry_point(meta)
        if register_fn is None:
            return LoadResult(
                meta=meta,
                success=False,
                error=f"Could not resolve entry_point: {meta.entry_point!r}",
            )

        # 3. Call register
        try:
            register_fn(self.hook_registry, self.model_registry)
        except Exception as exc:
            return LoadResult(
                meta=meta,
                success=False,
                error=f"register() raised: {exc}",
            )

        return LoadResult(meta=meta, success=True)

    def _resolve_entry_point(self, meta: PluginMeta) -> Optional[Callable]:
        """Import module and return the register function."""
        entry = meta.entry_point  # "module:function" or "package.module:function"

        if ":" not in entry:
            logger.warning("entry_point %r must be in 'module:function' format", entry)
            return None

        module_path, _, fn_name = entry.partition(":")

        # Add the plugin's directory to sys.path temporarily
        inserted = False
        if meta.source_path is not None:
            src = str(meta.source_path)
            if src not in sys.path:
                sys.path.insert(0, src)
                inserted = True

        try:
            module = importlib.import_module(module_path)
            fn = getattr(module, fn_name, None)
            if fn is None:
                logger.warning(
                    "Module %r has no attribute %r", module_path, fn_name
                )
                return None
            return fn
        except ImportError as exc:
            logger.warning("Cannot import %r: %s", module_path, exc)
            return None
        finally:
            if inserted and str(meta.source_path) in sys.path:
                sys.path.remove(str(meta.source_path))

    @staticmethod
    def _check_dependencies(deps: List[str]) -> List[str]:
        """Return list of unmet dependency strings."""
        missing = []
        for dep in deps:
            # Strip version specifiers to get the package name
            pkg = dep.split(">=")[0].split("<=")[0].split("==")[0].split("!=")[0].strip()
            pkg = pkg.replace("-", "_")
            try:
                importlib.import_module(pkg)
            except ImportError:
                missing.append(dep)
        return missing