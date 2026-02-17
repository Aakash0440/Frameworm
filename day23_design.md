# Day 23: Plugin System

## Systems
1. Hook Registry - Lifecycle callbacks
2. Plugin Discovery - Auto-load from paths
3. Plugin Loader - Import & register
4. Example Plugins - Custom model, metric, callback
5. CLI Commands - List/load/unload plugins

## Architecture
Plugins → Register Hooks → Core calls hooks at lifecycle points

## Plugin Structure
frameworm_plugins/my_plugin/
init.py          # Entry point
plugin.yaml          # Metadata
models.py            # Custom models
callbacks.py         # Custom callbacks