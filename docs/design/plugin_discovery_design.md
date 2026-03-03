# Plugin Discovery Design

## Goal
Automatically find and import Python files in plugins/ directory

## Algorithm

1. Scan plugins/ directory for .py files
2. Import each file
3. Registration happens via decorators during import
4. Cache imported modules to avoid re-import

## File Structure
plugins/
├── init.py           # Empty (makes it a package)
├── my_model.py           # User's custom model
├── my_trainer.py         # User's custom trainer
└── subdirs/
└── another_model.py  # Nested plugins also discovered

## Safety

- Only import .py files (not .pyc, __pycache__, etc.)
- Skip __init__.py
- Handle import errors gracefully
- Prevent infinite recursion
- Sandbox imports (catch exceptions)

## Performance

- Lazy discovery (only when requested)
- Cache discovered plugins
- Option to disable auto-discovery
