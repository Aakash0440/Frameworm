"""Type definitions for Frameworm"""

from typing import Any, Dict, List, Union, Optional
from pathlib import Path

# Config types
ConfigDict = Dict[str, Any]
ConfigPath = Union[str, Path]

# Common types
PathLike = Union[str, Path]