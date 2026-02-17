"""
CLI logging utilities.
"""

import logging

from rich.logging import RichHandler


def setup_logging(level: str = "INFO", log_file: str = None):
    """Setup CLI logging"""

    # Create logger
    logger = logging.getLogger("frameworm")
    logger.setLevel(getattr(logging, level.upper()))

    # Console handler with rich formatting
    console_handler = RichHandler(rich_tracebacks=True)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

    return logger
