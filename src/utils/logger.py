"""
Logging utilities for OSINTropy.
Provides consistent logging across all modules with configurable output.
"""

import logging
import sys
from typing import Optional
from pathlib import Path


# Default log format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Color codes for terminal output
COLORS = {
    'DEBUG': '\033[36m',    # Cyan
    'INFO': '\033[32m',     # Green
    'WARNING': '\033[33m',  # Yellow
    'ERROR': '\033[31m',    # Red
    'CRITICAL': '\033[35m', # Magenta
    'RESET': '\033[0m'      # Reset
}


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for terminal output."""
    
    def format(self, record):
        """Format log record with colors."""
        # Add color to level name
        levelname = record.levelname
        if levelname in COLORS:
            record.levelname = f"{COLORS[levelname]}{levelname}{COLORS['RESET']}"
        
        return super().format(record)


def setup_logging(log_level: str = 'INFO', 
                  log_file: Optional[str] = None,
                  use_colors: bool = True) -> logging.Logger:
    """
    Setup logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        use_colors: Whether to use colored output for console
        
    Returns:
        Root logger instance
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    if use_colors and sys.stdout.isatty():
        console_formatter = ColoredFormatter(LOG_FORMAT, DATE_FORMAT)
    else:
        console_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def set_log_level(level: str):
    """
    Change logging level for all loggers.
    
    Args:
        level: New logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger().setLevel(numeric_level)
    
    # Update all handlers
    for handler in logging.getLogger().handlers:
        handler.setLevel(numeric_level)


def disable_logging():
    """Disable all logging output."""
    logging.disable(logging.CRITICAL)


def enable_logging():
    """Re-enable logging output."""
    logging.disable(logging.NOTSET)


# Initialize default logging on import
if not logging.getLogger().handlers:
    setup_logging()
