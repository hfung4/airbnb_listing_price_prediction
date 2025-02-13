import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from airbnb_listing.config import LOGS_DIR


def setup_logger(
    name="airbnb_listing",
    log_file=LOGS_DIR / "airbnb_listing.log",
    file_log_level=logging.INFO,
    console_log_level=logging.INFO,
    max_bytes=50 * 1024 * 1024,  # 50 MB
    backup_count=2,
    disable_logging=False,
) -> logging.Logger:
    """
    Sets up a logger with specified configurations.

    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file.
        file_log_level (int): Logging level for file handler.
        console_log_level (int): Logging level for console handler.
        max_bytes (int): Maximum bytes per log file for rotating file handler.
        backup_count (int): Number of backup files to keep.
        disable_logging (bool): If True, disables logging.

    Returns:
        logger[logging.Logger]: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(file_log_level)

    # If logging is disabled, return a disabled logger
    if disable_logging:
        logger.disabled = True
        return logger

    log_file = Path(log_file)  # Ensure it's a Path object
    if not log_file.exists():
        log_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent directories exist
        log_file.touch()  # Create an empty log file

    # Create a rotating file handler
    file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    file_handler.setLevel(file_log_level)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_log_level)

    # Create a logging format
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    if not logger.handlers:  # To avoid adding handlers multiple times in case of reconfiguration
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


# Create a logger instance
logger = setup_logger()
