"""
PhenoCluster Logging Module
============================

Provides structured logging configuration for the entire pipeline.
"""

import logging
import sys
import threading
from pathlib import Path
from typing import Optional


class PhenoClusterLogger:
    """
    Centralized logging configuration for PhenoCluster.

    Features:
    - Multiple format styles (minimal, standard, detailed)
    - File and console logging
    - Context managers for temporary log level changes
    """

    _loggers = {}
    _lock = threading.Lock()

    @classmethod
    def get_logger(
        cls,
        name: str,
        level: str = "INFO",
        log_format: str = "detailed",
        log_to_file: bool = False,
        log_file: Optional[str] = None,
    ) -> logging.Logger:
        """
        Get or create a logger with specified configuration.

        Parameters
        ----------
        name : str
            Logger name
        level : str
            Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format : str
            Format style: 'simple' or 'detailed'
        log_to_file : bool
            Whether to log to a file
        log_file : str, optional
            Path to log file

        Returns
        -------
        logging.Logger
            Configured logger instance
        """
        with cls._lock:
            if name in cls._loggers:
                return cls._loggers[name]

            logger = logging.getLogger(name)
            logger.setLevel(getattr(logging, level.upper()))

            logger.handlers.clear()

            # Define formats
            if log_format == "minimal":
                fmt = logging.Formatter("%(message)s")
            elif log_format == "standard":
                fmt = logging.Formatter("[%(levelname)s] %(message)s")
            elif log_format == "detailed":
                fmt = logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
                )
            else:  # default to detailed
                fmt = logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
                )

            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(fmt)
            logger.addHandler(console_handler)

            # File handler
            if log_to_file and log_file:
                file_path = Path(log_file)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(file_path, mode="a")
                file_handler.setFormatter(fmt)
                logger.addHandler(file_handler)

            cls._loggers[name] = logger
            return logger


def get_logger(name: str, config=None) -> logging.Logger:
    """
    Simplified logger factory function.

    This is the recommended way to create loggers in PhenoCluster modules.
    It reduces boilerplate by automatically extracting logging configuration
    from the config object.

    Parameters
    ----------
    name : str
        Logger name (e.g., 'preprocessing', 'modeling', 'evaluation')
    config : PhenoClusterConfig, optional
        Configuration object. If None, uses defaults.

    Returns
    -------
    logging.Logger
        Configured logger instance

    Examples
    --------
    >>> from phenocluster.logger import get_logger
    >>> logger = get_logger('my_module', config)
    >>> logger.info("Processing data...")
    """
    if config is None:
        return PhenoClusterLogger.get_logger(name)

    log_path = Path(config.output_dir) / "logs" / config.logging.log_file
    return PhenoClusterLogger.get_logger(
        name,
        level=config.logging.level,
        log_format=config.logging.format,
        log_to_file=config.logging.log_to_file,
        log_file=log_path,
    )
