import logging
import sys
from typing import Literal

from loguru import logger as loguru_logger

LogLevel = Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]


def get_logger(name: str, level: LogLevel = "INFO", log_file: str = None) -> logging.Logger:
    """
    Create and return a customized logger with error message coloring.

    Args:
        name (str): Name of the logger.
        level (int): Logging level, default is logging.INFO.
        log_file (str): File path where logs should be written.

    Returns:
        logging.Logger: Configured logger instance.
    """
    level = level.upper()
    loguru_logger.remove()
    # https://loguru.readthedocs.io/en/stable/api/logger.html#record
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[passed_name]}</cyan>:<cyan>{line}</cyan> | - <level>{message}</level>"
    )
    # fmt = "[<green><b>{time:YYYY-MM-DD hh:mm:ss}</b></green>][<cyan><b>{file}:{name}:{line}</b></cyan> - <cyan>{name}:{function}</cyan>][ {extra[passed_name]} ] HELLO {message}\n"
    sink = sys.stdout if log_file is None else log_file
    loguru_logger.add(
        sink,
        format=fmt,
        level=level,
        filter=lambda record: "passed_name" in record["extra"],
    )
    return loguru_logger.bind(passed_name=name)


if __name__ == "__main__":
    logger_A = get_logger("A", level="INFO", log_file=None)
    logger_B = get_logger("B", level="INFO", log_file=None)
    logger_A.warning("This is a info message.")
    logger_B.info("This is an info message.")
    logger_B.debug("This is an info message.")
