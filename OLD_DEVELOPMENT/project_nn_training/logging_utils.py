# logging_utils.py
import logging
import sys

def get_logger(log_file='training.log', level=logging.INFO):
    """
    Configures and returns a logger that writes to both console and a file.

    :param log_file: Path to the file where logs will be saved.
    :param level: Logging level (e.g. logging.INFO, logging.DEBUG, etc.).
    :return: A configured logger instance.
    """
    logger = logging.getLogger("semi_supervised_logger")
    logger.setLevel(level)

    # Avoid adding handlers multiple times if get_logger is called repeatedly
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
