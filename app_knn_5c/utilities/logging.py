import logging


class MyLogger:
    """
    Custom Logger class to simplify logging operations with configurable levels and file destinations.

    Parameters:
        name (str): The name of the logger.
        level (int): The logging level (default is logging.DEBUG).
        filex (str): The name of the log file (default is __name__).

    Attributes:
        logger (logging.Logger): The logger instance for logging operations.

    Methods:
        debug(msg): Log a message with DEBUG level.
        info(msg): Log a message with INFO level.
        warning(msg): Log a message with WARNING level.
        error(msg): Log a message with ERROR level.
        critical(msg): Log a message with CRITICAL level.
    """

    def __init__(self, name, level=logging.DEBUG, filex=__name__):
        """
        Initialize the MyLogger instance.

        Args:
            name (str): The name of the logger.
            level (int, optional): The logging level. Defaults to logging.DEBUG.
            filex (str, optional): The name of the log file. Defaults to __name__.
        """
            
        filename = f"logs/{filex}.log"
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        formatter = logging.Formatter(
            "%(asctime)s:%(name)s:%(module)s:%(levelname)s:%(message)s"
        )

        file_handler = logging.FileHandler(filename)

        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)
