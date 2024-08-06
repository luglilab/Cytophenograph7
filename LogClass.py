import logging
import sys


class CustomFormatter(logging.Formatter):
    """
    Custom formatter to display log messages with different prefixes
    based on their severity level.
    """
    FORMATS = {
        logging.INFO: "###%(msg)s",
        logging.WARNING: "$$$%(msg)s",
        logging.ERROR: "@@@%(msg)s",
        "DEFAULT": "%(msg)s",
    }

    def format(self, record):
        """
        Format log messages according to their severity level.
        """
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS['DEFAULT'])
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class LoggerSetup:
    """
    A class to set up logging configuration using a custom formatter.
    """

    @staticmethod
    def setup_logging(level=logging.INFO):
        """
        Set up logging configuration with a custom formatter.

        :param level: The logging level (default is logging.INFO).
        :return: A logger instance with the configured settings.
        """
        # Create a logger with the specified level
        logger = logging.getLogger(__name__)
        logger.setLevel(level)

        # Set up the stream handler to output logs to stdout
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(CustomFormatter())

        # Add the handler to the logger if it's not already added
        if not logger.handlers:
            logger.addHandler(handler)

        return logger
