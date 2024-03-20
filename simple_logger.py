import logging
from logging.handlers import RotatingFileHandler


class SimpleLogger:
    """ Pretty much self explanatory... """
    def __init__(self, name='MainLogger', logfile='logfile.log', max_bytes=20000000, backup_count=25):
        self.logger = logging.getLogger(name)
        # Set log Level here
        self.logger.setLevel(logging.DEBUG)

        file_handler = RotatingFileHandler(
            logfile, maxBytes=max_bytes, backupCount=backup_count)
        stream_handler = logging.StreamHandler()

        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)
