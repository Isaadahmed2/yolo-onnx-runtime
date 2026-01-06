import logging
import sys
from logging.handlers import TimedRotatingFileHandler
import os

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

FORMATTER = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOG_FILE = os.path.join(LOGS_DIR, "app.log")

def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler

def get_file_handler():
    # Rotates the log file every day, keeps 7 days of logs
    file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight', backupCount=7)
    file_handler.setFormatter(FORMATTER)
    return file_handler

def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler())
    # Propagate the error up to the root logger
    logger.propagate = False
    return logger
