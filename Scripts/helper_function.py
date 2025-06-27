import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
LOGS_DIR = os.path.join(BASE_DIR, os.getenv('LOGS_DIR', 'logs'))

os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "output.log")

def setup_logger():
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        logger.addHandler(stream_handler)

def get_logger():
    return logging.getLogger()

def log_info(message):
    logger = get_logger()
    logger.info(message)
    print(f"INFO: {message}")

def log_error(message):
    logger = get_logger()
    logger.error(message)
    print(f"ERROR: {message}")

def log_warning(message):
    logger = get_logger()
    logger.warning(message)
    print(f"WARNING: {message}")