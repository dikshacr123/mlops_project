import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define base paths dynamically
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # go one level up
LOGS_DIR = os.path.join(BASE_DIR, os.getenv('LOGS_DIR', 'logs'))

# Ensure Logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

# Define log file path
LOG_FILE = os.path.join(LOGS_DIR, "output.log")

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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

