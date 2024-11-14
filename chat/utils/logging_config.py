# chat/utils/logging_config.py
import logging
import os
from datetime import datetime
from pathlib import Path

def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    # Create log filename with timestamp
    log_file = log_dir / f'llm_app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Console output
        ]
    )

    # Create logger
    logger = logging.getLogger('LLMApp')
    logger.setLevel(logging.DEBUG)

    return logger