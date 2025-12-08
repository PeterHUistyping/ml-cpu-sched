# utils/logging.py

import logging
from pathlib import Path


def create_logger(extra_args=''):
    log_filename = f'../../outputs/simulation_{extra_args}.log'
    log_path = Path(log_filename)

    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(filename=log_path,
                        filemode='w',
                        level=logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        root_logger.addHandler(console_handler)

    return logging
