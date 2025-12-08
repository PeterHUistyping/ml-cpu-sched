import logging


def create_logger(extra_args=''):
    logging.basicConfig(filename=f'outputs/simulation_{extra_args}.log', 
    filemode='w',
    level=logging.INFO)

    return logging

    