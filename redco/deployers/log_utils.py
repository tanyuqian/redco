import logging
import jax


def get_logger(verbose):
    logger = logging.getLogger('redco')

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        fmt="[%(asctime)s - %(name)s - %(levelname)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S"))

    logger.addHandler(handler)
    logger.propagate = False

    if verbose and jax.process_index() == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)

    return logger
