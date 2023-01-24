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


def log_info(logger, info, title):
    info = str(info)

    if title is not None:
        max_len = max(max([len(t) for t in info.split('\n')]), len(title) + 4)

        logger.info('=' * max_len)
        logger.info(f'### {title}')
        logger.info('-' * max_len)
        for t in info.split('\n'):
            logger.info(t)
        logger.info('=' * (max_len + 4))
    else:
        logger.info(info)