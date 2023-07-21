#  Copyright 2021 Google LLC
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import logging
import json
import jax


def get_logger(verbose, workdir):
    logger = logging.getLogger('redco')

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        fmt="[%(asctime)s - %(name)s - %(levelname)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S"))
    logger.addHandler(handler)

    if workdir is not None:
        handler = logging.FileHandler(filename=f'{workdir}/log.txt')
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


def log_info(info, title, logger, summary_writer, step):
    info = str(info)

    if title is not None:
        if step is not None:
            title_ = f'{title} (step {step})'
        else:
            title_ = title
            step = 0

        if summary_writer is not None:
            summary_writer.text(title, info.replace('\n', '\n\n'), step=step)

        max_len = max(max([len(t) for t in info.split('\n')]), len(title_) + 4)

        logger.info('=' * max_len)
        logger.info(f'### {title_}')
        logger.info('-' * max_len)
        for t in info.split('\n'):
            logger.info(t)
        logger.info('=' * max_len)

    else:
        logger.info(info)


def save_outputs(outputs, workdir, desc, logger, summary_writer, step):
    outputs = jax.tree_util.tree_map(str, outputs)
    json.dump(outputs, open(f'{workdir}/outputs_{desc}.json', 'w'), indent=4)
    logger.info(
        f'Outputs ({desc}) has been saved into {workdir}/outputs_{desc}.json.')

    if summary_writer is not None:
        samples_str = json.dumps(outputs[:10], indent=4).replace('\n', '\n\n')
        summary_writer.text('outputs', samples_str, step=step)

    return f'{workdir}/outputs_{desc}.json'