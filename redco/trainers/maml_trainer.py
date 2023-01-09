from functools import partial

from .trainer import Trainer
from ..utils.maml_utils import maml_default_collate_fn, maml_default_loss_fn


class MAMLTrainer(Trainer):
    def __init__(self,
                 deployer,
                 apply_fn,
                 params,
                 optimizer,
                 lr_schedule_fn,
                 inner_loss_fn,
                 inner_learning_rate,
                 inner_n_steps,
                 dummy_example,
                 train_key='train',
                 val_key='test',
                 params_shard_rules=None):
        collate_fn = partial(
            maml_default_collate_fn, train_key=train_key, val_key=val_key)

        loss_fn = partial(
            maml_default_loss_fn,
            inner_loss_fn=inner_loss_fn,
            inner_learning_rate=inner_learning_rate,
            inner_n_steps=inner_n_steps)

        super(MAMLTrainer, self).__init__(
            deployer=deployer,
            collate_fn=collate_fn,
            apply_fn=apply_fn,
            loss_fn=loss_fn,
            params=params,
            optimizer=optimizer,
            lr_schedule_fn=lr_schedule_fn,
            dummy_example=dummy_example,
            params_shard_rules=params_shard_rules)