from functools import partial

from .trainer import Trainer
from ..utils.maml_utils import maml_default_collate_fn, maml_default_loss_fn

from ..predictors import MAMLPredictor


class MAMLTrainer(Trainer):
    def __init__(self,
                 deployer,
                 apply_fn,
                 params,
                 optimizer,
                 inner_loss_fn,
                 inner_learning_rate,
                 inner_n_steps,
                 train_key='train',
                 val_key='test',
                 lr_schedule_fn=None,
                 params_shard_rules=None):
        collate_fn = partial(
            maml_default_collate_fn, train_key=train_key, val_key=val_key)

        loss_fn = partial(
            maml_default_loss_fn,
            inner_loss_fn=inner_loss_fn,
            inner_learning_rate=inner_learning_rate,
            inner_n_steps=inner_n_steps)

        self._default_predictor_fn = partial(
            MAMLPredictor,
            deployer=deployer,
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
            params_shard_rules=params_shard_rules)

    def get_default_predictor(self, inner_pred_fn):
        return self._default_predictor_fn(inner_pred_fn=inner_pred_fn)
