from functools import partial

from .predictor import Predictor
from ..utils.maml_utils import maml_default_collate_fn, maml_default_pred_fn


class MAMLPredictor(Predictor):
    def __init__(self,
                 deployer,
                 inner_loss_fn,
                 inner_learning_rate,
                 inner_n_steps,
                 inner_pred_fn,
                 train_key='train',
                 val_key='test',
                 output_fn=None,
                 params=None,
                 params_shard_rules=None):
        collate_fn = partial(
            maml_default_collate_fn, train_key=train_key, val_key=val_key)

        pred_fn = partial(
            maml_default_pred_fn,
            inner_loss_fn=inner_loss_fn,
            inner_learning_rate=inner_learning_rate,
            inner_n_steps=inner_n_steps,
            inner_pred_fn=inner_pred_fn)

        super(MAMLPredictor, self).__init__(
            deployer=deployer,
            collate_fn=collate_fn,
            pred_fn=pred_fn,
            output_fn=output_fn,
            params=params,
            params_shard_rules=params_shard_rules)
