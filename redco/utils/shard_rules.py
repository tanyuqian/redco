from jax.experimental.pjit import PartitionSpec as P


def get_shard_rules(arch_name):
    if 't5' in arch_name.lower():
        return _get_partition_rules_t5_v1_1()
    else:
        return None


def _get_partition_rules_t5_v1_1():
    return [
        # embeddings
        (("shared", "embedding"), P("mp", None)),
        (("relative_attention_bias", "embedding"), None),
        # self atention
        (("SelfAttention", "(k|q|v)", "kernel"), P(None, "mp")),
        (("SelfAttention", "o", "kernel"), P("mp", None)),
        # cross atention
        (("EncDecAttention", "(k|q|v)", "kernel"), P(None, "mp")),
        (("EncDecAttention", "o", "kernel"), P("mp", None)),
        # mlp
        (("DenseReluDense", "wi_0", "kernel"), P(None, "mp")),
        (("DenseReluDense", "wi_1", "kernel"), P(None, "mp")),
        (("DenseReluDense", "wo", "kernel"), P("mp", None)),
        # layer norms
        (("layer_norm", "weight"), None),
        (("final_layer_norm", "weight"), None),
        # output head
        (("lm_head", "kernel"), P(None, "mp")),
    ]
