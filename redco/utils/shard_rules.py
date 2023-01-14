from jax.experimental.pjit import PartitionSpec as P


def get_shard_rules(model_type):
    if model_type == 't5':
        return _get_partition_rules_t5_v1_1()
    elif model_type == 'bart':
        return _get_partition_rules_bart()
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


def _get_partition_rules_bart():
    return [
        (('final_logits_bias',), None),
        (('bias', ), None),
        (('self_attn_layer_norm', 'scale'), None),
        (('layernorm_embedding', 'scale'), None),
        (('encoder_attn_layer_norm', 'scale'), None),
        (('final_layer_norm', 'scale'), None),
        (('shared', 'embedding'), P(None, 'mp')),
        (('fc1', 'kernel'), P(None, "mp")),
        (('fc2', 'kernel'), P('mp', None)),
        (('k_proj', 'kernel'), P(None, "mp")),
        (('q_proj', 'kernel'), P(None, "mp")),
        (('v_proj', 'kernel'), P(None, "mp")),
        (('out_proj', 'kernel'), P('mp', None)),
        (('embed_positions', 'embedding'), P(None, 'mp'))
    ]
