from jax.experimental.pjit import PartitionSpec as P


def get_shard_rules(model_type):
    if model_type == 't5':
        return _get_partition_rules_t5_v1_1()
    elif model_type == 'opt':
        return _get_partition_rules_opt()
    elif model_type == 'gptj':
        return _get_partition_rules_gptj()
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


def _get_partition_rules_gptj():
    return [
        # embeddings
        (("transformer", "wte", "embedding"), P(None, 'mp')),
        # atention
        (("attn", "(k_proj|q_proj|v_proj)", "kernel"), P(None, "mp")),
        (("attn", "out_proj", "kernel"), P("mp", None)),
        # mlp
        (("mlp", "fc_in", "kernel"), P(None, "mp")),
        (("mlp", "fc_in", "bias"), P("mp")),
        (("mlp", "fc_out", "kernel"), P("mp", None)),
        (("mlp", "fc_out", "bias"), None),
        # layer norms
        ((r"ln_\d+", "bias"), None),
        ((r"\d+", r"ln_\d+", "scale"), None),
        (("ln_f", "bias"), None),
        (("ln_f", "scale"), None),
        # output head
        (("lm_head", "kernel"), P(None, "mp")),
        (("lm_head", "bias"), P("mp")),
    ]


def _get_partition_rules_opt():
    return [
        # embeddings
        (("model", "decoder", "embed_positions", "embedding"), P(None, "mp")),
        (("model", "decoder", "embed_tokens", "embedding"), P(None, "mp")),
        (("model", "decoder", "project_in", "kernel"), None),
        (("model", "decoder", "project_out", "kernel"), None),
        # atention
        (("self_attn", "(k_proj|q_proj|v_proj)", "kernel"), P(None, "mp")),
        (("self_attn", "(k_proj|q_proj|v_proj)", "bias"), P("mp")),
        (("self_attn", "out_proj", "kernel"), P("mp", None)),
        (("self_attn", "out_proj", "bias"), P(None)),
        # mlp
        (("fc1", "kernel"), P(None, "mp")),
        (("fc1", "bias"), P("mp")),
        (("fc2", "kernel"), P("mp", None)),
        (("fc2", "bias"), None),
        # layer norms
        (("final_layer_norm", "bias"), None),
        (("final_layer_norm", "scale"), None),
        (("self_attn_layer_norm", "bias"), None),
        (("self_attn_layer_norm", "scale"), None),
        # output head
        (("model", "lm_head", "kernel"), P(None, "mp")),
    ]


def _get_partition_rules_bart():
    return [
        (('final_logits_bias',), None),
        (('(bias|scale)', ), None),
        (('embedding', ), P(None, 'mp')),
        (('fc1', 'kernel'), P(None, "mp")),
        (('fc2', 'kernel'), P('mp', None)),
        (("(q_proj|k_proj|v_proj)", "kernel"), P(None, "mp")),
        (('out_proj', 'kernel'), P('mp', None)),
    ]
