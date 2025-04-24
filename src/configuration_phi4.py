from transformers.configuration_utils import PretrainedConfig

#Modified from Phi3
class Phi4Config(PretrainedConfig):
    model_type = "phi4"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=200064,
        hidden_size=3072,
        intermediate_size=8192,
        num_hidden_layers=32,
        num_attention_heads=24,
        num_key_value_heads=8,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attention_dropout=0.0,
        hidden_act="silu",
        max_position_embeddings=131072,
        original_max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        partial_rotary_factor=0.75,
        bos_token_id=199999,
        eos_token_id=199999,
        pad_token_id=199999,
        sliding_window=None,
        attention_bias=False,
        mlp_bias=False,
        lm_head_bias=False,
        full_attn_mod=1,
        expansion_factor=4,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling_config
        self.partial_rotary_factor = partial_rotary_factor
        self._rope_scaling_adjustment()
        self._rope_scaling_validation()
        self.sliding_window = sliding_window
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.lm_head_bias = lm_head_bias
        self.full_attn_mod = full_attn_mod
        self.expansion_factor = expansion_factor

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_adjustment(self):
        """
        Adjust the `type` of the `rope_scaling` configuration for backward compatibility.
        """
        if self.rope_scaling is None:
            return

        rope_scaling_type = self.rope_scaling.get("type", None)

        if rope_scaling_type is not None and rope_scaling_type in ["su", "yarn"]:
            self.rope_scaling["type"] = "longrope"

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 3:
            raise ValueError(
                "`rope_scaling` must be a dictionary with three fields, `type`, `short_factor` and `long_factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_short_factor = self.rope_scaling.get("short_factor", None)
        rope_scaling_long_factor = self.rope_scaling.get("long_factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["longrope"]:
            raise ValueError(f"`rope_scaling`'s type field must be one of ['longrope'], got {rope_scaling_type}")
        if not (
            isinstance(rope_scaling_short_factor, list)
            and all(isinstance(x, (int, float)) for x in rope_scaling_short_factor)
        ):
            raise ValueError(
                f"`rope_scaling`'s short_factor field must be a list of numbers, got {rope_scaling_short_factor}"
            )
        rotary_ndims = int(self.hidden_size // self.num_attention_heads * 0.75)
        if not len(rope_scaling_short_factor) == rotary_ndims // 2:
            raise ValueError(
                f"`rope_scaling`'s short_factor field must have length {rotary_ndims // 2}, got {len(rope_scaling_short_factor)}"
            )
        if not (
            isinstance(rope_scaling_long_factor, list)
            and all(isinstance(x, (int, float)) for x in rope_scaling_long_factor)
        ):
            raise ValueError(
                f"`rope_scaling`'s long_factor field must be a list of numbers, got {rope_scaling_long_factor}"
            )
        if not len(rope_scaling_long_factor) == rotary_ndims // 2:
            raise ValueError(
                f"`rope_scaling`'s long_factor field must have length {rotary_ndims // 2}, got {len(rope_scaling_long_factor)}"
            )


rope_scaling_config = {
    "type": "longrope",
    "short_factor": [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    "long_factor": [
      1,
      1.118320672,
      1.250641126,
      1.398617824,
      1.564103225,
      1.74916897,
      1.956131817,
      2.187582649,
      2.446418898,
      2.735880826,
      3.059592084,
      3.421605075,
      3.826451687,
      4.279200023,
      4.785517845,
      5.351743533,
      5.984965424,
      6.693110555,
      7.485043894,
      8.370679318,
      9.36110372,
      10.4687158,
      11.70738129,
      13.09260651,
      14.64173252,
      16.37415215,
      18.31155283,
      20.47818807,
      22.90118105,
      25.61086418,
      28.64115884,
      32.03,
      32.1,
      32.13,
      32.23,
      32.6,
      32.61,
      32.64,
      32.66,
      32.7,
      32.71,
      32.93,
      32.97,
      33.28,
      33.49,
      33.5,
      44.16,
      47.77
    ]
}