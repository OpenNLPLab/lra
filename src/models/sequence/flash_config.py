FlashAttention(
	embed_dim,
	args.encoder_attention_heads,
	dropout=args.attention_dropout,
	self_attention=True,
	q_noise=self.quant_noise,
	qn_block_size=self.quant_noise_block_size,
	# add
	s=getattr(args, "s", 128),
	norm_type=getattr(args, "norm_type", "layer_norm"),
	eps=getattr(args, "eps", 1e-5),
	max_position_embeddings=getattr(args, "max_position_embeddings", 512),
	expansion_factor=getattr(args, "expansion_factor", 2)
)

num_heads = 1
