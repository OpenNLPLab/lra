LSAttentionNonCausal(
	dim=embed_dim,
	num_heads=args.encoder_attention_heads,
	max_seq_len=getattr(args, "max_seq_len", 512),
	dropout=args.attention_dropout,
	num_landmarks=getattr(args, "num_landmarks", 32),
	window_size=getattr(args, "window_size", 8),
)