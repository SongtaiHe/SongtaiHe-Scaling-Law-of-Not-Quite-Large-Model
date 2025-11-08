from models.factory import approx_params

# Decoder-only 粗估：FLOPs ≈ 6 * N * D
def flops_estimate(param_count: int, train_tokens: int) -> float:
    return 6.0 * param_count * train_tokens
