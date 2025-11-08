from dataclasses import dataclass
from transformers import GPT2Config

# 备选隐层宽度与层数（适配 head_dim=64）
CAND_D = [256, 384, 512, 768, 1024, 1280]
CAND_L = [6, 8, 12, 16, 20, 24]

@dataclass
class ModelSpec:
    config: GPT2Config
    params: int

def approx_params(cfg: GPT2Config) -> int:
    """
    粗估参数量：embedding + 每层(注意力+MLP)
    GPT2 层内大致参数：4d^2 (attn qkv o) + 8d^2 (mlp) ≈ 12 d^2
    """
    V = cfg.vocab_size
    d = cfg.n_embd
    L = cfg.n_layer
    per_layer = 12 * d * d
    E = V * d  # token embedding（与 lm_head 权重共享时不重复计算）
    return int(E + L * per_layer)


def make_config(target_params: int, vocab_size: int = 50257, n_positions: int = 1024) -> ModelSpec:
    """
    给定目标参数量，搜索 (d_model, n_layer) 组合，使估算参数量接近 target_params（误差尽量小）。
    n_head 由 d_model // 64 确定。
    """
    best: ModelSpec | None = None
    for d in CAND_D:
        if d % 64 != 0:
            continue
        for L in CAND_L:
            cfg = GPT2Config(
                vocab_size=vocab_size,
                n_positions=n_positions,
                n_ctx=n_positions,
                n_embd=d,
                n_layer=L,
                n_head=d // 64,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
            )
            p = approx_params(cfg)
            if best is None or abs(p - target_params) < abs(best.params - target_params):
                best = ModelSpec(cfg, p)
    assert best is not None
    return best
