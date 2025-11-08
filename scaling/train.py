import math, argparse, time, os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from models.factory import make_config, approx_params
from utils.logging import JsonlLogger
from utils.flops import flops_estimate

def collate_pad(tok):
    def _fn(batch):
        return tok.pad(batch, return_tensors="pt")
    return _fn

def evaluate(acc: Accelerator, model: GPT2LMHeadModel, loader) -> float:
    model.eval()
    losses, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            with acc.autocast():
                out = model(**batch, labels=batch["input_ids"])
            losses += out.loss.item() * len(batch["input_ids"])
            n += len(batch["input_ids"])
    return losses / max(n, 1)

def cycle(loader):
    while True:
        for x in loader:
            yield x

def main(args):
    # 初始化 Accelerator（根据 accelerate_config.yaml 自动选择分布式/混精）
    acc = Accelerator()
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 模型构建
    spec = make_config(int(args.target_params), vocab_size=tok.vocab_size, n_positions=args.seq_len)
    model = GPT2LMHeadModel(spec.config)
    n_params = approx_params(model.config)

    # 数据集：本地 text 文件
    ds = load_dataset("text", data_files={"train": args.train_path, "val": args.val_path})
    def tok_fn(ex):
        return tok(ex["text"], truncation=True, max_length=args.seq_len)
    ds = ds.map(tok_fn, batched=True, remove_columns=["text"]).with_format("torch")

    train_loader = torch.utils.data.DataLoader(
        ds["train"], batch_size=args.micro_bsz, shuffle=True, collate_fn=collate_pad(tok)
    )
    val_loader = torch.utils.data.DataLoader(
        ds["val"], batch_size=args.micro_bsz, shuffle=False, collate_fn=collate_pad(tok)
    )

    # 优化器与调度
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    total_tokens_per_step = args.seq_len * args.global_bsz
    warmup_steps = args.warmup

    model, opt, train_loader, val_loader = acc.prepare(model, opt, train_loader, val_loader)
    max_steps = math.ceil(args.train_tokens / total_tokens_per_step)
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps, max_steps)

    os.makedirs(args.save_dir, exist_ok=True)
    logger = JsonlLogger(f"{args.save_dir}/metrics.jsonl")

    seen_tokens = 0
    best_val = float("inf")
    start = time.time()

    grad_accum = max(1, args.grad_accum)
    acc.print(
        f"Start training | approx_params={n_params:,} | target_params={int(args.target_params):,} "
        f"| max_steps={max_steps} | global_bsz={args.global_bsz}"
    )

    # 训练循环
    model.train()
    step = 0
    for batch in cycle(train_loader):
        with acc.autocast():
            out = model(**batch, labels=batch["input_ids"])
            loss = out.loss / grad_accum
        acc.backward(loss)

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            seen_tokens += total_tokens_per_step

            # 评估与日志
            if (step + 1) % args.eval_every == 0:
                val_ce = evaluate(acc, model, val_loader)
                best_val = min(best_val, val_ce)
                logger.log(
                    step=step + 1,
                    seen_tokens=seen_tokens,
                    val_loss=val_ce,
                    train_loss=loss.item() * grad_accum,  # 还原
                    lr=sched.get_last_lr()[0],
                    flops_cum=flops_estimate(n_params, seen_tokens),
                )
                if acc.is_main_process:
                    torch.save(model.state_dict(), f"{args.save_dir}/ckpt.pt")

            if seen_tokens >= args.train_tokens:
                break
        step += 1

    if acc.is_main_process:
        final = {
            "N": int(n_params),
            "D": int(seen_tokens),
            "best_val_loss": float(best_val),
            "flops_total": float(flops_estimate(n_params, seen_tokens)),
            "walltime_s": float(time.time() - start),
        }
        import json
        with open(f"{args.save_dir}/final.json", "w") as f:
            json.dump(final, f)
        print(final)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_params", type=float, required=True, help="目标参数量，例如 1.6e8")
    ap.add_argument("--train_tokens", type=int, required=True)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--tokenizer", type=str, default="gpt2")
    ap.add_argument("--train_path", type=str, default="scaling/data/corpus/train.txt")
    ap.add_argument("--val_path", type=str, default="scaling/data/corpus/val.txt")
    ap.add_argument("--micro_bsz", type=int, default=1)
    ap.add_argument("--global_bsz", type=int, default=512)  # 通过 grad_accum * world_size * micro_bsz 实现
    ap.add_argument("--grad_accum", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--save_dir", type=str, default="runs/tmp")
    args = ap.parse_args()
    main(args)
