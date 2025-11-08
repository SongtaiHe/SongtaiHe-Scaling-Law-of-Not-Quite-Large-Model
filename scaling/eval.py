import argparse, torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from datasets import load_dataset

def main(args):
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 从目录加载配置与权重；若提供 --from_ckpt 则覆盖权重
    model = GPT2LMHeadModel.from_pretrained(args.model_dir)
    if args.from_ckpt:
        sd = torch.load(args.from_ckpt, map_location="cpu")
        model.load_state_dict(sd, strict=False)

    ds = load_dataset("text", data_files={"val": args.val_path})["val"]
    ds = ds.map(lambda ex: tok(ex["text"], truncation=True, max_length=args.seq_len),
                batched=True, remove_columns=["text"]).with_format("torch")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.bsz, shuffle=False, collate_fn=lambda b: tok.pad(b, return_tensors="pt")
    )

    model.eval()
    losses, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            out = model(**batch, labels=batch["input_ids"])
            losses += out.loss.item() * len(batch["input_ids"])
            n += len(batch["input_ids"])
    ce = losses / max(n, 1)
    bpb = ce / 0.69314718056  # nats -> bits

    print({"val_ce": ce, "val_bpb": bpb})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True, help="模型目录（包含 config.json 等）")
    ap.add_argument("--from_ckpt", type=str, help="可选：权重文件路径，例如 runs/.../ckpt.pt")
    ap.add_argument("--val_path", type=str, default="scaling/data/corpus/val.txt")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--tokenizer", type=str, default="gpt2")
    ap.add_argument("--bsz", type=int, default=4)
    args = ap.parse_args()
    main(args)
