import argparse, datasets, random, pathlib
from datasets import load_dataset

# 示例：用 wikitext-103 + oscar-en 小片段，快速跑通
SOURCES = [
    ("wikitext", {"name": "wikitext-103-raw-v1"}),
    ("oscar", {"name": "unshuffled_deduplicated_en"})
]

def main(out_dir: str, train_ratio=0.8, val_ratio=0.1, seed=42, n_samples=2_000_000):
    random.seed(seed)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    texts = []
    for name, kw in SOURCES:
        ds = load_dataset(name, **kw, split="train", streaming=True)
        for i, ex in enumerate(ds):
            text = ex.get("text") or ex.get("content") or ""
            if text and len(text) > 50:
                texts.append(text)
            if len(texts) >= n_samples:
                break
        if len(texts) >= n_samples:
            break

    random.shuffle(texts)
    n = len(texts)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train, val, test = texts[:n_train], texts[n_train:n_train+n_val], texts[n_train+n_val:]

    (out/"train.txt").write_text("\n\n".join(train))
    (out/"val.txt").write_text("\n\n".join(val))
    (out/"test.txt").write_text("\n\n".join(test))
    print({"train": len(train), "val": len(val), "test": len(test)})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="./data/corpus")
    ap.add_argument("--samples", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.out, n_samples=args.samples, seed=args.seed)
