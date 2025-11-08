from datasets import load_dataset
from transformers import AutoTokenizer

class TokenCounter:
    def __init__(self, tok_name: str = "gpt2"):
        self.tok = AutoTokenizer.from_pretrained(tok_name, use_fast=True)

    def count_text(self, text: str) -> int:
        return len(self.tok.encode(text))

    def count_dataset(self, dataset_name: str, split: str, field: str = "text", limit: int | None = None) -> int:
        ds = load_dataset(dataset_name, split=split, streaming=True)
        n = 0
        for i, ex in enumerate(ds):
            n += self.count_text(ex[field])
            if limit and i >= limit:
                break
        return n
