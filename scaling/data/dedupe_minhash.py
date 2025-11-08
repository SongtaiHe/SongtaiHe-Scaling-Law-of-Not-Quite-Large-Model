import argparse, pathlib
from datasketch import MinHash, MinHashLSH

def minhash(text, num_perm=128):
    mh = MinHash(num_perm=num_perm)
    for token in text.split():
        mh.update(token.encode("utf-8"))
    return mh

def dedupe(in_path: str, out_path: str, threshold=0.8):
    p_in, p_out = pathlib.Path(in_path), pathlib.Path(out_path)
    lines = p_in.read_text().splitlines()
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    kept = []
    for i, line in enumerate(lines):
        mh = minhash(line)
        if not lsh.query(mh):
            lsh.insert(str(i), mh)
            kept.append(line)
    p_out.write_text("\n".join(kept))
    print({"input": len(lines), "kept": len(kept)})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="i", required=True)
    ap.add_argument("--out", dest="o", required=True)
    ap.add_argument("--thr", type=float, default=0.8)
    args = ap.parse_args()
    dedupe(args.i, args.o, args.thr)
