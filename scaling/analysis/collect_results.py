import json, csv, pathlib, sys


def main(root="runs/grid", out_csv="analysis/results.csv"):
    root = pathlib.Path(root)
    rows = []
    for final in root.rglob("final.json"):
        try:
            obj = json.loads(final.read_text())
            N = obj.get("N")
            D = obj.get("D")
            val = obj.get("best_val_loss")
            if N and D and val:
                rows.append((N/1e6, D/1e6, val))
        except Exception:
            pass
    out = pathlib.Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N_million","D_million","val_loss"])
        for r in sorted(rows):
            w.writerow(r)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv)>1 else "runs/grid"
    out  = sys.argv[2] if len(sys.argv)>2 else "analysis/results.csv"
    main(root, out)
