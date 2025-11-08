import json, pathlib

class JsonlLogger:
    def __init__(self, path: str):
        self.p = pathlib.Path(path)
        self.p.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.p.open("a", encoding="utf-8")
    def log(self, **kw):
        self.f.write(json.dumps(kw, ensure_ascii=False) + "\n")
        self.f.flush()
    def close(self):
        self.f.close()
