"""Journal partagé des runs (semaine 2, partie A).

Chaque run écrit UN petit fichier JSON dans results/runs/ (un fichier par run :
pas de conflit Git quand plusieurs personnes poussent en même temps).

Dans votre script d'entraînement :

    from runlog import RunLogger
    with RunLogger("B_lstm_resnet", model=model, seed=42, lr=1e-3, epochs=10) as run:
        ...entraînement...
        run.add(**evaluate(preds))          # scores de evaluate.py

Le temps, la mémoire GPU max, le nombre de paramètres (total / entraînables)
et le commit Git sont enregistrés automatiquement.

Voir le tableau de tous les runs :   python runlog.py
"""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import torch
except ImportError:  # permet d'afficher le tableau sans torch
    torch = None

RUNS_DIR = Path("results/runs")


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _git_commit():
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def _cuda():
    return torch is not None and torch.cuda.is_available()


class RunLogger:
    def __init__(self, name, model=None, runs_dir=RUNS_DIR, **config):
        self.name, self.model, self.runs_dir, self.config = name, model, Path(runs_dir), config
        self.metrics = {}

    def add(self, **metrics):
        self.metrics.update(metrics)

    def __enter__(self):
        self.t0 = time.time()
        if _cuda():
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, exc_type, exc, tb):
        rec = {
            "name": self.name,
            "date": datetime.now().isoformat(timespec="seconds"),
            "status": "ok" if exc_type is None else f"erreur: {exc_type.__name__}",
            "time_min": round((time.time() - self.t0) / 60, 2),
            "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20) if _cuda() else None,
            "git_commit": _git_commit(),
            "config": self.config,
            "metrics": self.metrics,
        }
        if self.model is not None:
            rec["params_total"], rec["params_trainable"] = count_params(self.model)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        path = self.runs_dir / f"{self.name}_{datetime.now():%Y%m%d_%H%M%S}.json"
        path.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
        print(f"[runlog] run enregistré : {path}")
        return False  # ne masque jamais une erreur


def show(runs_dir=RUNS_DIR):
    files = sorted(Path(runs_dir).glob("*.json"))
    if not files:
        print(f"Aucun run dans {runs_dir}")
        return
    cols = ["name", "status", "params_trainable", "time_min", "gpu_peak_mb", "BLEU-4", "CIDEr", "CLIPScore"]
    rows = []
    for f in files:
        r = json.loads(f.read_text())
        flat = {**r, **r.get("metrics", {})}
        rows.append([("-" if flat.get(c) is None else str(flat[c])) for c in cols])
    widths = [max(len(c), *(len(row[k]) for row in rows)) for k, c in enumerate(cols)]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for row in rows:
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))


if __name__ == "__main__":
    show(sys.argv[1] if len(sys.argv) > 1 else RUNS_DIR)