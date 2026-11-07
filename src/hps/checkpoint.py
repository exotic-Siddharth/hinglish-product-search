"""Resumable training state, so a restart costs minutes instead of the whole run."""
import hashlib
import json
import os
from pathlib import Path

import torch

# Arguments that do not change what is learned, so a resume may differ in them.
NON_SEMANTIC = {"out", "ckpt_every"}


def run_fingerprint(args, heldout, files=()):
    """Identifies what a run learns from: its settings, its lexicon split and the bytes of
    the files that shape its data (lexicon, brand list, generator code). A checkpoint is
    only ever resumed by a run with the same fingerprint."""
    digests = [hashlib.md5(Path(f).read_bytes()).hexdigest() if Path(f).exists() else None for f in files]
    payload = json.dumps({"args": {k: v for k, v in args.items() if k not in NON_SEMANTIC},
                          "heldout": sorted(heldout), "files": digests}, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()


def save_ckpt(path, model, opt, sched, fingerprint, **extra):
    """Write to a temp file first: a crash mid-save never corrupts the last checkpoint.
    The torch RNG state is saved too, so dropout continues as in an uninterrupted run."""
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "rng": torch.get_rng_state(), "fingerprint": fingerprint, **extra}, tmp)
    os.replace(tmp, path)


def load_ckpt(path, model, opt, sched, fingerprint):
    """Restores the training state in place and returns the saved extras, or None
    if there is no checkpoint for this exact run (the caller then starts fresh)."""
    if not path.exists():
        return None
    ck = torch.load(path)
    if ck.pop("fingerprint", None) != fingerprint:
        print(f"ignoring {path.name}: it belongs to a different run", flush=True)
        return None
    model.load_state_dict(ck.pop("model"))
    opt.load_state_dict(ck.pop("opt"))
    sched.load_state_dict(ck.pop("sched"))
    if "rng" in ck:
        torch.set_rng_state(ck.pop("rng"))
    return ck
