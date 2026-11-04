"""Resumable training state, so a restart costs minutes instead of the whole run."""
import hashlib
import json
import os

import torch


def run_fingerprint(args, heldout):
    """Identifies a run's settings and lexicon split, so a checkpoint is only ever
    resumed by the run that wrote it, never by one with a different configuration."""
    payload = json.dumps({"args": args, "heldout": sorted(heldout)}, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()


def save_ckpt(path, model, opt, sched, fingerprint, **extra):
    """Write to a temp file first: a crash mid-save never corrupts the last checkpoint."""
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "fingerprint": fingerprint, **extra}, tmp)
    os.replace(tmp, path)


def load_ckpt(path, model, opt, sched, fingerprint):
    """Restores the three objects in place and returns the saved extras, or None
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
    return ck
