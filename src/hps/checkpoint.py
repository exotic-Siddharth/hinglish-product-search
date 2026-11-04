"""Resumable training state, so a restart costs minutes instead of the whole run."""
import os

import torch


def save_ckpt(path, model, opt, sched, **extra):
    """Write to a temp file first: a crash mid-save never corrupts the last checkpoint."""
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), **extra}, tmp)
    os.replace(tmp, path)


def load_ckpt(path, model, opt, sched):
    """Restores the three objects in place and returns the saved extras."""
    ck = torch.load(path)
    model.load_state_dict(ck.pop("model"))
    opt.load_state_dict(ck.pop("opt"))
    sched.load_state_dict(ck.pop("sched"))
    return ck
