"""
Download the restricted eSEN checkpoint and print embedded config info.

Requires an environment variable HF_TOKEN with access to facebook/OMol25.
Uses the existing uv-managed virtual environment.
"""

from __future__ import annotations

import os
from pathlib import Path

import requests
import torch
from pprint import pprint


CKPT_URL = (
    "https://huggingface.co/facebook/OMol25/resolve/main/checkpoints/esen_sm_direct_all.pt"
)
CKPT_PATH = Path("checkpoints/esen_sm_direct_all.pt")


def main() -> None:

    checkpoint = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    print("checkpoint type:", type(checkpoint))
    print("checkpoint:", dir(checkpoint))

    print("\nmodel_config:", type(checkpoint.model_config))
    model_config = checkpoint.model_config
    print(type(model_config))
    pprint(model_config)
    
    tasks_config = checkpoint.tasks_config
    for i, cfg in enumerate(tasks_config):
        print(f"task[{i}]", type(cfg))
        pprint(cfg)


if __name__ == "__main__":
    main()

