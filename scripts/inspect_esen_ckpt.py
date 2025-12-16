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
from omegaconf import OmegaConf
from fairchem.core.units.mlip_unit.utils import load_inference_model


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

    # instantiate the model
    model = load_inference_model(str(CKPT_PATH), return_checkpoint=False)
    n_params = sum(p.numel() for p in model.parameters())
    print("\nNumber of model parameters:", n_params, "\n")

    tasks_config = checkpoint.tasks_config
    for i, cfg in enumerate(tasks_config):
        print(f"\ntask[{i}]", type(cfg))
        # omegaconf.dictconfig.DictConfig -> dict
        cfg = OmegaConf.to_container(cfg)
        cfg.pop("element_references", None) # only in task[0]
        pprint(cfg)


if __name__ == "__main__":
    main()

