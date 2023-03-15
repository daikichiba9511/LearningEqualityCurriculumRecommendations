from pathlib import Path

import torch

from src.constants import OUTPUT_DIR


class CFG:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 384
    outp_dirs: Path = OUTPUT_DIR / ("exp002-" + model_name.split("/")[-1] + "-MNR-tuned")

    folds: int = 4
    lr: float = 5e-5
    wd: float = 0.01
    warmup_ratio: float = 0.1
    epochs: int = 20
    evals_per_epoch: int = 2
    num_devices = torch.cuda.device_count()
    scheduler_type: str = "cosine"
    mixed_precision: str = "fp16"

    topic_cols = ["title", "description", "parent_description", "children_description"]
    content_cols = ["title", "description", "text"]
    max_length = 128
    num_proc = 24
    grad_accum = 1

    tokenized_ds_name = str(OUTPUT_DIR / "tokenized.pqt")
    use_wandb = True
    debug = False
    seed = 18
    log_per_epoch = 10

    metric_to_track = "recall@100"
