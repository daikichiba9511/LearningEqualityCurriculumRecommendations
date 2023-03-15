"""finetuning SBert with supervised manner
"""
# =========================================================================================
# Libraries
# =========================================================================================
from __future__ import annotations

import gc
import math
import os
import random
import time
import warnings

warnings.filterwarnings("ignore")
import pdb

import numpy as np
import pandas as pd

# import tokenizers
import torch
import torch.nn as nn

# import transformers
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW

# from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, DataCollatorWithPadding, get_cosine_schedule_with_warmup

from src.constants import ROOT
from src.model import SBert
from src.train import AWP

load_dotenv()

# %env TOKENIZERS_PARALLELISM=true
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================================================
# Configurations
# =========================================================================================
class CFG:
    print_freq = 500
    num_workers = 24
    model = "sentence-transformers/all-MiniLM-L6-v2"
    uns_model_weight = f"{ROOT}/output/exp003-all-MiniLM-L6-v2-MNR-tuned-fold"
    tokenizer = AutoTokenizer.from_pretrained(model)
    gradient_checkpointing = False
    num_cycles = 0.5
    warmup_ratio = 0.1
    epochs = 8
    encoder_lr = 1e-5
    decoder_lr = 1e-4
    eps = 1e-6
    betas = (0.9, 0.999)
    batch_size = 32
    weight_decay = 0.01
    max_grad_norm = 0.012
    max_len = 512
    n_folds = 4
    seed = 42
    adv_lr = 1e-5
    adv_eps = 3
    debug = False


# =========================================================================================
# Seed everything for deterministic results
# =========================================================================================
def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# =========================================================================================
# F2 score metric
# =========================================================================================
def f2_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    tp = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    fp = np.array([len(x[1] - x[0]) for x in zip(y_true, y_pred)])
    fn = np.array([len(x[0] - x[1]) for x in zip(y_true, y_pred)])
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    f2 = tp / (tp + 0.2 * fp + 0.8 * fn)
    return round(f2.mean(), 4)


# =========================================================================================
# Data Loading
# =========================================================================================
def read_data(cfg: CFG) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(f"{ROOT}/output/lecr-unsupervised-train-set/train_1200.pqt")
    # train.to_parquet(f"{ROOT}/output/lecr-unsupervised-train-set/train_1200.pqt", index=False)
    print("traih.shape: ", train.shape)

    train = train.sample(n=int(len(train) * 0.8), random_state=42)

    if cfg.debug:
        train = train.sample(n=3000).reset_index(drop=True)
    print("Train: ")
    print(train.columns)
    print(train)

    train["title1"].fillna("Title does not exist", inplace=True)
    train["title2"].fillna("Title does not exist", inplace=True)
    train["topics_parent_description"].fillna("Topics parent description does not exist", inplace=True)
    train["topics_children_description"].fillna("Topics children description does not exist", inplace=True)

    correlations = pd.read_csv(f"{ROOT}/input/learning-equality-curriculum-recommendations/correlations.csv")
    print("Correlation: ")
    print(correlations)

    # Create feature column
    train["text"] = (
        train["title1"]
        + "[SEP]"
        + train["title2"]
        + "[SEP]"
        + train["topics_parent_description"]
        + "[SEP]"
        + train["topics_children_description"]
    )
    print(" ")
    print("-" * 50)
    print(f"train.shape: {train.shape}")
    print(f"correlations.shape: {correlations.shape}")
    return train, correlations


# =========================================================================================
# CV split
# =========================================================================================
def cv_split(train, cfg):
    kfold = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    for num, (train_index, val_index) in enumerate(kfold.split(train, train["target"], train["topics_ids"])):
        train.loc[val_index, "fold"] = int(num)
    train["fold"] = train["fold"].astype(int)
    return train


# =========================================================================================
# Get max length
# =========================================================================================
def get_max_length(train, cfg):
    lengths = []
    for text in tqdm(train["text"].fillna("").values, total=len(train)):
        length = len(cfg.tokenizer(text, add_special_tokens=False)["input_ids"])
        lengths.append(length)
    cfg.max_len = max(lengths) + 2  # cls & sep
    print(f"max_len: {cfg.max_len}")


# =========================================================================================
# Prepare input, tokenize
# =========================================================================================
def prepare_input(text, cfg):
    inputs = cfg.tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens=True,
        max_length=min(cfg.max_len, 512),
        # max_length=512,
        pad_to_max_length=True,
        truncation=True,
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


# =========================================================================================
# Custom dataset
# =========================================================================================
class custom_dataset(Dataset):
    def __init__(self, df, cfg):
        self.cfg = cfg
        self.texts = df["text"].values
        self.labels = df["target"].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.texts[item], self.cfg)
        label = torch.tensor(self.labels[item], dtype=torch.float)
        return inputs, label


# =========================================================================================
# Collate function for training
# =========================================================================================
def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:, :mask_len]
    return inputs


# =========================================================================================
# Mean pooling class
# =========================================================================================
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


# =========================================================================================
# Model
# =========================================================================================
class custom_model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        self.config.hidden_dropout = 0.0
        self.config.hidden_dropout_prob = 0.0
        self.config.attention_dropout = 0.0
        self.config.attention_probs_dropout_prob = 0.0
        # self.transformer = AutoModel.from_pretrained(cfg.model, config=self.config)
        self.model = SBert(cfg.model)
        self.model.transformer.config = self.config
        if self.cfg.gradient_checkpointing:
            if isinstance(self.model, SBert):
                self.model.transformer.gradient_checkpointing_enable()
            else:
                self.model.gradient_checkpointing_enable()
        self.pool = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.fc)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.model(**inputs)
        last_hidden_state = outputs.output.last_hidden_state
        feature = self.pool(last_hidden_state, inputs["attention_mask"])
        return feature

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        feature = self.feature(inputs)
        output = self.fc(feature)
        return output


# =========================================================================================
# Helper functions
# =========================================================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))


# =========================================================================================
# Train function loop
# =========================================================================================
def train_fn(
    train_loader: DataLoader,
    model: nn.Module,
    criterion: nn.modules._Loss,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    cfg: CFG,
    awp: AWP,
):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    awp.scaler = scaler
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    for step, (inputs, target) in enumerate(train_loader):
        # print("Before: ", inputs)
        # pdb.set_trace()
        inputs = collate(inputs)
        # print("After: ", inputs)
        # pdb.set_trace()
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        target = target.to(device)
        batch_size = target.size(0)
        with torch.cuda.amp.autocast(enabled=True):
            y_preds = model(inputs)
            loss = criterion(y_preds.view(-1), target)
        losses.update(loss.item(), batch_size)

        scaler.scale(loss).backward()
        awp.attack_backward(inputs["input_ids"], target, inputs["attention_mask"], step)
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        global_step += 1
        scheduler.step()
        end = time.time()
        if step % cfg.print_freq == 0 or step == (len(train_loader) - 1):
            print(
                "Epoch: [{0}][{1}/{2}] "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) "
                "Grad: {grad_norm:.4f}  "
                "LR: {lr:.8f}  ".format(
                    epoch + 1,
                    step,
                    len(train_loader),
                    remain=timeSince(start, float(step + 1) / len(train_loader)),
                    loss=losses,
                    grad_norm=grad_norm,
                    lr=scheduler.get_lr()[0],
                )
            )
    return losses.avg


# =========================================================================================
# Valid function loop
# =========================================================================================
def valid_fn(
    valid_loader: DataLoader, model: nn.Module, criterion: nn.modules._Loss, device: torch.device, cfg: CFG
) -> tuple[float, np.ndarray]:
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, (inputs, target) in enumerate(valid_loader):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        target = target.to(device)
        batch_size = target.size(0)
        with torch.no_grad():
            y_preds = model(inputs)
        loss = criterion(y_preds.view(-1), target)
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.sigmoid().squeeze().to("cpu").numpy().reshape(-1))
        end = time.time()
        if step % cfg.print_freq == 0 or step == (len(valid_loader) - 1):
            print(
                "EVAL: [{0}/{1}] "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) ".format(
                    step, len(valid_loader), loss=losses, remain=timeSince(start, float(step + 1) / len(valid_loader))
                )
            )
    predictions = np.concatenate(preds, axis=0)
    return losses.avg, predictions


# =========================================================================================
# Get best threshold
# =========================================================================================
def get_best_threshold(x_val, val_predictions, correlations):
    best_score = 0
    best_threshold = -0.1
    for thres in np.arange(0.001, 0.1, 0.001):
        x_val["predictions"] = np.where(val_predictions > thres, 1, 0)
        x_val1 = x_val[x_val["predictions"] == 1]
        x_val1 = x_val1.groupby(["topics_ids"])["content_ids"].unique().reset_index()
        x_val1["content_ids"] = x_val1["content_ids"].apply(lambda x: " ".join(x))
        x_val1.columns = ["topic_id", "predictions"]
        x_val0 = pd.Series(x_val["topics_ids"].unique())
        x_val0 = x_val0[~x_val0.isin(x_val1["topic_id"])]
        x_val0 = pd.DataFrame({"topic_id": x_val0.values, "predictions": ""})
        x_val_r = pd.concat([x_val1, x_val0], axis=0, ignore_index=True)
        x_val_r = x_val_r.merge(correlations, how="left", on="topic_id")
        score = f2_score(x_val_r["content_ids"], x_val_r["predictions"])
        if score > best_score:
            best_score = score
            best_threshold = thres
    return best_score, best_threshold


# =========================================================================================
# prapare train dataset
# =========================================================================================
def build_training_set(topics: pd.DataFrame, content: pd.DataFrame, cfg: CFG):
    # Create lists for training
    topics_ids = []
    content_ids = []
    title1 = []
    title2 = []
    targets = []

    # Iterate over each topic
    for k in tqdm(range(len(topics))):
        row = topics.iloc[k]
        topics_id = row["id"]
        topics_title = row["title"]
        predictions = row["predictions"].split(" ")
        ground_truth = row["content_ids"].split(" ")
        for pred in predictions:
            content_title = content.loc[pred, "title"]
            topics_ids.append(topics_id)
            content_ids.append(pred)
            title1.append(topics_title)
            title2.append(content_title)
            # If pred is in ground truth, 1 else 0
            if pred in ground_truth:
                targets.append(1)
            else:
                targets.append(0)
    # Build training dataset
    train = pd.DataFrame(
        {"topics_ids": topics_ids, "content_ids": content_ids, "title1": title1, "title2": title2, "target": targets}
    )
    return train


# =========================================================================================
# Train & Evaluate
# =========================================================================================
def train_and_evaluate_one_fold(train, correlations, fold, cfg):
    print(" ")
    print(f"========== fold: {fold} training ==========")
    # Split train & validation
    x_train = train[train["fold"] != fold]
    x_val = train[train["fold"] == fold]
    valid_labels = x_val["target"].values
    train_dataset = custom_dataset(x_train, cfg)
    valid_dataset = custom_dataset(x_val, cfg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    # Get model
    model = custom_model(cfg)
    state = torch.load(CFG.uns_model_weight + f"{fold}/model_state_dict.pt", map_location=torch.device("cpu"))
    model.model.load_state_dict(state)
    model.to(device)

    # Optimizer
    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "lr": encoder_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
                "lr": encoder_lr,
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if "model" not in n],
                "lr": decoder_lr,
                "weight_decay": 0.0,
            },
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(
        model, encoder_lr=cfg.encoder_lr, decoder_lr=cfg.decoder_lr, weight_decay=cfg.weight_decay
    )
    optimizer = AdamW(optimizer_parameters, lr=cfg.encoder_lr, eps=cfg.eps, betas=cfg.betas)
    num_train_steps = int(len(x_train) / cfg.batch_size * cfg.epochs)
    num_warmup_steps = num_train_steps * cfg.warmup_ratio
    # Scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
    )

    awp = AWP(
        model=model,
        optimizer=optimizer,
        adv_lr=cfg.adv_lr,
        adv_eps=cfg.adv_eps,
        start_epoch=num_train_steps // cfg.epochs,
        scaler=None,
    )

    # Training & Validation loop
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    best_score = 0
    val_predictions = None
    for epoch in range(cfg.epochs):
        start_time = time.time()
        # Train
        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device, cfg, awp)
        # Validation
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, device, cfg)
        # Compute f2_score
        score, threshold = get_best_threshold(x_val, predictions, correlations)
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s"
        )
        print(f"Epoch {epoch+1} - Score: {score:.4f} - Threshold: {threshold:.5f}")
        if score > best_score:
            best_score = score
            print(f"Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model")
            torch.save(
                {"model": model.state_dict(), "predictions": predictions},
                f"exp003-best-{cfg.model.replace('/', '-')}_fold{fold}_{cfg.seed}.pth",
            )
            val_predictions = predictions
        torch.save(
            {"model": model.state_dict(), "predictions": predictions},
            f"exp003-last-{cfg.model.replace('/', '-')}_fold{fold}_{cfg.seed}.pth",
        )

    torch.cuda.empty_cache()
    gc.collect()
    if val_predictions is not None:
        # Get best threshold
        best_score, best_threshold = get_best_threshold(x_val, val_predictions, correlations)
        print(f"Our CV score is {best_score} using a threshold of {best_threshold}")


def main() -> None:
    cfg = CFG()
    # Seed everything
    seed_everything(cfg.seed)
    # Read data
    train, correlations = read_data(cfg)
    # CV split
    cv_split(train, cfg)
    # Get max length
    get_max_length(train, cfg)
    # Train and evaluate one fold
    for fold in range(cfg.n_folds):
        train_and_evaluate_one_fold(train, correlations, fold, cfg)


if __name__ == "__main__":
    main()
