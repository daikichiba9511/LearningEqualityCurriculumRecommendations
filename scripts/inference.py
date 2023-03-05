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
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
import tokenizers
import torch
import torch.nn as nn
import transformers
from cuml.metrics import pairwise_distances
from cuml.neighbors import NearestNeighbors
from dotenv import load_dotenv
from torch.optim import AdamW
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, DataCollatorWithPadding, get_cosine_schedule_with_warmup

ROOT = Path(".")

load_dotenv()

#%env TOKENIZERS_PARALLELISM=false
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================================================
# Configurations
# =========================================================================================
class CFG:
    print_freq = 3000
    num_workers = 4
    uns_model = f"{ROOT}/input/paraphrasemultilingualmpnetbasev2/all-distilroberta-v1"
    sup_model = f"{ROOT}/input/xlmroberta/xlm-roberta-base"
    model_state_weight = f"{ROOT}/output/xlm-roberta-base_fold0_42.pth"
    uns_tokenizer = AutoTokenizer.from_pretrained(uns_model)
    sup_tokenizer = AutoTokenizer.from_pretrained(sup_model)
    gradient_checkpointing = False
    batch_size = 32
    n_folds = 5
    top_n = 1000
    seed = 42
    threshold = 0.001


# =========================================================================================
# Data Loading
# =========================================================================================
def read_data(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    topics = pd.read_csv(f"{ROOT}/input/learning-equality-curriculum-recommendations/topics.csv")
    content = pd.read_csv(f"{ROOT}/input/learning-equality-curriculum-recommendations/content.csv")
    sample_submission = pd.read_csv(f"{ROOT}/input/learning-equality-curriculum-recommendations/sample_submission.csv")
    # Merge topics with sample submission to only infer test topics
    topics = topics.merge(sample_submission, how="inner", left_on="id", right_on="topic_id")
    # Fillna titles
    topics["title"].fillna("", inplace=True)
    content["title"].fillna("", inplace=True)
    # Sort by title length to make inference faster
    topics["length"] = topics["title"].apply(lambda x: len(x))
    content["length"] = content["title"].apply(lambda x: len(x))
    topics.sort_values("length", inplace=True)
    content.sort_values("length", inplace=True)

    topics.rename(columns={"description": "top"}, inplace=True)

    topics = topics.rename(
        columns={
            # "id": "topic_id",
            # "title": "topic_title",
            # "description": "topic_description",
            "language": "topic_language",
        }
    )
    content = content.rename(
        columns={
            # "id": "content_id",
            # "title": "content_title",
            # "description": "content_description",
            "text": "content_text",
            "language": "content_language",
        }
    )

    # Drop cols
    # topics.drop(
    #     [
    #         "description",
    #         "channel",
    #         "category",
    #         "level",
    #         "language",
    #         "parent",
    #         "has_content",
    #         "length",
    #         "topic_id",
    #         "content_ids",
    #     ],
    #     axis=1,
    #     inplace=True,
    # )
    # content.drop(
    #     ["description", "kind", "language", "text", "copyright_holder", "license", "length"], axis=1, inplace=True
    # )
    # Reset index
    topics.reset_index(drop=True, inplace=True)
    content.reset_index(drop=True, inplace=True)
    print(" ")
    print("-" * 50)
    print(f"topics.shape: {topics.shape}")
    print(f"content.shape: {content.shape}")
    return topics, content


# =========================================================================================
# Prepare input, tokenize
# =========================================================================================
def prepare_uns_input(text: str, cfg: CFG):
    """
    Returns:
        keys: input_ids, attention_mask
    """
    inputs = cfg.uns_tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens=True,
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


# =========================================================================================
# Unsupervised dataset
# =========================================================================================
class UnsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: CFG) -> None:
        self.cfg = cfg
        self.texts = df["title"].values

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, item: int):
        inputs = prepare_uns_input(self.texts[item], self.cfg)
        return inputs


# =========================================================================================
# Prepare input, tokenize
# =========================================================================================
def prepare_sup_input(text: str, cfg: CFG):
    inputs = cfg.sup_tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens=True,
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


# =========================================================================================
# Supervised dataset
# =========================================================================================
class SupDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: CFG) -> None:
        self.cfg = cfg
        self.texts = df["text"].values

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_sup_input(self.texts[item], self.cfg)
        return inputs


# =========================================================================================
# Mean pooling class
# =========================================================================================
class MeanPooling(nn.Module):
    def __init__(self) -> None:
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


# =========================================================================================
# Unsupervised model
# =========================================================================================
class UnsModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.uns_model)
        self.model = AutoModel.from_pretrained(cfg.uns_model, config=self.config)
        self.pool = MeanPooling()

    def feature(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pool(last_hidden_state, inputs["attention_mask"])
        return feature

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        feature = self.feature(inputs)
        return feature


# =========================================================================================
# Get embeddings
# =========================================================================================
def get_embeddings(loader: DataLoader, model: nn.Module, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    for step, inputs in enumerate(tqdm(loader)):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.to("cpu").numpy())
    preds = np.concatenate(preds)
    return preds


# =========================================================================================
# Get the amount of positive classes based on the total
# =========================================================================================
def get_pos_socre(y_true, y_pred) -> float:
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    int_true = np.array([len(x[0] & x[1]) / len(x[0]) for x in zip(y_true, y_pred)])
    return round(np.mean(int_true), 5)


# =========================================================================================
# Build our inference set
# =========================================================================================
def build_inference_set(topics: pd.DataFrame, content: pd.DataFrame, cfg) -> pd.DataFrame:
    # Create lists for training
    topics_ids = []
    content_ids = []
    title1 = []
    title2 = []
    topic_languages = []
    content_languates = []
    # Iterate over each topic
    for k in tqdm(range(len(topics))):
        row = topics.iloc[k]
        topics_id = row["id"]
        topics_title = row["title"]
        predictions = row["predictions"].split(" ")
        topic_language = row["topic_language"]
        for pred in predictions:
            content_title = content.loc[pred, "title"]
            content_language = content.loc[pred, "content_language"]
            topics_ids.append(topics_id)
            content_ids.append(pred)
            title1.append(topics_title)
            title2.append(content_title)
            topic_languages.append(topic_language)
            content_languates.append(content_language)
    # Build training dataset
    test = pd.DataFrame(
        {
            "topics_ids": topics_ids,
            "content_ids": content_ids,
            "title1": title1,
            "title2": title2,
            "topic_language": topic_languages,
            "content_language": content_languates,
        }
    )
    # Release memory
    del topics_ids, content_ids, title1, title2
    gc.collect()
    return test


# =========================================================================================
# Get neighbors
# =========================================================================================
def get_neighbors(topics: pd.DataFrame, content: pd.DataFrame, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Create topics dataset
    topics_dataset = UnsDataset(topics, cfg)
    # Create content dataset
    content_dataset = UnsDataset(content, cfg)
    # Create topics and content dataloaders
    topics_loader = DataLoader(
        topics_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=cfg.uns_tokenizer, padding="longest"),
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    content_loader = DataLoader(
        content_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=cfg.uns_tokenizer, padding="longest"),
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    # Create unsupervised model to extract embeddings
    model = UnsModel(cfg)
    model.to(device)
    # Predict topics
    topics_preds = get_embeddings(topics_loader, model, device)
    content_preds = get_embeddings(content_loader, model, device)
    # Transfer predictions to gpu
    topics_preds_gpu = cp.array(topics_preds)
    content_preds_gpu = cp.array(content_preds)

    # Release memory
    torch.cuda.empty_cache()
    del topics_dataset, content_dataset, topics_loader, content_loader, topics_preds, content_preds
    gc.collect()

    # KNN model
    print(" ")
    print("Training KNN model...")
    neighbors_model = NearestNeighbors(n_neighbors=cfg.top_n, metric="cosine")
    neighbors_model.fit(content_preds_gpu)
    indices = neighbors_model.kneighbors(topics_preds_gpu, return_distance=False)
    predictions = []
    for k in range(len(indices)):
        pred = indices[k]
        p = " ".join([content.loc[ind, "id"] for ind in pred.get()])
        predictions.append(p)
    topics["predictions"] = predictions

    # Release memory
    del topics_preds_gpu, content_preds_gpu, neighbors_model, predictions, indices, model
    gc.collect()

    return topics, content


# =========================================================================================
# Process test
# =========================================================================================
def preprocess_test(test: pd.DataFrame) -> pd.DataFrame:
    test["title1"].fillna("Title does not exist", inplace=True)
    test["title2"].fillna("Title does not exist", inplace=True)
    # Create feature column
    test["text"] = test["title1"] + "[SEP]" + test["title2"]
    # Drop titles
    test.drop(["title1", "title2"], axis=1, inplace=True)
    # Sort so inference is faster
    test["length"] = test["text"].apply(lambda x: len(x))
    test.sort_values("length", inplace=True)
    test.drop(["length"], axis=1, inplace=True)
    test.reset_index(drop=True, inplace=True)
    gc.collect()
    return test


# =========================================================================================
# Model
# =========================================================================================
class CustomModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.sup_model, output_hidden_states=True)
        self.config.hidden_dropout = 0.0
        self.config.hidden_dropout_prob = 0.0
        self.config.attention_dropout = 0.0
        self.config.attention_probs_dropout_prob = 0.0
        self.model = AutoModel.from_pretrained(cfg.sup_model, config=self.config)
        if self.cfg.gradient_checkpointing:
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
        last_hidden_state = outputs.last_hidden_state
        feature = self.pool(last_hidden_state, inputs["attention_mask"])
        return feature

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        feature = self.feature(inputs)
        output = self.fc(feature)
        return output


# =========================================================================================
# Inference function loop
# =========================================================================================
def inference_fn(test_loader: DataLoader, model: nn.Module, device: torch.device) -> np.ndarray:
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            y_preds = model(inputs)
        preds.append(y_preds.sigmoid().squeeze().to("cpu").numpy().reshape(-1))
    predictions = np.concatenate(preds)
    return predictions


# =========================================================================================
# Inference
# =========================================================================================
def inference(test: pd.DataFrame, cfg) -> pd.DataFrame:
    # Create dataset and loader
    test_dataset = SupDataset(test, cfg)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=cfg.sup_tokenizer, padding="longest"),
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    # Get model
    model = CustomModel(cfg)
    # Load weights
    state = torch.load(CFG.model_state_weight, map_location=torch.device("cpu"))
    model.load_state_dict(state["model"])
    prediction = inference_fn(test_loader, model, device)
    # Release memory
    torch.cuda.empty_cache()
    del test_dataset, test_loader, model, state
    gc.collect()

    # Use threshold
    test["probs"] = prediction
    test["predictions"] = np.where(prediction > cfg.threshold, 1, 0)
    test = test.merge(test.groupby("topics_ids", as_index=False)["probs"].max(), on="topics_ids", suffixes=["", "_max"])
    # print(test.head(30))
    # test = test[test["has_content"]]

    test1 = test[(test["predictions"] == 1) & (test["topic_language"] == test["content_language"])]
    test1 = test1.groupby(["topics_ids"])["content_ids"].unique().reset_index()
    test1["content_ids"] = test1["content_ids"].apply(lambda x: " ".join(x))
    test1.columns = ["topic_id", "content_ids"]

    test0 = pd.Series(test["topics_ids"].unique())
    test0 = test0[~test0.isin(test1["topic_id"])]
    test0 = pd.DataFrame({"topic_id": test0.values, "content_ids": ""})

    test_r = pd.concat([test1, test0], axis=0, ignore_index=True)

    return test_r


def main() -> None:
    # Read data
    topics, content = read_data(CFG)
    # Run nearest neighbors
    topics, content = get_neighbors(topics, content, CFG)
    gc.collect()
    # Set id as index for content
    content.set_index("id", inplace=True)
    # Build training set
    test = build_inference_set(topics, content, CFG)
    # Process test set
    test = preprocess_test(test)
    # Inference
    test_r = inference(test, CFG)
    print(f"Sub: \n{test_r}")
    save_path = "submission.csv"
    test_r.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
