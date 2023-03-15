"""
Referenct:
[1]
https://www.kaggle.com/code/ragnar123/lecr-unsupervised-train-set-public/notebook
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
from dataclasses import dataclass

warnings.filterwarnings("ignore")
from pathlib import Path

import cupy as cp
import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tokenizers
import torch
import torch.nn as nn
import transformers
from cuml.metrics import pairwise_distances
from cuml.neighbors import NearestNeighbors
from dotenv import load_dotenv
from IPython.display import Markdown, display
from torch.optim import AdamW
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, DataCollatorWithPadding, get_cosine_schedule_with_warmup
from transformers.utils import ModelOutput

from src.constants import ROOT

load_dotenv()

# %env TOKENIZERS_PARALLELISM=false
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================================================
# Configurations
# =========================================================================================
class CFG:
    num_workers = 4
    # model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_weight_path = f"{ROOT}/output/exp003-all-MiniLM-L6-v2-MNR-tuned-fold0/model_state_dict.pt"
    tokenizer = AutoTokenizer.from_pretrained(model)
    batch_size = 32
    top_n = 1200
    seed = 42
    topic_cols = ["title", "description", "parent_description", "children_description"]
    content_cols = ["title", "description", "text"]
    max_length = 128
    debug = False
    num_proc = 2


def topic_tokenize(batch, tokenizer, topic_cols, max_length):
    """
    Tokenizes the dataset on the specific columns, truncated/padded to a max length.
    Adds the suffix "_content" to the input ids and attention mask of the content texts.

    Returns dummy labels that make the evaluation work in Trainer.
    """
    sep = tokenizer.sep_token

    topic_texts = [sep.join(cols) for cols in zip(*[batch[c] for c in topic_cols])]
    tokenized_topic = tokenizer(topic_texts, truncation=True, max_length=max_length, padding=False)

    # Remove token_type_ids. They will just cause errors.
    if "token_type_ids" in tokenized_topic:
        del tokenized_topic["token_type_ids"]

    return {
        **{f"{k}": v for k, v in tokenized_topic.items()},
    }


def content_tokenize(batch, tokenizer, content_cols, max_length):
    """
    Tokenizes the dataset on the specific columns, truncated/padded to a max length.
    Adds the suffix "_content" to the input ids and attention mask of the content texts.

    Returns dummy labels that make the evaluation work in Trainer.
    """
    sep = tokenizer.sep_token

    content_texts = [sep.join(cols) for cols in zip(*[batch[c] for c in content_cols])]
    tokenized_content = tokenizer(content_texts, truncation=True, max_length=max_length, padding=False)

    # Remove token_type_ids. They will just cause errors.
    if "token_type_ids" in tokenized_content:
        del tokenized_content["token_type_ids"]

    return {
        **{f"{k}": v for k, v in tokenized_content.items()},
    }


def get_topic_tokenized_ds(ds, tokenizer, max_length=64, debug=False):
    if not isinstance(ds, datasets.Dataset):
        ds = datasets.Dataset.from_pandas(ds)
    if debug:
        ds = ds.shuffle().select(range(5000))

    topic_cols = [f"topic_{c}" for c in CFG.topic_cols]
    tokenized_ds = ds.map(
        topic_tokenize,
        batched=True,
        fn_kwargs=dict(
            tokenizer=tokenizer,
            topic_cols=topic_cols,
            max_length=max_length,
        ),
        remove_columns=ds.column_names,
        num_proc=CFG.num_proc,
    )

    return tokenized_ds


def get_content_tokenized_ds(ds, tokenizer, max_length=64, debug=False):
    if not isinstance(ds, datasets.Dataset):
        ds = datasets.Dataset.from_pandas(ds)
    if debug:
        ds = ds.shuffle().select(range(5000))

    content_cols = [f"content_{c}" for c in CFG.content_cols]

    tokenized_ds = ds.map(
        content_tokenize,
        batched=True,
        fn_kwargs=dict(
            tokenizer=tokenizer,
            content_cols=content_cols,
            max_length=max_length,
        ),
        remove_columns=ds.column_names,
        num_proc=CFG.num_proc,
    )

    return tokenized_ds


data_dir = Path("./input/learning-equality-curriculum-recommendations")
topics_df = pd.read_csv(data_dir / "topics.csv", index_col=0).fillna({"title": "", "description": ""})
content_df = pd.read_csv(data_dir / "content.csv", index_col=0).fillna("")
correlations_df = pd.read_csv(data_dir / "correlations.csv", index_col=0)


def print_markdown(md):
    display(Markdown(md))


class Topic:
    def __init__(self, topic_id):
        self.id = topic_id

    @property
    def parent(self):
        parent_id = topics_df.loc[self.id].parent
        if pd.isna(parent_id):
            return None
        else:
            return Topic(parent_id)

    @property
    def ancestors(self):
        """祖先"""
        ancestors = []
        parent = self.parent
        while parent is not None:
            ancestors.append(parent)
            parent = parent.parent
        return ancestors

    @property
    def siblings(self):
        """兄弟・姉妹"""
        if not self.parent:
            return []
        else:
            return [topic for topic in self.parent.children if topic != self]

    @property
    def content(self):
        if self.id in correlations_df.index:
            return [ContentItem(content_id) for content_id in correlations_df.loc[self.id].content_ids.split()]
        else:
            return tuple([]) if self.has_content else []

    def get_breadcrumbs(self, separator=" >> ", include_self=True, include_root=True):
        ancestors = self.ancestors
        if include_self:
            ancestors = [self] + ancestors
        if not include_root:
            ancestors = ancestors[:-1]
        return separator.join(reversed([a.title for a in ancestors]))

    @property
    def children(self):
        return [Topic(child_id) for child_id in topics_df[topics_df.parent == self.id].index]

    def subtree_markdown(self, depth=0):
        markdown = "  " * depth + "- " + self.title + "\n"
        for child in self.children:
            markdown += child.subtree_markdown(depth=depth + 1)
        for content in self.content:
            markdown += ("  " * (depth + 1) + "- " + "[" + content.kind.title() + "] " + content.title) + "\n"
        return markdown

    def __eq__(self, other):
        if not isinstance(other, Topic):
            return False
        return self.id == other.id

    def __getattr__(self, name):
        return topics_df.loc[self.id][name]

    def __str__(self):
        return self.title

    def __repr__(self):
        return f'<Topic(id={self.id}, title="{self.title}")>'


class ContentItem:
    def __init__(self, content_id):
        self.id = content_id

    @property
    def topics(self):
        return [
            Topic(topic_id)
            for topic_id in topics_df.loc[
                correlations_df[correlations_df.content_ids.str.contains(self.id)].index
            ].index
        ]

    def __getattr__(self, name):
        return content_df.loc[self.id][name]

    def __str__(self):
        return self.title

    def __repr__(self):
        return f'<ContentItem(id={self.id}, title="{self.title}")>'

    def __eq__(self, other):
        if not isinstance(other, ContentItem):
            return False
        return self.id == other.id

    def get_all_breadcrumbs(self, separator=" >> ", include_root=True):
        breadcrumbs = []
        for topic in self.topics:
            new_breadcrumb = topic.get_breadcrumbs(separator=separator, include_root=include_root)
            if new_breadcrumb:
                new_breadcrumb = new_breadcrumb + separator + self.title
            else:
                new_breadcrumb = self.title
            breadcrumbs.append(new_breadcrumb)
        return breadcrumbs


# =========================================================================================
# Data Loading
# =========================================================================================
def read_data(cfg):
    topics = pd.read_csv(f"{ROOT}/input/learning-equality-curriculum-recommendations/topics.csv")
    content = pd.read_csv(f"{ROOT}/input/learning-equality-curriculum-recommendations/content.csv")
    correlations = pd.read_csv(f"{ROOT}/input/learning-equality-curriculum-recommendations/correlations.csv")
    # Fillna titles
    topics["title"].fillna("", inplace=True)
    content["title"].fillna("", inplace=True)
    # Fillna descriptions
    topics["description"].fillna("", inplace=True)
    content["description"].fillna("", inplace=True)
    # Sort by title length to make inference faster
    topics["length"] = topics["title"].apply(lambda x: len(x))
    content["length"] = content["title"].apply(lambda x: len(x))
    topics.sort_values("length", inplace=True)
    content.sort_values("length", inplace=True)

    topics = topics.rename(columns={"id": "topic_id", "description": "topic_description"})
    content = content.rename(columns={"description": "content_description"})
    print(topics)
    """
    add column (parent description & children description)
    """
    id_end = 20
    data = []
    for idx, topic_id in tqdm(enumerate(topics["topic_id"]), total=len(topics)):
        # print(topics.query("topic_id == @topic_id"))
        topic = Topic(topic_id)
        if topic.parent is None:
            parent_topic_description = ""
        else:
            parent_id = str(topic.parent.id)
            parent_idx = int(topics.query("topic_id == @parent_id").index[0])
            parent_topic_description = topics.iloc[parent_idx]["topic_description"]

        children_topic_descriptions = [t.description for t in topic.children]
        if children_topic_descriptions:
            children_topic_description = "[SEP]".join(children_topic_descriptions)
        else:
            children_topic_description = ""

        data.append(
            {
                "topic_parent_description": parent_topic_description,
                "topic_children_description": children_topic_description,
                "topic_id": topic_id,
            }
        )
    topics = topics.merge(pd.DataFrame(data), how="left", on="topic_id")

    # Drop cols
    topics.drop(
        [
            # "description",
            "channel",
            "category",
            "level",
            "language",
            "parent",
            "has_content",
            "length",
        ],
        axis=1,
        inplace=True,
    )
    content.drop(
        [
            # "description",
            "kind",
            "language",
            # "text",
            "copyright_holder",
            "license",
            "length",
        ],
        axis=1,
        inplace=True,
    )
    topics = topics.rename(columns={"topic_id": "id"})
    content = content.fillna("No content text")
    # Reset index
    topics.reset_index(drop=True, inplace=True)
    content.reset_index(drop=True, inplace=True)
    print(" ")
    print("-" * 50)
    print(f"topics.shape: {topics.shape}")
    print(topics.columns)
    print(topics)
    print(f"content.shape: {content.shape}")
    print(content.columns)
    print(content)
    print(f"correlations.shape: {correlations.shape}")
    print(correlations)
    return topics, content, correlations


# =========================================================================================
# Prepare input, tokenize
# =========================================================================================
# def prepare_input(text, cfg):
#     inputs = cfg.tokenizer.encode_plus(
#         text,
#         return_tensors=None,
#         add_special_tokens=True,
#     )
#     for k, v in inputs.items():
#         inputs[k] = torch.tensor(v, dtype=torch.long)
#     return inputs


# =========================================================================================
# Unsupervised dataset
# =========================================================================================
# class uns_dataset(Dataset):
#     def __init__(self, df, cfg):
#         self.cfg = cfg
#         self.texts = df["title"].values
#
#     def __len__(self):
#         return len(self.texts)
#
#     def __getitem__(self, item):
#         inputs = prepare_input(self.texts[item], self.cfg)
#         return inputs


class UnsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: CFG, kind: str = "topic") -> None:
        self.cfg = cfg
        tokenizer = AutoTokenizer.from_pretrained(CFG.model)
        if kind == "topic":
            df = df.rename(columns={"title": "topic_title"})
            tokenized_ds = get_topic_tokenized_ds(df, tokenizer, max_length=CFG.max_length, debug=CFG.debug)
            self.tokenized_ds = tokenized_ds
        elif kind == "content":
            df = df.rename(columns={"title": "content_title", "text": "content_text"})
            tokenized_ds = get_content_tokenized_ds(df, tokenizer, max_length=CFG.max_length, debug=CFG.debug)
            self.tokenized_ds = tokenized_ds
        else:
            raise ValueError(f"Expected topic / content, but got {kind}")

    def __len__(self) -> int:
        return len(self.tokenized_ds)

    def __getitem__(self, item: int):
        #         inputs = prepare_uns_input(self.texts[item], self.cfg, as_tensor=False)
        inputs = self.tokenized_ds[item]
        return inputs


# =========================================================================================
# Mean pooling class
# =========================================================================================
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


# =========================================================================================
# Unsupervised model
# =========================================================================================
class uns_model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.model)
        self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        self.pool = MeanPooling()

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pool(last_hidden_state, inputs["attention_mask"])
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        return feature


@dataclass
class SBertOutput(ModelOutput):
    """
    Used for SBert Model Output
    """

    loss: torch.Tensor | None = None
    pooled_embeddings: torch.Tensor | None = None


class SBert(torch.nn.Module):
    """
    Basic SBert wrapper. Gets output embeddings and averages them, taking into account the mask.
    """

    def __init__(self, model_name):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)

    @staticmethod
    def mean_pooling(token_embeddings, attention_mask):
        """
        Average the output embeddings using the attention mask
        to ignore certain tokens.
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.transformer(input_ids, attention_mask=attention_mask, **kwargs)

        return SBertOutput(
            loss=None,  # loss is calculated in `compute_loss`, but needed here as a placeholder
            pooled_embeddings=self.mean_pooling(outputs[0], attention_mask),
        )


# =========================================================================================
# Get embeddings
# =========================================================================================
def get_embeddings(loader, model, device):
    model.eval()
    preds = []
    for step, inputs in enumerate(tqdm(loader)):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(**inputs).pooled_embeddings
        preds.append(y_preds.to("cpu").numpy())
    preds = np.concatenate(preds)
    return preds


# =========================================================================================
# Get the amount of positive classes based on the total
# =========================================================================================
def get_pos_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    int_true = np.array([len(x[0] & x[1]) / len(x[0]) for x in zip(y_true, y_pred)])
    return round(np.mean(int_true), 5)


# =========================================================================================
# Build our training set
# =========================================================================================
def build_training_set(topics, content, cfg):
    # Create lists for training
    topics_ids = []
    content_ids = []
    title1 = []
    title2 = []
    targets = []
    parent_descriptions = []
    children_descriptions = []
    # Iterate over each topic
    for k in tqdm(range(len(topics))):
        row = topics.iloc[k]
        topics_id = row["id"]
        topics_title = row["title"]
        predictions = row["predictions"].split(" ")
        ground_truth = row["content_ids"].split(" ")
        topics_parent_description = row["topic_parent_description"]
        topics_children_description = row["topic_children_description"]
        for pred in predictions:
            content_title = content.loc[pred, "title"]
            topics_ids.append(topics_id)
            content_ids.append(pred)
            title1.append(topics_title)
            title2.append(content_title)
            parent_descriptions.append(topics_parent_description)
            children_descriptions.append(topics_children_description)
            # If pred is in ground truth, 1 else 0
            if pred in ground_truth:
                targets.append(1)
            else:
                targets.append(0)
    # Build training dataset
    train = pd.DataFrame(
        {
            "topics_ids": topics_ids,
            "content_ids": content_ids,
            "title1": title1,
            "title2": title2,
            "target": targets,
            "topics_parent_description": parent_descriptions,
            "topics_children_description": children_descriptions,
        }
    )
    # Release memory
    del topics_ids, content_ids, title1, title2, targets
    gc.collect()
    return train


# =========================================================================================
# Get neighbors
# =========================================================================================
def get_neighbors(topics, content, cfg: CFG):
    # Create topics dataset
    topics_dataset = UnsDataset(topics, cfg)
    # Create content dataset
    content_dataset = UnsDataset(content, cfg, kind="content")
    # Create topics and content dataloaders
    topics_loader = DataLoader(
        topics_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=cfg.tokenizer, padding="longest"),
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    content_loader = DataLoader(
        content_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=cfg.tokenizer, padding="longest"),
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    # Create unsupervised model to extract embeddings
    # model = uns_model(cfg)
    model = SBert(model_name=cfg.model)
    state = torch.load(cfg.model_weight_path, map_location=torch.device("cpu"))
    model.load_state_dict(state)
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
    for k in tqdm(range(len(indices))):
        pred = indices[k]
        p = " ".join([content.loc[ind, "id"] for ind in pred.get()])
        predictions.append(p)
    topics["predictions"] = predictions
    # Release memory
    del topics_preds_gpu, content_preds_gpu, neighbors_model, predictions, indices, model
    gc.collect()
    return topics, content


def main() -> None:
    # Read data
    topics, content, correlations = read_data(CFG)

    # Run nearest neighbors
    topics, content = get_neighbors(topics, content, CFG)
    # Merge with target and comput max positive score
    topics = topics.merge(correlations, how="inner", left_on=["id"], right_on=["topic_id"])
    pos_score = get_pos_score(topics["content_ids"], topics["predictions"])
    print(f"Our max positive score is {pos_score}")
    # We can delete correlations
    del correlations
    gc.collect()
    # Set id as index for content
    content.set_index("id", inplace=True)
    # Build training set
    train = build_training_set(topics, content, CFG)
    print(f"Our training set has {len(train)} rows")
    # Save train set to disk to train on another notebook
    train.to_csv(f"{ROOT}/output/lecr-unsupervised-train-set/train_{CFG.top_n}.csv", index=False)
    train.to_parquet(f"{ROOT}/output/lecr-unsupervised-train-set/train_{CFG.top_n}.pqt", index=False)
    train.head()


if __name__ == "__main__":
    main()
