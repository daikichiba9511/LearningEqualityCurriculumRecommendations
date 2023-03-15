"""from exp001

Reference
[1]
https://www.kaggle.com/code/karakasatarik/0-459-single-model-inference-w-postprocessing
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
from dataclasses import dataclass
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
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizer,
    get_cosine_schedule_with_warmup,
)
from transformers.utils import ModelOutput

# from src.utils import cos_sim

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
    # uns_model = f"{ROOT}/input/paraphrasemultilingualmpnetbasev2/all-distilroberta-v1"
    uns_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sup_model = f"{ROOT}/input/xlmroberta/xlm-roberta-base"
    sup_model_state_weight = f"{ROOT}/output/xlm-roberta-base_fold0_42.pth"
    uns_model_state_weight = f"{ROOT}/output/all-MiniLM-L6-v2-MNR-tuned-fold0/model_state_dict.pt"
    uns_tokenizer = AutoTokenizer.from_pretrained(uns_model)
    sup_tokenizer = AutoTokenizer.from_pretrained(sup_model)
    gradient_checkpointing = False
    batch_size = 32
    n_folds = 5
    top_n = 1200
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
def prepare_uns_input(text: str, cfg: CFG, as_tensor: bool = True):
    inputs = cfg.uns_tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens=True,
    )
    if as_tensor:
        inputs = {k: torch.tensor(v, dtype=torch.long) for k, v in inputs.items()}
    return inputs


# =========================================================================================
# cosine similarity
# =========================================================================================
def cos_sim(a, b):
    # From https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/util.py#L31
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


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
        inputs = prepare_uns_input(self.texts[item], self.cfg, as_tensor=False)
        return inputs


# =========================================================================================
# Prepare input, tokenize
# =========================================================================================
def prepare_sup_input(text: str, cfg: CFG, as_tensor: bool = True):
    inputs = cfg.sup_tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens=True,
    )
    if as_tensor:
        inputs = {k: torch.tensor(v, dtype=torch.long) for k, v in inputs.items()}
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
        inputs = prepare_sup_input(self.texts[item], self.cfg, as_tensor=False)
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
def get_pos_and_tt_ids(input_ids, device):
    """
    This is a weird thing needed when using a distributed setup.
    If you don't have it, you face this: https://github.com/huggingface/accelerate/issues/97
    """
    position_ids = torch.arange(0, input_ids.shape[-1], device=device).expand((1, -1))
    token_type_ids = torch.zeros(
        input_ids.shape,
        dtype=torch.long,
        device=position_ids.device,
    )

    return {"position_ids": position_ids, "token_type_ids": token_type_ids}


def get_embeddings(loader: DataLoader, model: UnsModel | SBert, device: torch.device) -> np.ndarray:
    model.eval()

    if isinstance(model, SBert):
        preds_embeds = []
        for step, batch in enumerate(tqdm(loader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            # print("input_ids shape: ", batch["input_ids"].shape)
            # print("input_ids max value: ", max(max(x) for x in batch["input_ids"]))
            # print(dir(model.transformer))
            with torch.inference_mode():
                embeds = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **get_pos_and_tt_ids(batch["input_ids"], device),
                ).pooled_embeddings
            preds_embeds.append(embeds.detach().to("cpu").numpy())
        preds_embeds = np.concatenate(preds_embeds)
        return preds_embeds

    if isinstance(model, UnsModel):
        preds = []
        for step, inputs in enumerate(tqdm(loader)):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                y_preds = model(inputs)
            preds.append(y_preds.to("cpu").numpy())
        preds = np.concatenate(preds)
        return preds

    raise TypeError(f"Expected SBert | UnsModel, but got {type(model)}")


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
@dataclass
class MNRTopicCollator:

    tokenizer: PreTrainedTokenizer
    pad_to_multiple_of: int = 8
    max_length: int = 512

    def __call__(self, features):

        longest_topic = max([len(x["input_ids"]) for x in features])

        pad_token_id = self.tokenizer.pad_token_id

        input_ids_topic = [
            x["input_ids"] + [pad_token_id] * (min(longest_topic, self.max_length) - len(x["input_ids"]))
            for x in features
        ]
        attention_mask_topic = [
            x["attention_mask"] + [0] * (min(longest_topic, self.max_length) - len(x["attention_mask"]))
            for x in features
        ]

        return {
            "input_ids": torch.tensor(input_ids_topic),
            "attention_mask": torch.tensor(attention_mask_topic),
        }


@dataclass
class MNRContentCollator:

    tokenizer: PreTrainedTokenizer
    pad_to_multiple_of: int = 8
    max_length: int = 512

    def __call__(self, features):

        longest_content = max([len(x["input_ids"]) for x in features])

        pad_token_id = self.tokenizer.pad_token_id

        input_ids_content = [
            x["input_ids"] + [pad_token_id] * (min(longest_content, self.max_length) - len(x["input_ids"]))
            for x in features
        ]
        attention_mask_content = [
            x["attention_mask"] + [0] * (min(longest_content, self.max_length) - len(x["attention_mask"]))
            for x in features
        ]

        return {
            "input_ids": torch.tensor(input_ids_content),
            "attention_mask": torch.tensor(attention_mask_content),
        }


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
        collate_fn=MNRTopicCollator(tokenizer=cfg.uns_tokenizer, max_length=512, pad_to_multiple_of=8),
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    content_loader = DataLoader(
        content_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=MNRContentCollator(tokenizer=cfg.uns_tokenizer, max_length=512, pad_to_multiple_of=8),
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    # Create unsupervised model to extract embeddings
    # model = UnsModel(cfg).to(device)
    model = SBert(model_name=CFG.uns_model).to(device)
    # print("Vocab size: ", model.transformer.config.vocab_size)
    state = torch.load(CFG.uns_model_state_weight, map_location=torch.device("cpu"))
    model.load_state_dict(state)

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
    state = torch.load(CFG.sup_model_state_weight, map_location=torch.device("cpu"))
    model.load_state_dict(state["model"])
    prediction = inference_fn(test_loader, model, device)
    # Release memory
    torch.cuda.empty_cache()
    del test_dataset, test_loader, model, state
    gc.collect()

    # Use threshold
    test["probs"] = prediction
    test["predictions"] = np.where(prediction > cfg.threshold, 1, 0)
    test = test.sort_values("probs", acending=False)
    predicted = test[test["predictions"] == 1].groupby("topics_ids").agg(list).reset_index()
    # PP from Ref1.
    no_pos = test.groupby("topics_ids").head(1)
    no_pos = no_pos[no_pos["predictions"] == 0].groupby("topics_ids")["content_ids"].agg(list).reset_index()
    submissions = pd.concat([predicted, no_pos]).reset_index(drop=True)
    submissions["content_ids"] = submissions["content_ids"].apply(lambda x: " ".join(x))
    submissions.columns = ["topic_id", "content_ids"]

    return submissions


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
    sub = inference(test, CFG)
    print(f"Sub: \n{sub}")
    save_path = "submission.csv"
    sub.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
