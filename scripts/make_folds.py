"""
Reference:
1. https://www.kaggle.com/code/theoviel/modeling-oriented-eda-building-a-good-cv-split
2. https://www.kaggle.com/code/nbroad/multiple-negatives-ranking-loss-lecr
"""
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.constants import DATA_DIR, OUTPUT_DIR
from src.metrics import iou

topics = pd.read_csv(DATA_DIR / "topics.csv")
correlations = pd.read_csv(DATA_DIR / "correlations.csv")
contents = pd.read_csv(DATA_DIR / "content.csv")

topics = topics.rename(
    columns={
        "id": "topic_id",
        "title": "topic_title",
        "description": "topic_description",
        "language": "topic_language",
    }
)
contents = contents.rename(
    columns={
        "id": "content_id",
        "title": "content_title",
        "description": "content_description",
        "text": "content_text",
        "language": "content_language",
    }
)

topics["topic_title"].fillna("No topic title", inplace=True)
topics["topic_description"].fillna("No topic description", inplace=True)

contents["content_title"].fillna("No content title", inplace=True)
contents["content_description"].fillna("No content description", inplace=True)
contents["content_text"].fillna("No content text", inplace=True)
contents["content_text"] = [[x[:300]] for x in contents["content_text"]]

correlations["content_id"] = [x.split() for x in correlations["content_ids"]]
exploded_corr = correlations.explode("content_id")

print(topics)
print(correlations)
print(contents)

topics_val = topics[topics["category"] != "source"][["channel", "topic_id"]]
topics_val = topics_val.merge(correlations, on="topic_id")
channel_val = topics_val.groupby("channel").agg(list).reset_index()
channel_val["content_ids"] = channel_val["content_ids"].apply(
    lambda x: list(np.unique(np.concatenate([x_.split(" ") for x_ in x])))
)
ious = np.zeros((len(channel_val), len(channel_val)))

for i in range(len(channel_val)):
    for j in range(i):
        iou_ij = iou(channel_val["content_ids"][i], channel_val["content_ids"][j])
        ious[i, j] = iou_ij
        ious[j, i] = iou_ij

G = nx.Graph(ious)
components = list([list(k) for k in nx.connected_components(G)])
for i, c in enumerate(components):
    print(f"Comp:{i} of len {len(c)}")

channel_val["group"] = 0
for i, c in enumerate(components):
    channel_val.loc[np.array(c), "group"] = i

combined = (
    topics.merge(exploded_corr, on="topic_id")
    .merge(contents, on="content_id")
    .merge(channel_val[["channel", "group"]], on="channel", how="left")
)
combined["fold"] = -1
print(f"Combined: \n {combined}")

print(combined.group.value_counts())

groups = combined.loc[~combined["group"].isna()].reset_index(drop=True)
print(f"Groups: \n {groups}")

gkf = GroupKFold(n_splits=4)
for fold, (_, valid_idx) in enumerate(gkf.split(groups, groups=groups["group"].astype(int))):
    groups.loc[valid_idx, "fold"] = fold

combined.loc[~combined["group"].isna(), "fold"] = groups["fold"].values

print("Fold Counts")
print(combined["fold"].value_counts())

combined[["channel", "fold"]].to_csv(OUTPUT_DIR / "folds.csv")
print(combined[["channel", "fold"]])

combined.to_parquet(OUTPUT_DIR / "combined1.pqt", index=False)
