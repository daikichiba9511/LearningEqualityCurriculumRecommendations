"""
Reference:
1. https://www.kaggle.com/code/theoviel/modeling-oriented-eda-building-a-good-cv-split
2. https://www.kaggle.com/code/nbroad/multiple-negatives-ranking-loss-lecr
"""
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm

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
contents["content_text"] = [x[:300] for x in contents["content_text"]]

correlations["content_id"] = [x.split() for x in correlations["content_ids"]]
exploded_corr = correlations.explode("content_id")

"""
# parent description & children description

## Reference:
[1] https://www.kaggle.com/code/jamiealexandre/tips-and-recommendations-from-hosts


define some helper functions and classes to aid with data traversal
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

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


"""
add column (parent description & children description)
"""
print("Topic")
print(topics)
print("Correlation")
print(correlations)
print("Contents")
print(contents)
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
        children_topic_description = " [SEP] ".join(children_topic_descriptions)
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

"""
make fold
"""
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

combined.to_parquet(OUTPUT_DIR / "combined_adding_pc_desc.pqt", index=False)
