import json

import pandas as pd

from src.constants import OUTPUT_DIR

combined = pd.read_parquet(OUTPUT_DIR / "combined1.pqt")
groups = combined.groupby("content_id")["fold"].agg(list)

print(f"Before fixing folds: \n {groups.head()}")
print(f"Checking number of folds for each content_id. {set([len(set(x)) for x in groups.to_numpy()])}")

# content_idにcategory==sourceとのoverlapがあるのでfold=-1
source_content_ids = set(combined[combined["category"] == "source"]["content_id"])
mask = combined["content_id"].isin(source_content_ids)
combined.loc[mask, "fold"] = -1

groups = combined.groupby("content_id")["fold"].agg(list)
print(f"After fixing folds: \n {groups.head()}")
print(f"Checking number of folds for each content_id. {set([len(set(x)) for x in groups.to_numpy()])}")

fold_indices = {int(fold): combined[combined["fold"] == fold].index.to_list() for fold in combined["fold"].unique()}

with (OUTPUT_DIR / "training_idx.json").open("w") as fp:
    json.dump(fold_indices, fp)

combined.to_parquet(OUTPUT_DIR / "combined2.pqt", index=False)
print(combined["fold"].value_counts())
