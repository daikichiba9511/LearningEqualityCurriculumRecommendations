import numpy as np
from datasets import formatting, load_dataset
from datasets.utils.logging import disable_progress_bar
from transformers import AutoTokenizer

from src.constants import OUTPUT_DIR

disable_progress_bar()


def tokenize(
    batch: formatting.formatting.LazyBatch, column_name: str, tokenizer: AutoTokenizer
) -> formatting.formatting.LazyBatch:
    tokenized = tokenizer(batch[column_name], truncation=False)

    tokenized["length"] = [len(ids) for ids in tokenized.input_ids]
    unk = tokenizer.unk_token_id
    tokenized["num_unk"] = [len([x for x in ids if x == unk]) for ids in tokenized.input_ids]

    return tokenized


def main() -> None:
    model_names = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ]
    columns_to_check = ["topic_title", "content_title"]

    combined = load_dataset("parquet", data_files=str(OUTPUT_DIR / "combined2.pqt"), split="train")

    names_and_columns = [(name, column) for name in model_names for column in columns_to_check]
    for (name, col) in names_and_columns:

        tokenizer = AutoTokenizer.from_pretrained(name)
        tokenized = combined.map(
            tokenize,
            batched=True,
            num_proc=4,
            fn_kwargs=dict(column_name=col, tokenizer=tokenizer),
            desc=f"Tokenizing {col} using {name}",
        )

        quantiles = [0.5, 0.75, 0.9, 0.95]
        length_quantiles = np.quantile(tokenized["length"], quantiles)
        unk_quantiles = np.quantile(tokenized["num_unk"], quantiles)

        print("\n\n")
        print(f"Column {col} - tokenizer {name}".center(30, "*"), "\n")
        # Quantile 0.5 means 50% of the values are less than this value
        # Quantile 0.95 means 95% of the values are less than this value
        for q, l in zip(quantiles, length_quantiles):
            print(f"Token length at quantile {q}: {l}")

        print("\n\n")

        for q, u in zip(quantiles, unk_quantiles):
            print(f"Num unk tokens at quantile {q}: {u}")


if __name__ == "__main__":
    main()
