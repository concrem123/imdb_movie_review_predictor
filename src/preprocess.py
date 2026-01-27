from datasets import load_dataset, DatasetDict
from transformers import DistilBertTokenizer
from pathlib import Path
import yaml

# Preprocess the dataset and save to disk
def preprocess_and_save(cfg):
    tokenizer = DistilBertTokenizer.from_pretrained(cfg["model"]["name"])

    # remove unsupervised data as its not required for training
    # Load only the original training split
    dataset = load_dataset("imdb", split="train")

    # Step 1: split train -> train (70%) + temp (30%)
    train_temp = dataset.train_test_split(
        test_size=0.3,
        seed=42
    )

    # Step 2: split temp -> validation (10%) + test (20%)
    # 10 / (10 + 20) = 1/3
    val_test = train_temp["test"].train_test_split(
        test_size=2/3,
        seed=42
    )

    # Final datasets
    train_dataset = train_temp["train"]
    val_dataset = val_test["train"]
    test_dataset = val_test["test"]

    # Combine into DatasetDict
    dataset_dict = DatasetDict({
        "train": train_temp["train"],
        "validation": val_test["train"],
        "test": val_test["test"],
    })

    def preprocess_function(examples):
        return tokenizer(
            examples[cfg["data"]["text_column"]],
            truncation=True,
            padding="max_length",
            max_length=cfg["model"]["max_length"]
        )

    encoded_with_text = dataset_dict.map(
        preprocess_function,
        batched=True,
    )

    # saved the data with text column for debugging purposes
    encoded_with_text.save_to_disk(cfg["data"]["processed_path_debug"])

    # remove text column for training data 
    encoded_dataset = encoded_with_text.remove_columns(["text"])
    encoded_dataset.save_to_disk(cfg["data"]["processed_path"])



if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    config_path = ROOT / "configs/distilbert.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    preprocess_and_save(cfg)
