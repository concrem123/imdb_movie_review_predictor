from datasets import load_dataset, DatasetDict
from transformers import DistilBertTokenizer
from pathlib import Path
import yaml


def preprocess_and_save(cfg):
    # Resolve project root and dataset paths
    ROOT = Path(__file__).resolve().parent.parent
    processed_path = (ROOT / cfg["data"]["processed_path"]).resolve()
    processed_debug_path = (ROOT / cfg["data"]["processed_path_debug"]).resolve()

    tokenizer = DistilBertTokenizer.from_pretrained(cfg["model"]["name"])

    # Load original IMDB train split
    dataset = load_dataset("imdb", split="train")

    # Step 1: train 70% + temp 30%
    train_temp = dataset.train_test_split(test_size=0.3, seed=42)

    # Step 2: validation 10%, test 20%
    val_test = train_temp["test"].train_test_split(test_size=2/3, seed=42)

    # Combine into DatasetDict
    dataset_dict = DatasetDict({
        "train": train_temp["train"],
        "validation": val_test["train"],
        "test": val_test["test"],
    })

    # Tokenization function
    def preprocess_function(examples):
        return tokenizer(
            examples[cfg["data"]["text_column"]],
            truncation=True,
            padding="max_length",
            max_length=cfg["model"]["max_length"],
        )

    # Apply tokenization to all splits
    encoded_with_text = dataset_dict.map(preprocess_function, batched=True, desc="Tokenizing")

    # Ensure directories exist
    processed_debug_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    # Save dataset with text (for debugging)
    encoded_with_text.save_to_disk(str(processed_debug_path))

    # Remove text column for training
    encoded_dataset = encoded_with_text.remove_columns([cfg["data"]["text_column"]])
    encoded_dataset.save_to_disk(str(processed_path))


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    config_path = ROOT / "configs/distilbert.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    preprocess_and_save(cfg)

