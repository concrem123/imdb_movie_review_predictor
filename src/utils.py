from datasets import load_from_disk
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
config_path = ROOT / "configs/distilbert.yaml"

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

ROOT = Path(__file__).resolve().parent.parent
processed_path = (ROOT / cfg["data"]["processed_path"]).resolve()

print(f"Loading dataset from: {processed_path}")

dataset = load_from_disk(str(processed_path))

print(dataset['test'].features)

true_labels = dataset["test"]["label"]