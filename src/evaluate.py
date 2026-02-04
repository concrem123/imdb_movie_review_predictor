import numpy as np
from sklearn.metrics import classification_report
from datasets import load_from_disk
from transformers import Trainer, DistilBertForSequenceClassification
from pathlib import Path
import yaml
from transformers import TrainingArguments
from transformers import AutoTokenizer, DataCollatorWithPadding



def evaluate(cfg):

    # Resolve processed dataset path relative to project root
    ROOT = Path(__file__).resolve().parent.parent
    processed_path = (ROOT / cfg["data"]["processed_path"]).resolve()

    dataset = load_from_disk(str(processed_path))

    # Resolve output_dir relative to project root
    output_dir = (ROOT / cfg["training"]["output_dir"]).resolve()


    training_args = TrainingArguments(
        output_dir=str(output_dir / "eval"),
        per_device_eval_batch_size=cfg["training"]["batch_size"],
        fp16=False,
        )
    
    model = DistilBertForSequenceClassification.from_pretrained(
     str(output_dir / "final"))
    
    trainer = Trainer(
        model=model,
        args=training_args,
    )

    
    trainer = Trainer(model=model)

    predictions = trainer.predict(dataset["test"])
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = dataset["test"]["labels"]

    label_col = "labels" if "labels" in dataset["test"].column_names else "label"
    true_labels = dataset["test"][label_col]

    print(classification_report(true_labels, pred_labels))


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    config_path = ROOT / "configs/distilbert.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg)