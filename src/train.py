import yaml
from datasets import load_from_disk
from transformers import TrainingArguments, Trainer, DistilBertForSequenceClassification
from pathlib import Path

def train(cfg):
    dataset = load_from_disk(cfg["data"]["processed_path"])

    model = DistilBertForSequenceClassification.from_pretrained(
        cfg["model"]["name"],
        num_labels=cfg["model"]["num_labels"]
    )

    
    training_args = TrainingArguments(
        output_dir=cfg["training"]["output_dir"],
        eval_strategy="epoch",
        per_device_train_batch_size=cfg["training"]["batch_size"],
        num_train_epochs=cfg["training"]["epochs"],
        learning_rate=float(cfg["training"]["learning_rate"]),
        weight_decay=cfg["training"]["weight_decay"],
        logging_dir=f"{cfg['training']['output_dir']}/logs",
        report_to="tensorboard"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"]
    )

    trainer.train()
    trainer.save_model(f"{cfg['training']['output_dir']}/final")


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    config_path = ROOT / "configs/distilbert.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    train(cfg)
