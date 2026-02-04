import yaml
from datasets import load_from_disk
from transformers import TrainingArguments, Trainer, DistilBertForSequenceClassification
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
import torch

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"f1": f1_score(labels, preds)}


def build_optimizer(model, cfg):
    return torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
    )

def train(cfg):
    # Resolve processed dataset path relative to project root
    ROOT = Path(__file__).resolve().parent.parent
    processed_path = (ROOT / cfg["data"]["processed_path"]).resolve()

    dataset = load_from_disk(str(processed_path))

    model = DistilBertForSequenceClassification.from_pretrained(
        cfg["model"]["name"],
        num_labels=cfg["model"]["num_labels"],
    )

    # Resolve output_dir relative to project root
    output_dir = (ROOT / cfg["training"]["output_dir"]).resolve()

    training_args = TrainingArguments(
        output_dir=str(output_dir),

        # evaluation & logging
        evaluation_strategy=cfg["training"].get("evaluation_strategy", "epoch"),
        logging_strategy="steps",
        logging_steps=cfg["training"].get("logging_steps", 100),

        # training parameters
        per_device_train_batch_size=cfg["training"]["batch_size"],
        per_device_eval_batch_size=cfg["training"]["batch_size"],
        num_train_epochs=cfg["training"]["epochs"],
        learning_rate=float(cfg["training"]["learning_rate"]),

        # optimizer
        optim=cfg["training"]["optimizer"], 

        # model selection
        load_best_model_at_end=cfg["training"]["load_best_model_at_end"],
        metric_for_best_model=cfg["training"].get("metric_for_best_model", "eval_loss"),
        greater_is_better=False,

        # output & logging
        logging_dir=str(output_dir / "logs"),
        report_to="tensorboard",

        # save strategy
        save_strategy=cfg["training"].get("save_strategy", "epoch"),
        save_total_limit=cfg["training"].get("save_total_limit", 5),

        # GPU friendly
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,  # Enables F1
    )

    trainer.optimizer = build_optimizer(model, cfg)
    
    trainer.train()
    trainer.save_model(str(output_dir / "final"))


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    config_path = ROOT / "configs/distilbert.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    train(cfg)
