import numpy as np
from sklearn.metrics import classification_report
from transformers import Trainer, DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained(
    "experiments/distilbert_v1/final"
)

trainer = Trainer(model=model)

predictions = trainer.predict(encoded_dataset["test"])
pred_labels = np.argmax(predictions.predictions, axis=1)
true_labels = encoded_dataset["test"]["labels"]

print(classification_report(true_labels, pred_labels))