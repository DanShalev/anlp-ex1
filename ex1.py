import os

import evaluate
import numpy as np
import wandb
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# Initialize Weights & Biases
wandb.init(project="mrpc-paraphrase-detection", name="bert-base-uncased")

# Load dataset
dataset = load_dataset("glue", "mrpc")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


# Tokenize dataset
def preprocess(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


encoded_dataset = dataset.map(preprocess, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)


# Define metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="no",
    logging_strategy="steps",
    logging_steps=1,
    learning_rate=2e-5,  # You can change this to tune
    per_device_train_batch_size=4,  # Change to tune
    per_device_eval_batch_size=4,
    num_train_epochs=1,  # Change to tune
    weight_decay=0.01,
    report_to="wandb",
    load_best_model_at_end=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Evaluate
eval_result = trainer.evaluate()
print("Validation Results:", eval_result)
