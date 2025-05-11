import argparse
import os

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate BERT model on MRPC dataset"
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=-1,
        help="Number of samples to use during training (-1 for all)",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=-1,
        help="Number of samples to use during validation (-1 for all)",
    )
    parser.add_argument(
        "--max_predict_samples",
        type=int,
        default=-1,
        help="Number of samples to use during prediction (-1 for all)",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Train batch size")
    parser.add_argument("--do_train", action="store_true", help="Run training")
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="Run prediction and generate predictions.txt",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="bert-base-uncased",
        help="Model path to use for prediction",
    )
    return parser.parse_args()


class MRPCTrainer:
    def __init__(self, args):
        self.args = args
        self.dataset = None
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.encoded_dataset = None
        self.data_collator = None

    def setup_wandb(self):
        """Initialize Weights & Biases for experiment tracking"""
        wandb.init(project="mrpc-paraphrase-detection", name="bert-base-uncased")

    def load_and_preprocess_data(self):
        """Load the MRPC dataset and preprocess it"""
        # Load dataset
        self.dataset = load_dataset("glue", "mrpc")
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

        # Tokenize dataset
        def preprocess(example):
            return self.tokenizer(
                example["sentence1"], example["sentence2"], truncation=True
            )

        self.encoded_dataset = self.dataset.map(preprocess, batched=True)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Limit dataset sizes if specified
        self._limit_dataset_sizes()

    def _limit_dataset_sizes(self):
        """Limit the size of datasets based on command line arguments"""
        if self.args.max_train_samples != -1:
            self.encoded_dataset["train"] = self.encoded_dataset["train"].select(
                range(self.args.max_train_samples)
            )
        if self.args.max_eval_samples != -1:
            self.encoded_dataset["validation"] = self.encoded_dataset[
                "validation"
            ].select(range(self.args.max_eval_samples))
        if self.args.max_predict_samples != -1:
            self.encoded_dataset["test"] = self.encoded_dataset["test"].select(
                range(self.args.max_predict_samples)
            )

    def compute_metrics(self, pred):
        """Compute evaluation metrics"""
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary"
        )
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    def setup_model(self):
        """Initialize the BERT model"""
        self.model = BertForSequenceClassification.from_pretrained(
            self.args.model_path, num_labels=2
        )

    def setup_trainer(self):
        """Configure and initialize the HuggingFace Trainer"""
        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
            save_strategy="no",
            logging_strategy="steps",
            logging_steps=1,
            learning_rate=self.args.lr,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            num_train_epochs=self.args.num_train_epochs,
            weight_decay=0.01,
            report_to="wandb",
            load_best_model_at_end=False,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.encoded_dataset["train"],
            eval_dataset=self.encoded_dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

    def train(self):
        """Train the model and evaluate on validation set"""
        self.trainer.train()
        eval_result = self.trainer.evaluate()
        print("Validation Results:", eval_result)

    def predict(self):
        """Generate predictions on test set and save to file"""
        # Load the model for prediction
        self.model = BertForSequenceClassification.from_pretrained(
            self.args.model_path, num_labels=2
        )
        self.trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )

        # Make predictions
        predictions = self.trainer.predict(self.encoded_dataset["test"])
        preds = np.argmax(predictions.predictions, axis=1)

        # Save predictions to file
        with open("predictions.txt", "w") as f:
            for pred in preds:
                f.write(f"{pred}\n")

    def run(self):
        """Main execution flow"""
        self.setup_wandb()
        self.load_and_preprocess_data()
        self.setup_model()

        if self.args.do_train:
            self.setup_trainer()
            self.train()

        if self.args.do_predict:
            self.predict()


def main():
    args = parse_args()
    trainer = MRPCTrainer(args)
    trainer.run()


if __name__ == "__main__":
    main()
