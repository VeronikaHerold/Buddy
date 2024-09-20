import json
import torch
from transformers import Trainer, TrainingArguments
from models.gpt2.responses import model, tokenizer
from training.feedback import prepare_data_with_feedback, load_feedback_data
from torch.utils.data import Dataset

class CustomFeedbackDataset(Dataset):
    def __init__(self, feedback_data, tokenizer, max_length=512):
        self.feedback_data = feedback_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.feedback_data)

    def __getitem__(self, idx):
        item = self.feedback_data[idx]
        input_text = item['input']
        output_text = item['output']
        feedback_score = item.get('feedback', 5)  # Standardwert 5 falls kein Feedback vorhanden

        encoding = self.tokenizer(
            input_text + self.tokenizer.eos_token + output_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids,
            'feedback': torch.tensor(feedback_score, dtype=torch.float)  # Feedback als zusätzlicher Faktor
        }
def fine_tune_gpt2_with_feedback():
    # Lade die Feedback-Daten
    feedback_data = load_feedback_data()

    # Erstelle das Dataset mit dem Feedback
    dataset = CustomFeedbackDataset(feedback_data, tokenizer)

    # Lade die Trainingsparameter aus der config.json
    with open("config/config.json", "r") as config_file:
        config = json.load(config_file)

    # Definiere die Trainingsparameter
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=config["train_epochs"],
        per_device_train_batch_size=config["batch_size"],
        save_steps=10_000,
        save_total_limit=2,
    )

    # Erstelle den Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        # Möglichkeit zur Gewichtung des Feedbacks implementieren
    )

    # Starte das Training
    trainer.train()
