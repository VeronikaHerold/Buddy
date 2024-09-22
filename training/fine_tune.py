import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import Dataset, DataLoader
from training.feedback import load_feedback_data

class SimpleNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        nn.init.kaiming_uniform_(self.fc1.weight)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_size, output_size)
        nn.init.kaiming_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

    def save_model(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_model(self, path: str) -> None:
        self.load_state_dict(torch.load(path))

def simple_tokenizer(text, vocab):
    return [vocab.get(word, vocab["<UNK>"]) for word in text.split()]

class CustomFeedbackDataset(Dataset):
    def __init__(self, feedback_data, vocab, max_length=512):
        self.feedback_data = feedback_data
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.feedback_data)

    def __getitem__(self, idx):
        item = self.feedback_data[idx]
        input_text = item['input']
        output_text = item['output']
        feedback_score = item.get('feedback', 5)

        input_tokens = simple_tokenizer(input_text, self.vocab)
        output_tokens = simple_tokenizer(output_text, self.vocab)

        input_tokens = input_tokens[:self.max_length] + [0] * (self.max_length - len(input_tokens))
        output_tokens = output_tokens[:self.max_length] + [0] * (self.max_length - len(output_tokens))

        return {
            'input_tokens': torch.tensor(input_tokens, dtype=torch.long),
            'output_tokens': torch.tensor(output_tokens, dtype=torch.long),
            'feedback': torch.tensor(feedback_score, dtype=torch.float)
        }

def build_vocab(feedback_data):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for item in feedback_data:
        for word in item['input'].split() + item['output'].split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

def fine_tune_custom_model():
    feedback_data = load_feedback_data("path/to/feedback.json")
    vocab = build_vocab(feedback_data)
    dataset = CustomFeedbackDataset(feedback_data, vocab)
    with open("config/config.json", "r") as config_file:
        config = json.load(config_file)

    input_size = 512  
    hidden_size = 128  
    output_size = len(vocab) 
    model = SimpleNN(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    for epoch in range(config["train_epochs"]):
        total_loss = 0
        for batch in dataloader:
            input_tokens = batch['input_tokens']
            output_tokens = batch['output_tokens']

            outputs = model(input_tokens.float())
            loss = criterion(outputs, output_tokens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{config['train_epochs']}, Verlust: {total_loss/len(dataloader)}")

    torch.save(model.state_dict(), "models/custom_model.pth")
    print("Modelltraining abgeschlossen und gespeichert.")