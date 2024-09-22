import torch
import torch.nn as nn

class Seq2SeqModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(Seq2SeqModel, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        encoder_output, (hidden, cell) = self.encoder(x)
        decoder_output, _ = self.decoder(encoder_output, (hidden, cell))
        output = self.fc(decoder_output)
        return output
