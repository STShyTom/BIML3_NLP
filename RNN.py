import torch.nn as nn
import torch.nn.functional as F
import torch

class RNN(torch.nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, combined_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.emb_layer = nn.Linear(input_size, emb_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.combined_layer = nn.Linear(emb_size + hidden_size, combined_size)
        self.i20_layer = nn.Linear(combined_size, output_size)
        self.i2h_layer = nn.Linear(combined_size, hidden_size)     
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x, hidden):
        embedding = self.emb_layer(x)
        combined = self.combined_layer(torch.cat((embedding, hidden), 1))
        i20 = self.combined_layer(combined)
        i2h = self.combined_layer(combined)
        output = self.i20_layer(i20)
        output = self.softmax(output)
        hidden = self.hidden_layer(i2h)

        return output, hidden
    
    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)
        