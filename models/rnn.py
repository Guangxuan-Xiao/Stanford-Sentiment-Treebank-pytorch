import torch.nn as nn


class NaiveRNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # text.shape = (sentence len, batch_size)
        embedded = self.embedding(text)
        # embedded.shape = (sentence len, batch size, embedded dim)
        output, hidden = self.rnn(embedded)
        return self.fc(output[-1])
