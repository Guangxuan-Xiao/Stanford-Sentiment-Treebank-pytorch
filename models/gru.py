import torch.nn as nn


class NaiveGRU(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # text.shape = (sentence len, batch_size)
        embedded = self.embedding(text)
        # embedded.shape = (sentence len, batch size, embedded dim)
        output, hidden = self.gru(embedded)
        return self.fc(output[-1])


class BiGRU(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text):
        # text.shape = (sentence len, batch_size)
        embedded = self.embedding(text)
        # embedded.shape = (sentence len, batch size, embedded dim)
        output, hidden = self.gru(embedded)
        return self.fc(output[-1])


class BiGRU_Dropout(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text):
        # text.shape = (sentence len, batch_size)
        embedded = self.embedding(text)
        # embedded.shape = (sentence len, batch size, embedded dim)
        output, hidden = self.gru(embedded)
        return self.fc(output[-1])
