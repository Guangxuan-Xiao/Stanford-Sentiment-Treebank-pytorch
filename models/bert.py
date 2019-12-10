from transformers import BertModel
import torch.nn as nn
import torch


class BERTGRUSentiment(nn.Module):
    def __init__(
        self, hidden_dim, output_dim, n_layers, bidirectional=True, dropout=0.5
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        embedding_dim = self.bert.config.to_dict()["hidden_size"]
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=0 if n_layers < 2 else dropout,
        )
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        with torch.no_grad():
            embedded = self.bert(text)[0]
        _, hidden = self.gru(embedded)
        if self.gru.bidirectional:
            hidden = self.dropout(
                torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            )
        else:
            hidden = self.dropout(hidden[-1, :, :])
        return self.fc(hidden)
