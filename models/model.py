from . import rnn, lstm, gru

models = {
    "Naive_LSTM": lstm.NaiveLSTM,
    "Bi_LSTM": lstm.BiLSTM,
    "Bi_LSTM_Dropout": lstm.BiLSTM_Dropout,
    "Bi_LSTM_Dropout_2L": lstm.BiLSTM_Dropout_2L,
    "Naive_GRU": gru.NaiveGRU,
    "Bi_GRU": gru.BiGRU,
    "Bi_GRU_Dropout": gru.BiGRU_Dropout,
    "Naive_RNN": rnn.NaiveRNN,
}


def model(name):
    return models[name]
