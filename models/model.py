from . import rnn, lstm


def model(name):
    if name == "naive_lstm":
        return lstm.NaiveLSTM
    if name == "bi_lstm":
        return lstm.BiLSTM
    if name == "naive_rnn":
        return rnn.NaiveRNN
    else:
        print("Invalid model name!")
        exit()
