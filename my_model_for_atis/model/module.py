import torch.nn as nn


class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class SlotClassifier(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, num_slot_labels, dropout_rate=0.):
    # def __init__(self, input_dim, lstm_hidden_dim, num_slot_labels, lstm_dropout_rate=0., dropout_rate=0.):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        # self.lstm_dropout = nn.Dropout(lstm_dropout_rate)

        self.bi_lstm = nn.LSTM(input_size=input_dim,
                               hidden_size=lstm_hidden_dim,
                            #    num_layers=1,
                            #    bias=True,
                               batch_first=True,
                            #    dropout=0,
                               bidirectional=True,
                            #    proj_size=0
                               )

        self.linear = nn.Linear(lstm_hidden_dim * 2, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)

        x, _ = self.bi_lstm(x, None)

        return self.linear(x)
