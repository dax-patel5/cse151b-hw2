import torch
import torch.nn as nn

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # x: (B, T) character indices. Teacher forcing: the ground-truth
        # character at every step is the input to the next step, and the
        # logits for the character following the sequence come from the
        # final hidden state.
        emb = self.embedding(x)               # (B, T, E)
        out, _ = self.lstm(emb)               # (B, T, H)
        logits = self.fc(out[:, -1, :])       # (B, vocab_size)
        return logits


class LSTMModelNoTeacherForcing(nn.Module):
    """Identical architecture to LSTMModel; only the forward pass differs.

    Only the first ground-truth character is fed in. Afterwards the argmax of
    the model's own output distribution at each step is embedded and used as
    the next input, so the model must learn to predict from its own sequence.
    """

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMModelNoTeacherForcing, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, return_all=False):
        # x: (B, T). Step 0 consumes the true first character; every later
        # step consumes the argmax of the previous step's prediction.
        B, T = x.shape
        inp = x[:, 0]                          # (B,)
        hidden = None
        all_logits = []
        for _ in range(T):
            emb = self.embedding(inp).unsqueeze(1)   # (B, 1, E)
            out, hidden = self.lstm(emb, hidden)     # (B, 1, H)
            logits = self.fc(out[:, -1, :])          # (B, vocab_size)
            all_logits.append(logits)
            inp = logits.argmax(dim=1)               # feed own prediction back

        if return_all:
            # (B, T, vocab_size): step t predicts the character at position t+1
            return torch.stack(all_logits, dim=1)
        return all_logits[-1]
