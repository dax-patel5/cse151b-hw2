import torch
import torch.nn as nn

# Define RNN Model
class RNNModel(nn.Module):
    # [AI-assisted: Claude Code]
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # x: (B, T) character indices. Teacher forcing: the ground-truth
        # character at every step is the input to the next step, and the
        # logits for the character following the sequence come from the
        # final hidden state.
        emb = self.embedding(x)               # (B, T, E)
        out, _ = self.rnn(emb)                # (B, T, H)
        logits = self.fc(out[:, -1, :])       # (B, vocab_size)
        return logits
