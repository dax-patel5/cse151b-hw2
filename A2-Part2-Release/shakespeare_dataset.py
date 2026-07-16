import torch
from torch.utils.data import Dataset

class ShakespeareDataset(Dataset):
    """Wraps pre-built (sequence, next-char) tensors for use with a DataLoader.
    [AI-assisted: Claude Code]


    Args:
        X (torch.LongTensor): (N, seq_len) input character sequences.
        y (torch.LongTensor): (N,) index of the character following each sequence.
    """

    def __init__(self, X, y):
        assert len(X) == len(y)
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
