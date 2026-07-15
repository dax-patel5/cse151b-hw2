"""Generate the three sample texts (temp0.5.txt, temp1.txt, temp2.txt) from the
best performing LSTM, as required by question 5(d).

Usage: python generate_samples.py [--model models/trained_lstm_model.pth]
"""
import argparse
import torch

from generate import generate_text
from util import encode_text
from config import load_config
from shakespeare_lstm import LSTMModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/trained_lstm_model.pth')
    parser.add_argument('--config', type=str, default='configs/base_lstm_config.yaml')
    parser.add_argument('--data', type=str, default='data/tiny_shakespeare.txt')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    encoded_text, vocab_size, char_to_idx, idx_to_char = encode_text(args.data)
    config = load_config(args.config)

    model = LSTMModel(vocab_size=vocab_size,
                      embed_size=config['embed_size'],
                      hidden_size=config['hidden_size'],
                      num_layers=config['num_layers'])
    model.load_state_dict(torch.load(args.model, map_location=device))

    for temp in (0.5, 1, 2):
        text = generate_text(model, device, char_to_idx, idx_to_char, max_len=1000, temp=temp)
        fname = f"temp{temp}.txt"
        with open(fname, 'w') as f:
            f.write(text)
        print(f"wrote {fname} ({len(text)} chars)")


if __name__ == '__main__':
    main()
