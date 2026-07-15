import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.data import random_split
import os

from util import encode_text, create_sequences
from shakespeare_dataset import ShakespeareDataset
from shakespeare_lstm import LSTMModel, LSTMModelNoTeacherForcing
from shakespeare_rnn import RNNModel
from config import load_config
from train import train, eval

def main():
    #argument parsing 
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='rnn',
                        help='Specify the experiment that you want to run')
    parser.add_argument('--data_dir', type=str, default='./data', help= 'Specify the directory that your shakespeare data is located')
    parser.add_argument('--config', type=str, default=None,
                        help='Optional path to a config yaml, overrides the default config for the experiment')
    args = parser.parse_args()

    #load configs
    config_path = ""
    if (args.experiment == 'rnn'):
        config_path = "configs/base_rnn_config.yaml"
    elif (args.experiment == 'lstm'):
        config_path = "configs/base_lstm_config.yaml"
    elif (args.experiment == 'noteacherforcing'):
        # separate config: the best LSTM uses hidden_size 300 while the best
        # no-teacher-forcing model uses the standard hidden_size 150
        config_path = "configs/base_noteacherforcing_config.yaml"
    if args.config is not None:
        config_path = args.config
    config = load_config(config_path)
    config['experiment'] = args.experiment
    config.setdefault('name', args.experiment)

    #load data
    input_file_path = f'{args.data_dir}/tiny_shakespeare.txt' 
    print('ENCODED TEXT DATA')

    encoded_text, vocab_size, char_to_idx, idx_to_char = encode_text(input_file_path)

    seq_length = config['seq_len']
    X, y = create_sequences(encoded_text, seq_length)

    print('CREATED SEQUENCES')

    # Convert to PyTorch tensors. Sequences are stored as int16 (vocab is only
    # 65 characters) to keep RAM manageable at seq_len 512; batches are cast
    # back to long on the device inside train/eval before the embedding lookup.
    X_tensor = torch.tensor(X, dtype=torch.int16)
    y_tensor = torch.tensor(y, dtype=torch.long)

    len_data = len(y_tensor)

    train_frac, val_frac, test_frac = 0.8, 0.1, 0.1  
    train_size = int(train_frac * len_data)
    val_size = int(val_frac * len_data)
    test_size = len_data - train_size - val_size  

    torch.manual_seed(0)
    indices = torch.randperm(len_data)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    assert set(train_indices).isdisjoint(set(val_indices)) and set(train_indices).isdisjoint(set(test_indices)) and set(val_indices).isdisjoint(set(test_indices))
    print('PERFORMED TRAIN/VAL/TEST SPLIT')

    # Index tensors to get non-overlapping splits
    X_train, y_train = X_tensor[train_indices], y_tensor[train_indices]
    X_val, y_val = X_tensor[val_indices], y_tensor[val_indices]
    X_test, y_test = X_tensor[test_indices], y_tensor[test_indices]

    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    del X, y, X_tensor, y_tensor  # free the pre-split copies (matters for long seq_len)

    batch_size = config.get('batch_size', 64)
    train_dataset = ShakespeareDataset(X_train, y_train)
    # drop_last: a trailing batch of size 1 (which occurs at seq_len 512:
    # 891905 = 3484*256 + 1) hard-crashes the MPS backend's LSTM backward
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    val_dataset = ShakespeareDataset(X_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = ShakespeareDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")

    #choose model based on config
    if args.experiment == 'noteacherforcing':
        model = LSTMModelNoTeacherForcing(vocab_size, config['embed_size'],
                                config['hidden_size'],
                                config['num_layers']).to(device)
    elif config["model"] == "RNN":
        model = RNNModel(vocab_size, config['embed_size'],
                                config['hidden_size'],
                                config['num_layers']).to(device)
        
    elif config["model"] == "LSTM":
        model = LSTMModel(vocab_size, config['embed_size'],
                                config['hidden_size'],
                                config['num_layers']).to(device)

    #IMPORTANT FOFR AUTOGRADING: MAKE SURE THAT YOU MAKE YOUR train.py train METHOD SAVES YOUR MODEL AND LOADS IT LATER
    modelsLocation = "./models/"
    model_path = os.path.join(modelsLocation, f"trained_{config['name']}_model.pth")
    if not os.path.exists(model_path):## TRAIN - must save best model weights
        train(model=model,
                device=device,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                config=config) 

    else: # load model if exists 
        model.load_state_dict(torch.load(model_path, weights_only = False))
    ## INFERENCE
    test_loss = eval(model=model, device=device,
    val_dataloader=test_dataloader) 
    print(f"Test set loss: {test_loss:.4f}") #DO NOT CHANGE THIS PRINT STATEMENT - USED FOR AUTOGRADING


if __name__ == '__main__':
    main()