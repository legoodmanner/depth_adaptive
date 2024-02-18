
import torch
from torch import nn
from omegaconf import OmegaConf
import importlib
from model import Downstream_MLP
from tqdm import tqdm
import wandb
import sys
# from datasets import load_dataset

def load_dataset(configs):
    # call dataset, build the set by config
    dataset = getattr(importlib.import_module('dataset'), f'{configs.name}')(**configs)
    return dataset
    
configs = OmegaConf.load('configs/MERT.yaml')
# loading our model weights
# loading the corresponding preprocessor config

# load demo audio and set processor
configs.data.batch_size = 250
dataset = load_dataset(configs.data)
train_loader, valid_loader, test_loader = dataset.train_loader, dataset.valid_loader, dataset.test_loader

# run = wandb.init(
#     project="depth",
#     dir='logs',
#     # Track hyperparameters and run metadata
# )
device = torch.device('cuda')
layer = int(sys.argv[1])
states = torch.load(f'param_l{layer}')
mlp = Downstream_MLP(n_class=10).to(device)
mlp.load_state_dict(states)
mlp.eval()
for idx, batch in enumerate(tqdm(test_loader)):
    h, label, = batch['hidden_states'], batch['label']
    pred_y = mlp(h[:,layer].squeeze(1).to(device))
    acc = (torch.argmax(pred_y, -1) == label.squeeze(1).to(device)).sum() / pred_y.size(0)
    print(f'layer: {layer} acc: {acc}')
    print(pred_y.size())

