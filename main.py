
from transformers import AutoModel
import torch
from torch import nn
from omegaconf import OmegaConf
import importlib
from model import Downstream_MLP
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
dataset = load_dataset(configs.data)
train_loader, valid_loader, test_loader = dataset.train_loader, dataset.valid_loader, dataset.test_loader

# run = wandb.init(
#     project="depth",
#     dir='logs',
#     # Track hyperparameters and run metadata
# )
device = torch.device('cuda')
# audio file is decoded on the fly
lossfn = nn.CrossEntropyLoss()

# model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True).to(device)


train_valid_n = 30
min_val = 1000

def train_entire():
    for n_epoch in range(1000):
        for batch in tqdm(train_loader):
            input_values, attn_mask, label, fn = batch
            input_values, attn_mask, label = input_values.to(device), attn_mask.to(device), label.to(device)
            with torch.no_grad():
                # batch['inputs']['input_values'] = batch['inputs']['input_values'].squeeze(1)
                # outputs = model(**batch['inputs'], output_hidden_states=True)
                input_values = input_values.squeeze(1)
                outputs = model(input_values= input_values, attention_mask=attn_mask, output_hidden_states=True)
            pred_y = mlp(outputs.hidden_states[-1].mean(1))
            loss = lossfn(pred_y, label)
            loss.backward()
            optimizer.step()
            wandb.log({'train_loss': loss})
        print(f'Epoch {n_epoch} | train_loss: {loss}')


        if n_epoch % train_valid_n == 0:
            for batch in tqdm(valid_loader):
                input_values, attn_mask, label, fn = batch
                input_values, attn_mask, label = input_values.to(device), attn_mask.to(device), label.to(device)
                # label = batch['label']
                with torch.no_grad():
                    # batch['inputs']['input_values'] = batch['inputs']['input_values'].squeeze(1)
                    # outputs = model(**batch['inputs'], output_hidden_states=True)
                    input_values = input_values.squeeze(1)
                    outputs = model(input_values= input_values, attention_mask=attn_mask, output_hidden_states=True)
                    pred_y = mlp(outputs.hidden_states[-1].mean(1))
                    loss = lossfn(pred_y, label)
            if min_val > loss:
                min_val = loss
                torch.save(mlp.state_dict(), 'param')
            scheduler.step(loss)
            print(f'Epoch {n_epoch} | valid_loss: {loss}')
            wandb.log({'valid_loss': loss})

def train_pretrained(layer):
    runs = wandb.init(
        project="depth",
        dir='logs',
        name=f'{layer}layers_1e-4'
        # Track hyperparameters and run metadata
    )
    mlp = Downstream_MLP(n_class=10).to(device)
    optimizer = Adam(params=mlp.parameters(), lr=configs.model.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.3, patience=80)    
    min_val = 1000
    for n_epoch in range(1000):
        tloss = 0
        mlp.train()
        mlp.zero_grad()
        for idx, batch in enumerate(tqdm(train_loader)):
            h, label, = batch['hidden_states'], batch['label']
            pred_y = mlp(h[:,layer].squeeze(1).to(device))
            loss = lossfn(pred_y, label.squeeze(1).to(device))
            loss.backward()
            tloss += loss.item()
            optimizer.step()
            wandb.log({'train_loss': loss})
        print(f'Epoch {n_epoch} | train_loss: {tloss/idx}')


        if n_epoch % train_valid_n == 0:
            vloss = 0
            mlp.eval()
            for idx, batch in enumerate(tqdm(valid_loader)):
                h, label, = batch['hidden_states'], batch['label']
                pred_y = mlp(h[:,layer].squeeze(1).to(device))
                loss = lossfn(pred_y, label.squeeze(1).to(device))
                vloss += loss.item()
                wandb.log({'valid_loss': loss})
            if min_val > loss:
                min_val = loss
                torch.save(mlp.state_dict(), f'param_l{layer}')
            scheduler.step(loss)
            print(f'Epoch {n_epoch} | valid_loss: {vloss/idx}')
    wandb.finish()

train_pretrained(int(sys.argv[1]))

        
    



