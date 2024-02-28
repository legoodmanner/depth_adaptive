
from transformers import AutoModel
import torch
from torch import nn
from omegaconf import OmegaConf
import importlib
from model import Downstream_MLP, parma_edit
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import wandb, argparse
import os, datetime

# from datasets import load_dataset

def train_entire(conf):
    # model init
    model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
    parma_edit(model, mode=conf.mode, reduction_layer=conf.reduction_layer,)
    model = model.to(device)
    mlp = Downstream_MLP(n_class=10).to(device)
    # optimizer
    optimizer_mlp = Adam(params=mlp.parameters(), lr=conf.mlp_lr)
    scheduler_mlp = ReduceLROnPlateau(optimizer_mlp, 'min', factor=0.3, patience=80)
    if conf.mode == 'lora':
        optimizer_reduction = Adam(params=model.parameters(), lr=conf.reduction_lr)
        scheduler_reduction = ReduceLROnPlateau(optimizer_reduction, 'min', factor=0.3, patience=80) 
    min_val = 1000

    for n_epoch in range(1000):
        model.train()
        mlp.train()
        for batch in tqdm(train_loader):
            input_values, attn_mask, label, fn = batch
            input_values, attn_mask, label = input_values.to(device), attn_mask.to(device), label.to(device)
            input_values = input_values.squeeze(1)
            outputs = model(input_values= input_values, attention_mask=attn_mask, output_hidden_states=True)
            pred_y = mlp(outputs.hidden_states[conf.hidden_layer].mean(1))
            loss = lossfn(pred_y, label)
            loss.backward()
            optimizer_mlp.step()
            if conf.mode == 'lora':
                optimizer_reduction.step()
            wandb.log({'train_loss': loss})
        print(f'Epoch {n_epoch} | train_loss: {loss}')


        if n_epoch % conf.train_valid_n == 0:
            model.eval()
            mlp.eval()
            for batch in tqdm(valid_loader):
                input_values, attn_mask, label, fn = batch
                input_values, attn_mask, label = input_values.to(device), attn_mask.to(device), label.to(device)
                with torch.no_grad():
                    input_values = input_values.squeeze(1)
                    outputs = model(input_values= input_values, attention_mask=attn_mask, output_hidden_states=True)
                    pred_y = mlp(outputs.hidden_states[conf.hidden_layer].mean(1))
                    loss = lossfn(pred_y, label)
            if min_val > loss:
                min_val = loss
                torch.save(mlp.state_dict(),)
            scheduler_mlp.step(loss)
            if conf.mode == 'lora':
                scheduler_reduction.step(loss)
            print(f'Epoch {n_epoch} | valid_loss: {loss}')
            wandb.log({'valid_loss': loss})

def train_pretrained(layer):
    
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


        if n_epoch % conf.train_valid_n == 0:
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', '-c', type=str,)
    args = parser.parse_args()

    def load_dataset(dconf):
        dataset = getattr(importlib.import_module('dataset'), f'{dconf.name}')(**dconf)
        return dataset
    
    # Load config (public and private)
    configs = OmegaConf.merge(
        OmegaConf.load('configs/private_conf.yaml'), 
        OmegaConf.load(f'configs/{args.conf.strip(".yaml")}.yaml')
    )
    
    # Build dataset from configs    
    dataset = load_dataset(configs.data)
    train_loader, valid_loader, test_loader = dataset.train_loader, dataset.valid_loader, dataset.test_loader

    # build logs directory
    name = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    logpath = f'logs/{name}'
    os.makedirs(logpath, exist_ok=True)
    for child in ['params', 'wandb', 'configs']:
        os.makedirs(f'{logpath}/{child}', exist_ok=True)
    configs.name = name
    OmegaConf.save(configs, f'{logpath}/configs/configs.yaml')

    # init logger of wandb
    runs = wandb.init( 
        project= configs.logger.project,
        dir=f'{logpath}/wandb',
        config= dict(configs),
        name=name,
    )
 
    device = torch.device('cuda')
    lossfn = nn.CrossEntropyLoss()

    train_entire(configs.model)

        
    



