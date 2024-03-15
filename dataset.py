import torchaudio.transforms as T
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os
import librosa
from transformers import Wav2Vec2FeatureExtractor
import pickle
import random

class AudioDataset(Dataset):
    def __init__(self, fns=None, labels=None, seg_len=None):
        super().__init__()
        # fns: A loadable paths of the audio file
        self.seqproc =  Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M",trust_remote_code=True)# wav2vec preproc
        self.fns = fns
        self.labels = labels
        self.seg_len = seg_len
        assert len(self.fns) == len(self.labels)
        # breakpoint()
    def __getitem__(self, idx):
        wav = librosa.load(self.fns[idx])[0]
        # inputs = self.resampler(torch.from_numpy(wav))
        inputs = librosa.resample(wav, orig_sr=22050, target_sr=24000)[:24000*28]
        if self.seg_len is not None:
            i = random.randint(0, 24000*(28-self.seg_len))
            inputs = inputs[i:i+self.seg_len*24000]
        inputs = self.seqproc(inputs, sampling_rate=24000, return_tensors='np')
        return inputs['input_values'], inputs['attention_mask'], self.labels[idx], self.fns[idx]
        
    def __len__(self):
        return len(self.fns)
    

class FromPKLDataset(Dataset):
    def __init__(self, fns=None):
        super().__init__()
        self.fns = fns
    def __getitem__(self, index):
        with open(self.fns[index], 'rb') as f:
            pkl = pickle.load(f)
        return pkl
    def __len__(self):
        return len(self.fns)
        

# DataSet Module
# ====================
       
class GTZAN():
    def __init__(self, batch_size, num_workers, root=None, seg_len=None, **args):
        # Define: train_loader, valid_loader, test_loader, sampling_rate, processors 

        super().__init__()
        self.root = root or '../data/GTZAN/'
        self.sampling_rate = 22050
        self.labelID = {
            'blues': 0,
            'classical': 1,
            'country': 2,
            'disco': 3,
            'hiphop': 4,
            'jazz': 5,
            'metal': 6,
            'pop': 7,
            'reggae': 8,
            'rock': 9
        }
        # self.processors = [
        #     T.Resample(self.sampling_rate, 24000), # resampler 
        #     Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M",trust_remote_code=True), # wav2vec preproc
        # ]
        random.seed(4900)
        with open(f'configs/GTZAN_train.txt', 'r') as f:
            train_fns = f.readlines()
        train_labels = [ self.labelID[fn.split('/')[0]] for fn in train_fns]
        train_fns = [os.path.join(self.root, fn.split('/')[1].strip('\n')) for fn in train_fns]
        self.train_loader = DataLoader(
            AudioDataset(train_fns, labels=train_labels, seg_len=seg_len),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        with open(f'configs/GTZAN_valid.txt', 'r') as f:
            valid_fns = f.readlines()
        valid_labels = [ self.labelID[fn.split('/')[0]] for fn in valid_fns]
        valid_fns = [os.path.join(self.root, fn.split('/')[1].strip('\n')) for fn in valid_fns]
        self.valid_loader = DataLoader(
            AudioDataset(valid_fns, labels=valid_labels, seg_len=seg_len),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        with open(f'configs/GTZAN_test.txt', 'r') as f:
            test_fns = f.readlines()
        test_labels = [ self.labelID[fn.split('/')[0]] for fn in test_fns]
        test_fns = [os.path.join(self.root, fn.split('/')[1].strip('\n')) for fn in test_fns]
        self.test_loader = DataLoader(
            AudioDataset(test_fns, labels=test_labels,  seg_len=seg_len),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        
    
class GTZAN_MERT():
    # Dataset of GTZAN that being extracted by MERT
    def __init__(self, batch_size, num_workers, adaption=None, adpt_layers=None, adpt_confs=None, root=None, **args):
        # Define: train_loader, valid_loader, test_loader, sampling_rate, processors 
        
        super().__init__()
        self.root = root or '../data/MERT_repr/'
        if adaption is None:
            self.root = os.path.join(self.root, 'original')
        elif adaption and adpt_layers:
            if isinstance(adpt_layers, int): adpt_layers = [adpt_layers]
            sorted(adpt_layers)
            self.root = os.path.join(self.root, f'{adaption}_{"_".join([str(l) for l in adpt_layers])}_k={adpt_confs["k"]}')
            # if the dir not exisit -> the features haven't been extracted
            if not os.path.isdir(self.root):
                print('Extracting features....')
                os.makedirs(self.root, exist_ok=False)
                feature_extraction(self.root, adaption, adpt_layers, adpt_confs)

        self.sampling_rate = 22050
        with open(f'configs/GTZAN_train.txt', 'r') as f:
            train_fns = f.readlines()
        train_fns = [os.path.join(self.root, fn.split('/')[1].strip('.wav\n')+'.pkl') for fn in train_fns]
        self.train_loader = DataLoader(
            FromPKLDataset(train_fns),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        with open(f'configs/GTZAN_valid.txt', 'r') as f:
            valid_fns = f.readlines()
        valid_fns = [os.path.join(self.root, fn.split('/')[1].strip('.wav\n')+'.pkl') for fn in  valid_fns]
        self.valid_loader = DataLoader(
            FromPKLDataset(valid_fns),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        with open(f'configs/GTZAN_test.txt', 'r') as f:
            test_fns = f.readlines()
        test_fns = [os.path.join(self.root, fn.split('/')[1].strip('.wav\n')+'.pkl') for fn in  test_fns]
        self.test_loader = DataLoader(
            FromPKLDataset(test_fns,),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        


def feature_extraction(root, adaption, adpt_layers, adpt_confs):
    import importlib
    from transformers import AutoModel
    from model import parma_edit
    from omegaconf import OmegaConf
    from tqdm import tqdm
    def load_dataset(configs):
        # call dataset, build the set by config
        dataset = getattr(importlib.import_module('dataset'), f'{configs.name}')(**configs)
        return dataset

    device = torch.device('cuda')
    # Model should be selected through var, need update
    model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
    if adaption and adpt_layers:
        parma_edit(model, adaption, adpt_layers, adpt_confs)
    model = model.to(device)
    processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M" ,trust_remote_code=True)
    configs = OmegaConf.create(
        {
            'name': 'GTZAN',
            'batch_size': 1,
            'num_workers': 0,
        }
    )

    dataset = load_dataset(configs)
    train_loader, valid_loader, test_loader = dataset.train_loader, dataset.valid_loader, dataset.test_loader
    # make sure the sample_rate aligned
    for loader in [train_loader, valid_loader, test_loader]:
        for batch in tqdm(loader):
            input_values, attn_mask, label, fns = batch
            input_values, attn_mask, label = input_values.to(device), attn_mask.to(device), label.to(device)
            with torch.no_grad():
                input_values = input_values.squeeze(1)
                outputs = model(input_values= input_values, attention_mask=attn_mask, output_hidden_states=True)
                outputs['last_hidden_state'] = outputs['last_hidden_state'].mean(-2).squeeze().detach().cpu().numpy()
                outputs['hidden_states'] = np.array([h.mean(-2).detach().cpu().numpy() for h in  outputs['hidden_states']])
                outputs['filename'] = fns[0].split('/')[-1]
                outputs['label'] = label.detach().cpu().numpy()
                
                with open(f"{root}/{outputs['filename'].strip('.wav')}.pkl", 'wb') as f:
                    pickle.dump(dict(outputs), f)
    
    del model
    torch.cuda.empty_cache()