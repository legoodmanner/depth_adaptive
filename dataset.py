import torchaudio.transforms as T
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os
import librosa
from transformers import Wav2Vec2FeatureExtractor
import pickle

class AudioDataset(Dataset):
    def __init__(self, fns=None, labels=None):
        super().__init__()
        # fns: A loadable paths of the audio file
        self.seqproc =  Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M",trust_remote_code=True)# wav2vec preproc
        self.fns = fns
        self.labels = labels
        assert len(self.fns) == len(self.labels)
        # breakpoint()
    def __getitem__(self, idx):
        wav = librosa.load(self.fns[idx])[0]
        # inputs = self.resampler(torch.from_numpy(wav))
        inputs = librosa.resample(wav, orig_sr=22050, target_sr=24000)[:24000*28]
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
        
        
class GTZAN():
    def __init__(self, batch_size, num_workers, root=None, **args):
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

        with open(f'configs/GTZAN_train.txt', 'r') as f:
            train_fns = f.readlines()
        train_labels = [ self.labelID[fn.split('/')[0]] for fn in train_fns]
        train_fns = [os.path.join(self.root, fn.split('/')[1].strip('\n')) for fn in train_fns]
        self.train_loader = DataLoader(
            AudioDataset(train_fns, labels=train_labels,),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        with open(f'configs/GTZAN_valid.txt', 'r') as f:
            valid_fns = f.readlines()
        valid_labels = [ self.labelID[fn.split('/')[0]] for fn in valid_fns]
        valid_fns = [os.path.join(self.root, fn.split('/')[1].strip('\n')) for fn in valid_fns]
        self.valid_loader = DataLoader(
            AudioDataset(valid_fns, labels=valid_labels,),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        with open(f'configs/GTZAN_test.txt', 'r') as f:
            test_fns = f.readlines()
        test_labels = [ self.labelID[fn.split('/')[0]] for fn in test_fns]
        test_fns = [os.path.join(self.root, fn.split('/')[1].strip('\n')) for fn in test_fns]
        self.test_loader = DataLoader(
            AudioDataset(test_fns, labels=test_labels,),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        
    
class GTZAN_MERT():
    def __init__(self, batch_size, num_workers, root=None, **args):
        # Define: train_loader, valid_loader, test_loader, sampling_rate, processors 

        super().__init__()
        self.root = root or '../data/MERT_extracted'
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
        
        