from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
from torch import nn
import torchaudio.transforms as T

class Downstream_MLP(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1024,512),
            nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.Linear(512, n_class),
            nn.Dropout(p=0.25),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.mlp(x)
