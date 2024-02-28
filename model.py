from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
from torch import nn
import torchaudio.transforms as T
import loralib as lora
from loralib.layers import Linear as LoraLinear
from copy import deepcopy

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

def do_low_rank(weight, k, debug=False, niter=2):
    assert weight.ndim == 2
    max_rank = min(weight.shape[0], weight.shape[1])
    desired_rank = int(max_rank * k)
    if debug:
        print(f"Shape is {weight.shape} and shape is {weight.dtype} => desired rank {desired_rank}")
    results = torch.svd_lowrank(weight,
                                q=desired_rank,
                                niter=niter)
    weight_approx = results[0] @ torch.diag(results[1]) @ results[2].T
    if debug:
        print(f"New matrix has shape {weight_approx.shape}")
    assert weight_approx.shape[0] == weight.shape[0] and weight_approx.shape[1] == weight.shape[1]
    weight_approx = torch.nn.Parameter(weight_approx)

    return weight_approx

def parma_edit(model, mode, reduction_layer=20, **kwargs):
    # Trun off all the gradient computation by defualt
    for param in model.parameters():
        param.requires_grad = False

    if mode == 'laser':
        model.encoder.layers[reduction_layer].feed_forward.intermediate_dense.weight \
            = do_low_rank(model.encoder.layers[reduction_layer].feed_forward.intermediate_dense.wieght)
    elif mode == 'lora':
        orig_dense = model.encoder.layers[reduction_layer].feed_forward.intermediate_dense
        loralayer = LoraLinear(
            orig_dense.in_features,
            orig_dense.out_features,
            r = 512,
        )
        loralayer.weight, loralayer.bias = deepcopy(orig_dense.weight), deepcopy(loralayer.bias)
        model.encoder.layers[reduction_layer].feed_forward.intermediate_dense = loralayer
        

if __name__ == '__main__':
    from transformers import AutoModel
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True).to(device)
    parma_edit(model, mode='lora')

    
