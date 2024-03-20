from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
from torch import nn
import torchaudio.transforms as T
import loralib as lora
from loralib.layers import Linear as LoraLinear
from copy import deepcopy

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):  # Check for specific layer types
            torch.nn.init.xavier_normal_(m.weight,)  # Set mean and std

class StemAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(StemAttention, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # Linear projections for Q, K, and V
        self.D_q = nn.Linear(d_model, d_model//8)
        self.D_k = nn.Linear(d_model, d_model//8)
        self.D_v = nn.Linear(d_model, d_model)

        self.O_q = nn.Linear(d_model, d_model//8)
        self.O_k = nn.Linear(d_model, d_model//8)
        self.O_v = nn.Linear(d_model, d_model)

        self.V_q = nn.Linear(d_model, d_model//8)
        self.V_k = nn.Linear(d_model, d_model//8)
        self.V_v = nn.Linear(d_model, d_model)

        self.B_q = nn.Linear(d_model, d_model//8)
        self.B_k = nn.Linear(d_model, d_model//8)
        self.B_v = nn.Linear(d_model, d_model)

    def forward(self, inp):
        # Project input for queries, keys, and values
        Dq = self.D_q(inp['drums'])
        Dk = self.D_k(inp['drums'])
        Dv = self.D_v(inp['drums'])

        Vq = self.V_q(inp['vocals'])
        Vk = self.V_k(inp['vocals'])
        Vv = self.V_v(inp['vocals'])

        Bq = self.V_q(inp['bass'])
        Bk = self.V_k(inp['bass'])
        Bv = self.V_v(inp['bass'])

        Oq = self.V_q(inp['other'])
        Ok = self.V_k(inp['other'])
        Ov = self.V_v(inp['other'])


        
        q = torch.cat((Dq, Vq, Bq, Oq), dim=-2)
        v = torch.cat((Dv, Vv, Bv, Ov), dim=-2)
        k = torch.cat((Dk, Vk, Bk, Ok), dim=-2)
        

        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.d_model**0.5  # Scale by sqrt(d_model)
        scores = self.dropout(scores)

        # Apply softmax to normalize attention weights
        attention = torch.softmax(scores, dim=-1)

        # Context vector as weighted sum of values
        output = torch.matmul(attention, v)

        return output.mean(-2)
  
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
    
class Downstream_4stemsMLP(Downstream_MLP):
    def __init__(self, n_class) -> None:
        super().__init__(n_class)

        self.attn = StemAttention(1024)
        initialize_weights(self.attn)
        initialize_weights(self.mlp)
        
    def forward(self, x):
        x = self.attn(x)
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

def parma_edit(model, adaption=None, adpt_layers=None, adpt_confs=None, **kwargs):
    # Trun off all the gradient computation by defualt
    if isinstance(adpt_layers, int): adpt_layers = [adpt_layers]
    for param in model.parameters():
        param.requires_grad = False

    if adaption == 'laser' and adpt_layers is not None:
        print('Do the Laser reduction....')
        for layer in adpt_layers :
            model.encoder.layers[layer].feed_forward.intermediate_dense.weight \
                = do_low_rank(model.encoder.layers[layer].feed_forward.intermediate_dense.weight, k=adpt_confs['k'])
            
    elif adaption == 'lora':
        print('Do the Lora reduction....')
        for i in range(len(model.encoder.layers)):
            orig_dense = model.encoder.layers[i].feed_forward.intermediate_dense
            loralayer = LoraLinear(
                orig_dense.in_features,
                orig_dense.out_features,
                r = 512,
            )
            loralayer.weight, loralayer.bias = deepcopy(orig_dense.weight), deepcopy(loralayer.bias)
            model.encoder.layers[i].feed_forward.intermediate_dense = loralayer
    elif adaption is None:
        pass

if __name__ == '__main__':
    from transformers import AutoModel
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True).to(device)
    parma_edit(model, mode='lora')

    
