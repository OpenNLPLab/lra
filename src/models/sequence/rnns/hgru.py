import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

from hgru import BiHgru1d, Hgru1d, BiHgru2d, HgruReal2d, Hgru2d

class Hgru1dModule(nn.Module):
    def __init__(
        self, 
        embed_dim, 
        act_fun="silu", 
        causal=False,
        # add
        dropout=0.0,
        transposed=False,
        param_share=True,
        use_triton=False,
        use_lower_bound=True,
        use_real=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        if param_share:
            self.hgru = Hgru1d(embed_dim=embed_dim, act_fun=act_fun, causal=causal)
        else:
            self.hgru = BiHgru1d(embed_dim=embed_dim, act_fun=act_fun)
        self.d_output = embed_dim
        self.use_lower_bound = use_lower_bound
        
    def forward(self, x, state=None, **kwargs):
        if self.use_lower_bound:
            lower_bound = kwargs["lower_bound"]
        else:
            lower_bound = 0
        x = rearrange(x, 'b n d -> n b d')
        output = self.hgru(x, lower_bound)
        output = rearrange(output, 'n b d -> b n d')
        
        return output, None
    
    # # only for speed test!!!
    # def step(self, x, state=None, **kwargs):
    #     """ Step one time step as a recurrent model. Intended to be used during validation.

    #     x: (b, d)
    #     state: [hidden_real, hidden_imag], (b, d)
    #     Returns: output (b, d), state (b, d)
    #     """
    #     assert not self.training
    #     lower_bound = 0
    #     y, next_state = self.hgru.inference(x, state, lower_bound) # (B C H)
        
    #     return y, next_state
    
    # def default_state(self, *batch_shape, device=None):
    #     size = batch_shape + (self.embed_dim,)
    #     return [torch.zeros(size).to(device), torch.zeros(size).to(device)]
    
class Hgru2dModule(nn.Module):
    def __init__(
        self, 
        embed_dim, 
        act_fun="silu", 
        causal=False,
        # add
        dropout=0.0,
        transposed=False,
        param_share=True,
        use_triton=False,
        use_lower_bound=True,
        use_real=False,
    ):
        super().__init__()
        if not use_real:
            if param_share:
                self.hgru = Hgru2d(embed_dim=embed_dim, act_fun=act_fun, causal=causal)
            else:
                self.hgru = BiHgru2d(embed_dim=embed_dim, act_fun=act_fun, causal=causal)
        else:
            self.hgru = HgruReal2d(embed_dim=embed_dim, act_fun=act_fun, causal=causal)
        self.d_output = embed_dim
        self.use_lower_bound = use_lower_bound
        
    def forward(self, x, state=None, **kwargs):
        if self.use_lower_bound:
            lower_bound = kwargs["lower_bound"]
        else:
            lower_bound = 0
        x = rearrange(x, 'b n d -> n b d')
        n = x.shape[0]
        h = int(np.sqrt(n))
        x = rearrange(x, '(h w) b d -> h w b d', h=h)
        output = self.hgru(x, lower_bound)
        output = rearrange(output, 'h w b d -> b (h w) d')
        
        return output, None