import torch.nn as nn
from einops import rearrange

from .helpers import get_activation_fn
from .tno import Tno
from .helpers import get_norm_fn
import numpy as np


class Gtu2d(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.0,
        bias=True,
        act_fun="silu",
        causal=False,
        expand_ratio=3,
        use_norm=False,
        norm_type="layernorm",
        use_decay=False,
        use_multi_decay=False,
        rpe_layers=3,
        rpe_embedding=512,
        rpe_act="relu",
        normalize=False,
        par_type=1,
        residual=False,
        gamma=0.99,
        act_type="none",
    ):
        super().__init__()
        self.d_output = d_model
        self.p = dropout
        if self.p > 0:
            self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.expand_ratio = expand_ratio
        self.n_heads = n_heads
        self.normalize = normalize
        d1 = int(self.expand_ratio * d_model)
        d1 = (d1 // self.n_heads) * self.n_heads
        d2 = d_model
        self.head_dim = d1 // n_heads
        # linear projection
        self.v_proj = nn.Linear(d_model, d1, bias=bias)
        self.u_proj = nn.Linear(d_model, d1, bias=bias)
        self.o = nn.Linear(d1, d_model, bias=bias)
        self.act = get_activation_fn(act_fun)
        # tno
        self.toep1 = Tno(
            h=n_heads, 
            dim=self.head_dim,
            rpe_dim=rpe_embedding, 
            causal=causal, 
            use_decay=use_decay, 
            use_multi_decay=use_multi_decay,
            residual=residual,
            act=rpe_act,
            par_type=par_type,
            gamma=gamma,
            bias=bias,
            act_type=act_type,
            layers=rpe_layers,
            norm_type=norm_type,
        )
        self.toep2 = Tno(
            h=n_heads, 
            dim=self.head_dim,
            rpe_dim=rpe_embedding, 
            causal=causal, 
            use_decay=use_decay, 
            use_multi_decay=use_multi_decay,
            residual=residual,
            act=rpe_act,
            par_type=par_type,
            gamma=gamma,
            bias=bias,
            act_type=act_type,
            layers=rpe_layers,
            norm_type=norm_type,
        )
        # norm
        self.norm_type = norm_type
        self.use_norm = use_norm
        
        self.par_init()
        
    def par_init(self):
        nn.init.normal_(self.u_proj.weight, std=0.02)
        nn.init.normal_(self.u_proj.bias, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.bias, std=0.02)
        nn.init.normal_(self.o.weight, std=0.02)
        nn.init.normal_(self.o.bias, std=0.02)
    
    def forward(self, x, state=None):
        # x: b, h * w, d
        n = x.shape[1]
        H = int(np.sqrt(n))
        W = n //  H
        n_heads = self.n_heads
        
        u = self.act(self.u_proj(x))
        v = self.act(self.v_proj(x))
        # reshape
        v = rearrange(v, 'b (H W) (h d) -> b h H W d', h=n_heads, H=H, W=W)
        o1 = self.toep1(v, dim=-2, normalize=self.normalize)
        o1 = self.toep2(o1, dim=-3, normalize=self.normalize)
        o2 = self.toep2(v, dim=-3, normalize=self.normalize)
        o2 = self.toep1(o2, dim=-2, normalize=self.normalize)
        output = o1 + o2
        output = rearrange(output, 'b h H W d -> b (H W) (h d)')
        # dropout
        if self.p > 0:
            output = self.dropout(output)
        output = u * output
        output = self.o(output)
        
        return output, None

