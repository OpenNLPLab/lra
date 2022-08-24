import math
import numpy as np
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter
from torch.nn import Dropout
import sys
from .dpb_v4 import SimpleRMSNorm
from .dynamic_toeplitz_encoding_multihead_v4 import DynamicToepliztMultiheadV4
from einops import rearrange

class TNO(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.0,
        bias=True,
        # add
        index=0,
        act_fun="silu",
        causal=False,
        expand_ratio=2,
        shrink_ratio=1,
        resi_param=False,
        # norm
        use_norm=False,
        norm_type="simplermsnorm",
        # Toeplizt
        use_exp=False,
        use_neg_exp=False, 
        tno_max_l=512,
        use_decay=False,
        use_multi_decay=False,
        dpb_embedding=512,
        dpb_act="relu",
        dpb_use_pad=True,
        normalize=False,
        par_type=1,
        dpb_type=1,
        dpb_layers=3,
        residual=False,
        l=1, 
        transform_type=1,
        gamma=0.999,
        # token shift
        token_shift_type=-1,
        # tno
        tno_type=1,
    ):
        # add
        self.index = index

        super().__init__()
        self.d_output = d_model
        self.embed_dim = d_model
        self.num_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.expand_ratio = expand_ratio
        self.resi_param = resi_param
        print(f"self.expand_ratio {self.expand_ratio}")
        print(f"self.resi_param {self.resi_param}")
        if self.resi_param:
            self.d = nn.Parameter(torch.randn(self.embed_dim))
            
        d1 = int(self.expand_ratio * d_model)
        d1 = (d1 // self.num_heads) * self.num_heads
        d2 = d_model
        self.head_dim = d1 // n_heads
        # d^2
        self.v_proj = nn.Linear(d_model, d1, bias=bias)
        # d^2
        self.u_proj = nn.Linear(d_model, d1, bias=bias)
        # d^2
        self.o = nn.Linear(d1, d_model, bias=bias)

        self.causal = causal
        self.act = self.get_act_fun(act_fun)
        print(f"act_fun {act_fun}")
        print(f"causal {self.causal}")
        
        # toep
        self.max_l = tno_max_l
        self.use_exp = use_exp
        self.use_neg_exp = use_neg_exp
        self.use_decay = use_decay
        self.use_multi_decay = use_multi_decay
        self.dpb_embedding = dpb_embedding
        self.dpb_act = dpb_act
        self.dpb_use_pad = dpb_use_pad
        self.normalize = normalize
        self.par_type = par_type
        self.dpb_type = dpb_type
        self.residual = residual
        self.l = l
        self.transform_type = transform_type
        self.gamma = gamma
        self.bias = bias
        self.tno_type = tno_type
        self.dpb_layers = dpb_layers
        self.toep = DynamicToepliztMultiheadV4(
            h=self.num_heads, 
            n=self.max_l, 
            dim=self.head_dim,
            dpb_dim=self.dpb_embedding, 
            causal=self.causal, 
            use_exp=self.use_exp,
            use_neg_exp=self.use_neg_exp,
            use_decay=self.use_decay, 
            use_multi_decay=self.use_multi_decay,
            use_pad=self.dpb_use_pad,
            act=self.dpb_act,
            par_type=self.par_type,
            residual=self.residual,
            dpb_type=self.dpb_type,
            layers=self.dpb_layers,
            l=self.l,
            transform_type=self.transform_type,
            gamma=self.gamma,
            bias=self.bias,
        )
        print(f"self.num_heads {self.num_heads}")
        print(f"self.max_l {self.max_l}")
        print(f"self.use_exp {self.use_exp}")
        print(f"self.use_neg_exp {self.use_neg_exp}")
        print(f"self.use_decay {self.use_decay}")
        print(f"self.use_multi_decay {self.use_multi_decay}")
        print(f"self.dpb_embedding {self.dpb_embedding}")
        print(f"self.dpb_act {self.dpb_act}")
        print(f"self.dpb_use_pad {self.dpb_use_pad}")
        print(f"self.normalize {self.normalize}")
        print(f"self.par_type {self.par_type}")
        print(f"self.dpb_type {self.dpb_type}")
        print(f"self.residual {self.residual}")
        print(f"self.l {self.l}")
        print(f"self.transform_type {self.transform_type}")
        print(f"self.gamma {self.gamma}")
        print(f"bias {bias}")
        print(f"tno_type {tno_type}")
        print(f"dpb_layers {dpb_layers}")
        
        
        # norm
        self.norm_type = norm_type
        self.pre_norm = self.get_norm_fun(self.norm_type, d2)
        
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = self.get_norm_fun(norm_type, d1)
        print(f"use_norm {self.use_norm}")
        print(f"norm_type {self.norm_type}")
        
        self.token_shift_type = token_shift_type
        print(f"self.token_shift_type {self.token_shift_type}")
        if self.token_shift_type == 1:
            self.token_shift = nn.ZeroPad2d((0, 0, 1, -1))
        elif self.token_shift_type == 2:
            self.token_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.coef = 0.5

        self.par_init()
        
    def par_init(self):
        nn.init.normal_(self.u_proj.weight, std=0.02)
        nn.init.normal_(self.u_proj.bias, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.bias, std=0.02)
        nn.init.normal_(self.o.weight, std=0.02)
        nn.init.normal_(self.o.bias, std=0.02)

    def get_norm_fun(self, norm_type, d_model):
        if norm_type == "rmsnorm":
            print("here! rmsnorm")
            return RMSNorm(d_model)
        elif norm_type == "gatedrmsnorm":
            print("here! gatedrmsnorm")
            return GatedRMSNorm(d_model)
        elif norm_type == "simplermsnorm":
            print("here! simple rmsnorm")
            return SimpleRMSNorm(d_model)
        elif norm_type == "scalenorm":
            print("here! scale norm")
            return ScaleNorm(d_model)
        else:
            print("here! layer norm")
            return nn.LayerNorm(d_model)

    def get_act_fun(self, act_fun):
        print(act_fun)
        if act_fun == "gelu":
            return F.gelu
        elif act_fun == "relu":
            return F.relu
        elif act_fun == "elu":
            return F.elu
        elif act_fun == "sigmoid":
            return torch.sigmoid
        elif act_fun == "exp":
            return torch.exp
        elif act_fun == "leak":
            def f(x):
                return F.leaky_relu(x, negative_slope=self.negative_slope)
            return f
        elif act_fun == "1+elu":
            def f(x):
                return 1 + F.elu(x)
            return f
        elif act_fun == "silu":
            return F.silu
        elif self.act_fun == "relu2":
            def f(x):
                return torch.square(torch.relu(x))
            return f
        else:
            return lambda x: x

    # 1D
    def forward(self, x, state = None):
        # x: b, h * w, d
        num_heads = self.num_heads
        
        if self.token_shift_type == 1:
            x = self.token_shift(x)
        elif self.token_shift_type == 2:
            q1 = self.token_shift(x)
            x = self.coef * q1 + (1 - self.coef) * x

        shortcut, x = x, self.pre_norm(x)
        if self.resi_param:
            shortcut = shortcut * self.d
        u = self.act(self.u_proj(x))
        v = self.act(self.v_proj(x))
        # reshape
        v = rearrange(v, 'b n (h d) -> b h n d', h=num_heads)
        output = self.toep(v, dim=-2, normalize=self.normalize)
        output = rearrange(output, 'b h n d -> b n (h d)')
        output = u * output
        if self.use_norm:
            output = self.norm(output)
            
        output = self.o(output) + shortcut
        
        return output, None

