import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
# from reformer_pytorch import LSHSelfAttention
from einops import rearrange


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class CascadingHierarchicalAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False,max_level=None):
        super(CascadingHierarchicalAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.attention_layers = nn.ModuleList([
            FullAttention(mask_flag=False, scale=self.scale, attention_dropout=attention_dropout, output_attention=False) 
            for _ in range(max_level)
        ])

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, levels=None):
        B, L, H, E = queries.shape
        previous_output = torch.zeros_like(queries)  
        combined_output = torch.zeros_like(queries) 
        if levels == None:
            levels=np.ones(L)
        
        for level_index in range(1, (max(levels) + 1)):
            level_mask = torch.logical_or(levels == level_index, levels == -1)
            level_mask = level_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(B, L, H, E)
            
            current_level_q = queries * level_mask.float()
            current_level_k = keys * level_mask.float()
            current_level_v = values * level_mask.float()
            
            if level_index > 1: 
                current_level_q = previous_output+ current_level_q
                current_level_k = previous_output+ current_level_k
                current_level_v = previous_output+ current_level_v
               
            attention_module = self.attention_layers[level_index - 1]
            level_output, _ = attention_module(current_level_q, current_level_k, current_level_v, None)
            
            combined_output += level_output * level_mask.float()
            previous_output = level_output * level_mask.float() 
            
        return combined_output, None

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None,levels=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta,
            levels=levels
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


