import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer,CascadingHierarchicalAttention
from layers.Embed import DataEmbedding_inverted
import numpy as np

import torch
from torch import nn

class CouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, mask):
        super(CouplingLayer, self).__init__()
        self.mask = mask

        self.scale_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh(), 
        )
        self.translation_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        
    def forward(self, x):
        masked_x = self.mask * x
        scale = self.scale_net(masked_x) * (1 - self.mask)
        translation = self.translation_net(masked_x) * (1 - self.mask)
        
        z = masked_x + (1 - self.mask) * (x * torch.exp(scale) + translation)
        return z

# Construct Real-NVP 
class RealNVP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_flows,configs):
        super(RealNVP, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_flows):
            # Construct CouplingLayer
            mask = torch.arange(input_dim) % 2
            if input_dim % 2 == 1:
                mask[:input_dim // 2 + 1] = 1
                mask[input_dim // 2 + 1:] = 0
            mask = mask.float().to('cuda:0')
            coupling_layer = CouplingLayer(input_dim, hidden_dim, mask)
            self.layers.append(coupling_layer)

            # inversr mask after CouplingLayer
            mask = 1 - mask
            coupling_layer = CouplingLayer(input_dim, hidden_dim, mask)
            self.layers.append(coupling_layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        CascadingHierarchicalAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention,max_level=configs.max_level), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.projector_dis = nn.Linear(configs.d_model, 2 * configs.pred_len, bias=True)
        self.real_nvp = RealNVP(configs.enc_in * configs.pred_len, configs.d_model, 2,configs)


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, levels):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of indictors, can also includes covariates
        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
         # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None,levels=levels)

        # B N E -> B N S -> B S N 
        #dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        #B N 2E -> B N E -> B S N
        mean=self.projector_dis(enc_out)[:,:N,:self.pred_len].permute(0, 2, 1)
        std = nn.functional.softplus(self.projector_dis(enc_out)[:, :N, self.pred_len:]).permute(0, 2, 1) 
        #print(mean.shape)
        num_samples = 100
        base_distribution=D.Normal(mean,std)
        samples = [base_distribution.rsample() for _ in range(num_samples)]
        samples_stack = torch.stack(samples)  # (num_samples, B, S, N)
        z = samples_stack.mean(dim=0)  # (B, S, N)
        z = z.view(z.size(0), -1)
        z = self.real_nvp(z)
        zk = z.view(z.size(0), -1, N)
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            zk = zk * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            zk = zk + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            expanded_means = means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)  # B S N
            expanded_stdev = stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)  # B S N
            mean = mean * expanded_stdev + expanded_means  # B S N
            std = std * expanded_stdev  # B S N
        return zk,mean,std


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, levels=None):
        z0,mean,std = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, levels)
        return z0,mean,std
    
    def real_nvp_transform(self, samples):
        B,S,N=samples.size()
        samples_flat = samples.view(B, -1)
        transformed_samples = self.real_nvp(samples_flat)
        transformed_samples = transformed_samples.view(B, -1, N)
        return transformed_samples
