#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange, repeat
from models.model_basic import Base_Model

class PatchTST_Trendformer(Base_Model):
   def __init__(self, data_dim, in_len, out_len, use_filter, win_size, merge_size = 2, hop_len = 3,
                filter = "stft", d_model = 512, d_ff = 1024, n_heads = 8, e_layers = 3,
                dropout = 0.0, keepratio = 0.75, use_gmm = True, n_components = 3, dataset = "ETTh1"):
      super().__init__(data_dim, in_len, out_len, use_filter, win_size, merge_size, hop_len, filter, d_model, d_ff, n_heads, e_layers,
                dropout, keepratio, use_gmm, n_components, dataset)
      
      base_predictor = PatchTST_Series_predictor(self.data_dim, self.in_len, self.out_len, self.win_size, self.use_filter, merge_size=self.merge_size, hop_len=self.hop_len,
                                       filter=self.filter, d_model=d_model, d_ff=d_ff, n_heads=n_heads, e_layers=e_layers, dropout=dropout, keepratio=self.keepratio, use_gmm=self.use_gmm, n_components=self.n_components, dataset = dataset)
      if self.use_filter:
         self.trend_model = base_predictor
      else:
         self.series_model = base_predictor

   def init_gmm(self, new_mean, new_variance):
      super().init_gmm_mean(new_mean)
      super().init_gmm_variance(new_variance)

class PatchTST_Series_predictor(nn.Module):
    def __init__(self, data_dim, in_len, out_len, win_size, use_filter, merge_size=2, hop_len=3,
                 filter="stft", d_model=512, d_ff=1024, n_heads=8, e_layers=3,
                 dropout=0.0, keepratio=0.75, use_gmm=True, n_components=3, dataset="ETTh1"):
         super().__init__()
         self.data_dim = data_dim
         self.in_len = in_len+(hop_len - in_len%hop_len)%hop_len
         self.out_len = out_len
         self.win_size = win_size
         self.use_filter = use_filter
         self.filter = filter
         self.e_layers = e_layers
         self.use_gmm = use_gmm
         self.n_components = n_components if self.use_gmm else 1
         self.patch_len = win_size if (not self.use_filter) and dataset!="ILI" else 1
         self.stride_len = hop_len if (not self.use_filter) and dataset!="ILI" else 1
         self.reinv = True

         # The padding operation to handle invisible sgemnet length
         if self.use_filter:
            self.in_seg_num, self.out_seg_num = self.filter.generate_trend_in_out_len(self.in_len, self.out_len)
         else:
            pad_len = (hop_len - self.in_len % hop_len) % hop_len
            self.in_seg_num = self.in_len + pad_len
            self.in_nodisclo_seg_num = self.in_len + pad_len
            self.out_seg_num = self.out_len

         self.model = PatchTST_backbone(c_in=self.data_dim*self.n_components, context_window=self.in_seg_num*self.n_components, target_window=self.out_seg_num,
                                       n_components=self.n_components, patch_len=self.patch_len*self.n_components, stride=self.stride_len*self.n_components,
                                       max_seq_len=1024, n_layers=e_layers, d_model=d_model,
                                       n_heads=n_heads, d_k=None, d_v=None, d_ff=d_ff, attn_dropout=0.,
                                       dropout=dropout, act="gelu", head_dropout=0)

    def forward(self, x_seq):
         if self.use_gmm:
            x_seq_embed = rearrange(x_seq, "batch var seq_len gmm->batch var (seq_len gmm)")
         else:
            x_seq_embed = x_seq[:]

         x_seq_embed = self.model(x_seq_embed)

         if self.use_gmm:        
            predict_y = torch.sigmoid(x_seq_embed)
            predict_y = rearrange(predict_y,"batch var gmm out_len->batch var out_len gmm",gmm=self.n_components)
            predict_y = torch.clamp(predict_y, min=1e-5, max=1-1e-5)
         else:
            predict_y = x_seq_embed.squeeze(-2)
            
         return predict_y

class PatchTST_backbone(nn.Module):
   def __init__(self, c_in: int, context_window: int, target_window: int, n_components: int, patch_len: int, stride: int,
                max_seq_len = 1024, n_layers: int = 3, d_model=128, n_heads=16, d_k = None, d_v = None,
                d_ff: int = 256, attn_dropout: float = 0., dropout: float = 0.,
                act: str = "gelu", head_dropout=0,):

      super().__init__()

      # Patching
      self.patch_len = patch_len
      self.stride = stride
      patch_num = int((context_window - patch_len) / stride + 1)
      self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
      patch_num += 1

      # Backbone
      self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                  n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                  attn_dropout=attn_dropout, dropout=dropout, act=act)

      # Head
      self.head_nf = d_model * patch_num
      self.n_vars = c_in

      self.head = Flatten_Head(self.n_vars, d_model, patch_num, n_components, target_window, head_dropout=head_dropout)

   def forward(self, z):  # z: [bs x nvars x seq_len]
      # patching
      z = self.padding_patch_layer(z)
      z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # z: [bs x nvars x patch_num x patch_len]
      z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x patch_len x patch_num]

      # model
      z = self.backbone(z)  # z: [bs x nvars x d_model x patch_num]
      z = self.head(z)  # z: [bs x nvars x target_window]

      return z


class Flatten_Head(nn.Module):
   def __init__(self, n_vars, d_model, patch_num, n_components, target_window, head_dropout=0):
      super().__init__()

      self.n_vars = n_vars
      self.linear1 = nn.Linear(d_model, n_components)
      self.gelu = nn.GELU()
      self.linear2 = nn.Linear(patch_num, target_window)
      self.dropout = nn.Dropout(head_dropout)

   def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
      x = x.permute((0,1,3,2))
      x = self.gelu(self.linear1(x))
      x = x.permute((0,1,3,2))
      x = self.linear2(x)
      x = self.dropout(x)
      return x

class TSTiEncoder(nn.Module):  # i means channel-independent
   def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                d_ff=256, attn_dropout=0., dropout=0., act="gelu",):
      super().__init__()

      self.patch_num = patch_num
      self.patch_len = patch_len

      # Input encoding
      q_len = patch_num
      self.W_P = nn.Linear(patch_len, d_model)  # Eq 1: projection of feature vectors onto a d-dim vector space
      self.seq_len = q_len

      # Positional encoding
      self.W_pos = positional_encoding(q_len, d_model)

      # Residual dropout
      self.dropout = nn.Dropout(dropout)

      # Encoder
      self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout,
                                activation=act,  n_layers=n_layers,)

   def forward(self, x):  # x: [bs x nvars x patch_len x patch_num]

      n_vars = x.shape[1]
      # Input encoding
      x = x.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_num x patch_len]
      x = self.W_P(x)  # x: [bs x nvars x patch_num x d_model]

      u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # u: [bs * nvars x patch_num x d_model]
      u = self.dropout(u + self.W_pos)  # u: [bs * nvars x patch_num x d_model]

      # Encoder
      z = self.encoder(u)  # z: [bs * nvars x patch_num x d_model]
      z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))  # z: [bs x nvars x patch_num x d_model]
      z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x d_model x patch_num]

      return z

class TSTEncoder(nn.Module):
   def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                attn_dropout=0., dropout=0., activation='gelu', n_layers=1):
      super().__init__()

      self.layers = nn.ModuleList(
         [TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                          attn_dropout=attn_dropout, dropout=dropout,
                          activation=activation) for i in range(n_layers)])

   def forward(self, src, key_padding_mask = None, attn_mask = None):
      output = src
      scores = None
      for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask,
                                                      attn_mask=attn_mask)
      return output


class TSTEncoderLayer(nn.Module):
   def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256,
                attn_dropout=0, dropout=0., bias=True, activation="gelu",):
      super().__init__()
      assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
      d_k = d_model // n_heads if d_k is None else d_k
      d_v = d_model // n_heads if d_v is None else d_v

      # Multi-Head attention
      self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout,)

      # Add & Norm
      self.dropout_attn = nn.Dropout(dropout)
      self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))

      # Position-wise Feed-Forward
      self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                              get_activation_fn(activation),
                              nn.Dropout(dropout),
                              nn.Linear(d_ff, d_model, bias=bias))

      # Add & Norm
      self.dropout_ffn = nn.Dropout(dropout)
      self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))


   def forward(self, src, prev = None, key_padding_mask = None,  attn_mask = None):

      ## Multi-Head attention
      src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask,
                                             attn_mask=attn_mask)
      ## Add & Norm
      src = src + self.dropout_attn(src2)  # Add: residual connection with residual dropout
      src = self.norm_attn(src)

      ## Position-wise Feed-Forward
      src2 = self.ff(src)
      ## Add & Norm
      src = src + self.dropout_ffn(src2)  # Add: residual connection with residual dropout
      src = self.norm_ffn(src)

      return src, scores


class _MultiheadAttention(nn.Module):
   def __init__(self, d_model, n_heads, d_k=None, d_v=None, attn_dropout=0., proj_dropout=0.,
                qkv_bias=True, lsa=False):
      """Multi Head Attention Layer
      Input shape:
          Q:       [batch_size (bs) x max_q_len x d_model]
          K, V:    [batch_size (bs) x q_len x d_model]
          mask:    [q_len x q_len]
      """
      super().__init__()
      d_k = d_model // n_heads if d_k is None else d_k
      d_v = d_model // n_heads if d_v is None else d_v

      self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

      self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
      self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
      self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

      # Scaled Dot-Product Attention (multiple heads)
      self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, lsa=lsa)

      # Poject output
      self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

   def forward(self, Q, K = None, V = None, prev = None,
               key_padding_mask = None, attn_mask = None):

      bs = Q.size(0)
      if K is None: K = Q
      if V is None: V = Q

      # Linear (+ split in multiple heads)
      q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,
                                                                       2)  # q_s    : [bs x n_heads x max_q_len x d_k]
      k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3,
                                                                     1)  # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
      v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)  # v_s    : [bs x n_heads x q_len x d_v]

      # Apply Scaled Dot-Product Attention (multiple heads)
      output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask,
                                                           attn_mask=attn_mask)
      # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

      # back to the original inputs dimensions
      output = output.transpose(1, 2).contiguous().view(bs, -1,
                                                        self.n_heads * self.d_v)  # output: [bs x q_len x n_heads * d_v]
      output = self.to_out(output)

      return output, attn_weights, attn_scores


class _ScaledDotProductAttention(nn.Module):
   r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
   (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
   by Lee et al, 2021)"""

   def __init__(self, d_model, n_heads, attn_dropout=0., lsa=False):
      super().__init__()
      self.attn_dropout = nn.Dropout(attn_dropout)
      head_dim = d_model // n_heads
      self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
      self.lsa = lsa

   def forward(self, q, k, v, prev = None,
               key_padding_mask = None, attn_mask = None):
      '''
      Input shape:
          q               : [bs x n_heads x max_q_len x d_k]
          k               : [bs x n_heads x d_k x seq_len]
          v               : [bs x n_heads x seq_len x d_v]
          prev            : [bs x n_heads x q_len x seq_len]
          key_padding_mask: [bs x seq_len]
          attn_mask       : [1 x seq_len x seq_len]
      Output shape:
          output:  [bs x n_heads x q_len x d_v]
          attn   : [bs x n_heads x q_len x seq_len]
          scores : [bs x n_heads x q_len x seq_len]
      '''

      # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
      attn_scores = torch.matmul(q, k) * self.scale  # attn_scores : [bs x n_heads x max_q_len x q_len]

      # Add pre-softmax attention scores from the previous layer (optional)
      if prev is not None: attn_scores = attn_scores + prev

      # Attention mask (optional)
      if attn_mask is not None:  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
         if attn_mask.dtype == torch.bool:
            attn_scores.masked_fill_(attn_mask, -np.inf)
         else:
            attn_scores += attn_mask

      # Key padding mask (optional)
      if key_padding_mask is not None:  # mask with shape [bs x q_len] (only when max_w_len == q_len)
         attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

      # normalize the attention weights
      attn_weights = F.softmax(attn_scores, dim=-1)  # attn_weights   : [bs x n_heads x max_q_len x q_len]
      attn_weights = self.attn_dropout(attn_weights)

      # compute the new values given the attention weights
      output = torch.matmul(attn_weights, v)  # output: [bs x n_heads x max_q_len x d_v]

      return output, attn_weights, attn_scores


class Transpose(nn.Module):
   def __init__(self, *dims, contiguous=False):
      super().__init__()
      self.dims, self.contiguous = dims, contiguous

   def forward(self, x):
      if self.contiguous:
         return x.transpose(*self.dims).contiguous()
      else:
         return x.transpose(*self.dims)


def get_activation_fn(activation):
   if callable(activation):
      return activation()
   elif activation.lower() == "relu":
      return nn.ReLU()
   elif activation.lower() == "gelu":
      return nn.GELU()
   raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')

# pos_encoding
def positional_encoding(q_len, d_model):
   # Positional encoding
   W_pos = torch.empty((q_len, d_model))
   nn.init.uniform_(W_pos, -0.02, 0.02)
   return nn.Parameter(W_pos, requires_grad=True)