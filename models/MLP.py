import torch
import torch.nn as nn
from einops import rearrange, repeat
from models.model_basic import Base_Model


class MLP_Trendformer(Base_Model):
    def __init__(self, data_dim, in_len, out_len, use_filter, win_size, merge_size = 2, hop_len = 3,
                filter = "stft", d_model = 512, d_ff = 1024, n_heads = 8, e_layers = 3,
                dropout = 0.0, keepratio = 0.75, use_gmm = True, n_components = 3, dataset = "ETTh1"):
        super().__init__(data_dim, in_len, out_len, use_filter, win_size, merge_size, hop_len, filter, d_model, d_ff, n_heads, e_layers,
                dropout, keepratio, use_gmm, n_components, dataset)

        base_predictor = MLP_Series_predictor(self.data_dim, self.in_len, self.out_len, self.win_size, self.use_filter, self.merge_size, self.hop_len,
                                       self.filter, d_model, d_ff, n_heads, e_layers, dropout, self.keepratio, use_gmm=self.use_gmm, n_components=self.n_components)
        if self.use_filter:
            self.trend_model = base_predictor
        else:
            self.series_model = base_predictor
    
    def init_gmm(self, new_mean, new_variance):
        super().init_gmm_mean(new_mean)
        super().init_gmm_variance(new_variance)

class MLP_Series_predictor(nn.Module):
    def __init__(self, data_dim, in_len, out_len, win_size, use_filter, merge_size=2, hop_len=3,
                 filter="stft", d_model=512, d_ff=1024, n_heads=8, e_layers=3,
                 dropout=0.0, keepratio=0.75, use_gmm=True, n_components=3):
        super().__init__()
        self.data_dim = data_dim
        self.in_len = in_len+(hop_len - in_len%hop_len)%hop_len
        self.out_len = out_len
        self.win_size = win_size
        self.use_filter = use_filter
        self.filter = filter
        self.merge_size = merge_size
        self.keep_ratio = keepratio
        self.e_layers = e_layers
        self.use_gmm = use_gmm
        self.n_components = n_components
        self.reinv=True

        # The padding operation to handle invisible sgemnet length
        if self.use_filter:
            self.in_seg_num, self.out_seg_num = self.filter.generate_trend_in_out_len(self.in_len, self.out_len)
        else:
            pad_len = (hop_len - self.in_len % hop_len) % hop_len
            self.in_seg_num = self.in_len + pad_len
            self.in_nodisclo_seg_num = self.in_len + pad_len
            self.out_seg_num = self.out_len

        # Embedding
        self.enc_value_embedding = nn.Linear(self.in_seg_num, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, 1, d_model))
        self.pre_norm = nn.LayerNorm(d_model)

        # Encoder
        encode_list = []

        for i in range(e_layers):
            if i>0 and self.merge_size>1:
                encode_list.append(nn.Linear(d_model//(merge_size**(i-1)), d_model//(merge_size**i)))
                encode_list.append(nn.LayerNorm(d_model//(merge_size**i)))

            encode_list.append(MLPLayer(d_model//(merge_size**i), d_ff//(merge_size**i), dropout))
        
        self.encoder = nn.Sequential(*encode_list)

        self.linear_final = nn.Linear(d_model//(self.merge_size**(e_layers-1)) if e_layers>0 and self.merge_size>1 else d_model, self.out_seg_num)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x_seq):
        if self.use_gmm:
            x_seq_embed = rearrange(x_seq, "batch var seq_len gmm->batch (var gmm) seq_len")
            x_seq_embed = self.enc_value_embedding(x_seq_embed)
        else:
            x_seq_embed = self.enc_value_embedding(x_seq)

        x_seq_embed = self.dropout(self.pre_norm(x_seq_embed+self.enc_pos_embedding))

        x_seq_embed = self.encoder(x_seq_embed)

        if self.use_gmm:        
            predict_y = torch.sigmoid(self.linear_final(x_seq_embed))
            predict_y = rearrange(predict_y,"batch (var gmm) out_len->batch var out_len gmm",gmm=self.n_components)
            predict_y = torch.clamp(predict_y, min=1e-5, max=1-1e-5)
        else:
            predict_y = self.linear_final(x_seq_embed)
            
        return predict_y
    
class MLPLayer(nn.Module):
    def __init__(self, d_model, d_ff = None, dropout = 0.1):
        super().__init__()
        d_ff = d_ff or 4*d_model
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model))
    
    def forward(self, dim_in):
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        final_out = self.norm(dim_in)
        return final_out