import torch
import torch.nn as nn
from einops import rearrange, repeat
from models.filter import STFT,MA,HP,STL

from math import ceil

class Base_Model(nn.Module):
    def __init__(self, data_dim, in_len, out_len, use_filter, win_size, merge_size = 2, hop_len = 3,
                filter = "stft", d_model = 512, d_ff = 1024, n_heads = 8, e_layers = 3,
                dropout = 0.0, keepratio = 0.75, use_gmm = True, n_components = 3, dataset = "ETTh1"):
        super().__init__()
        self.data_dim = data_dim
        self.in_len = in_len
        self.out_len = out_len
        self.use_filter = use_filter
        self.hop_len = hop_len
        self.win_size = win_size
        self.merge_size = merge_size
        self.keepratio = keepratio
        self.use_gmm = use_gmm
        self.n_components = n_components
        self.gmm_mean = None
        self.gmm_variance = None
        self.only_trend = False

        self.alpha1 = 0.070565992
        self.alpha2 = 1.5976

        filter_dict = {"stft":STFT, "ma":MA, "hp":HP, "stl":STL}
        if self.use_filter:
            self.filter = filter_dict[filter](self.win_size, self.hop_len, self.keepratio, config_name = dataset + "_{0}".format(self.in_len) + "_{0}".format(self.out_len))        
            self.trend_model = None
            self.period_model = Frequency_predictor(self.data_dim, self.in_len, self.out_len, self.win_size, self.merge_size, self.hop_len,
                                        self.filter, d_model, d_ff, n_heads, e_layers, dropout, self.keepratio)
        else:
            self.series_model = None
            self.filter = None
    
    def init_gmm_mean(self, new_value):
        self.gmm_mean = new_value

    def init_gmm_variance(self, new_value):
        self.gmm_variance = new_value

    def padding_layer(self, x):
        if self.in_len%self.hop_len==0:
            return x
        pad_len = (self.hop_len - self.in_len % self.hop_len) % self.hop_len
        x = rearrange(x, "batch seq_len var -> batch var seq_len")
        replicate = nn.ReplicationPad1d((pad_len,0))
        x = replicate(x)
        x = rearrange(x, "batch var seq_len -> batch seq_len var")
        return x

    def smooth_stft_transform(self, history_pad, x, center = False):
        batch, ts_len, ts_dim = history_pad.shape
        left_pad_len = self.win_size - self.hop_len + (self.in_len + self.win_size//2 - self.win_size)%self.hop_len
        total = torch.cat((history_pad[:,-left_pad_len:,:], x, x[:,-self.win_size//2-1:-1,:].flip(dims=[1])), dim=1)
        total = rearrange(total, 'b input_len d -> (b d) input_len')
        x_stft = torch.stft(total, n_fft=self.win_size, hop_length=self.hop_len, center=center, return_complex=False)
        x_stft = rearrange(x_stft, '(b d) channel seg_num spec_phase-> b d channel seg_num spec_phase', b=batch)
        return x_stft[:, :, 0, :, 0], x_stft[:, :, 1:1+int(self.keepratio*self.win_size // 2), :, :]

    def stft_transform(self, x, center = True):
        batch, ts_len, ts_dim = x.shape
        x_series = rearrange(x, 'b input_len d -> (b d) input_len')
        x_stft = torch.stft(x_series, n_fft=self.win_size, hop_length=self.hop_len, center=center, return_complex=False)
        x_stft = rearrange(x_stft, '(b d) channel seg_num spec_phase-> b d channel seg_num spec_phase', b=batch)
        return x_stft[:, :, 0, :, 0], x_stft[:, :, 1:1+int(self.keepratio*self.win_size // 2), :, :]

    def istft_transform(self, outputs_trend, outputs_period, center = True):
        outputs_trend = torch.stack((outputs_trend,torch.zeros_like(outputs_trend)),dim=-1).unsqueeze(2)
        outputs = torch.cat((outputs_trend,outputs_period), dim=2)
        batch, ts_dim, channel, seg_num, spec_phase = outputs.shape
        x_stft_series = rearrange(outputs, 'b d channel seg_num spec_phase -> (b d) channel seg_num spec_phase')
        x = torch.istft(torch.complex(x_stft_series[:,:,:,0],x_stft_series[:,:,:,1]), n_fft=self.win_size, hop_length=self.hop_len, center=center, return_complex=False)
        x = rearrange(x, '(b d) input_len -> b input_len d',b=batch)
        return x

    def gmm_inference(self, outputs):
        id_min = torch.argmin(torch.abs(outputs - 0.5), dim=-1)
            
        outputs_prob = torch.gather(outputs,-1,id_min.unsqueeze(-1)).squeeze(-1)
        outputs_prob = self.inv_page_gauss_radial_basis(outputs_prob)

        # .unsqueeze(1).unsqueeze(0)
        target_mean = torch.gather(self.gmm_mean.repeat(outputs_prob.shape[0],1,outputs_prob.shape[2],1),-1,id_min.unsqueeze(-1)).squeeze(-1)
        target_variance = torch.gather(self.gmm_variance.repeat(outputs_prob.shape[0],1,outputs_prob.shape[2],1),-1,id_min.unsqueeze(-1)).squeeze(-1)
            
        outputs_prob = outputs_prob * torch.sqrt(target_variance) + target_mean
        return outputs_prob

    def stft_inference(self, outputs_trend, outputs_period):
        if self.win_size//2-int(self.keepratio*self.win_size//2)>0:
            if self.keepratio>0:
                outputs_period_pad = torch.zeros_like(outputs_period[:,:,0].unsqueeze(2)).repeat((1, 1, self.win_size//2-int(self.keepratio*self.win_size//2), 1, 1))
                outputs_period = torch.cat([outputs_period, outputs_period_pad], dim=2)
            else:
                outputs_period = torch.zeros((outputs_period.shape[0], outputs_period.shape[1], self.win_size//2, outputs_period.shape[3], outputs_period.shape[4])).to(outputs_period.device)
        
        outputs = self.istft_transform(outputs_trend, outputs_period)[:, -self.out_len:, :]
        return outputs

    def page_gauss_radial_basis(self, x):
        tmp_x = (x.unsqueeze(-1) - self.gmm_mean)/torch.sqrt(self.gmm_variance)
        cdf_x = 1 - 1/(1 + torch.exp(self.alpha1*torch.pow(tmp_x,3) + self.alpha2*tmp_x))
        return cdf_x

    def inv_page_gauss_radial_basis(self, x):
        c = torch.log(1/x-1)
        itm1 = -c/(2*self.alpha1)
        itm2 = torch.sqrt(itm1**2+self.alpha2**3/(27*self.alpha1**3))
        outputs = torch.pow(itm1+itm2,1/3)+torch.sign(itm1-itm2)*torch.pow(torch.abs(itm1-itm2),1/3)
        return outputs

    def trend_period_forward(self, x, y):
        trend_x, period_x = self.filter.transform(x)
        x_mean = torch.mean(trend_x, dim=-1, keepdim=True)
        x_var = torch.var(trend_x, dim=-1, keepdim=True)
        x_var[x_var<=1e-5] = 1
        x_std = torch.sqrt(x_var + 1e-5)
        trend_x = (trend_x - x_mean)/x_std
        
        if self.use_gmm:
            trend_x = self.page_gauss_radial_basis(trend_x)
        
        predict_trend = self.trend_model(trend_x)
        predict_periodicity = self.period_model(period_x)
        
        if self.training:
            trend_y, period_y = self.filter.smooth_transform(x, y)
            trend_y = (trend_y - x_mean)/x_std
            
            if self.use_gmm:
                trend_y = self.page_gauss_radial_basis(trend_y)
            
            trend_y = torch.clamp(trend_y, min=-10, max=10)
            return (predict_trend, predict_periodicity), (trend_y, period_y)
        elif not self.trend_eval:
            if self.use_gmm:
                predict_trend = self.gmm_inference(predict_trend)
            
            predict_trend = predict_trend * x_std + x_mean
            outputs = self.filter.inverse_transform(predict_trend, predict_periodicity, self.out_len)
            return outputs, y
        else:
            if self.use_gmm:
                predict_trend = self.gmm_inference(predict_trend)
            
            predict_trend = predict_trend * x_std + x_mean
            trend_y, period_y = self.filter.smooth_transform(x, y)
            predict_trend = rearrange(predict_trend/self.win_size, "bsz var pred->bsz pred var")
            trend_y = rearrange(trend_y/self.win_size, "bsz var pred->bsz pred var")
            return predict_trend, trend_y

    def series_forward(self, x, y):
        series_x = x.permute((0,2,1))
        x_mean = torch.mean(series_x, dim=-1, keepdim=True)
        x_var = torch.var(series_x, dim=-1, keepdim=True)
        x_var[x_var<=1e-5] = 1
        x_std = torch.sqrt(x_var + 1e-5)
        series_x = (series_x - x_mean)/x_std
        
        if self.use_gmm:
            series_x = self.page_gauss_radial_basis(series_x)
        
        predict_series = self.series_model(series_x)
        
        if self.training:
            series_y = y.permute((0,2,1))
            series_y = (series_y - x_mean)/x_std
            if self.use_gmm:
                series_y = self.page_gauss_radial_basis(series_y)
            series_y = torch.clamp(series_y, min=-10, max=10)
            return predict_series, series_y
        else:
            if self.use_gmm:
                predict_series = self.gmm_inference(predict_series)
            
            outputs = predict_series * x_std + x_mean
            outputs = outputs.permute((0,2,1))

            return outputs, y


    def forward(self, x, y):
        if self.use_filter:
            outputs, label = self.trend_period_forward(x, y)
        else:
            outputs, label = self.series_forward(x, y)
        
        return outputs, label

class Frequency_predictor(nn.Module):
    def __init__(self, data_dim, in_len, out_len, win_size, merge_size = 2, hop_len=3,
                filter="stft", d_model=512, d_ff = 1024, n_heads=8, e_layers=3,
                dropout=0.0, keepratio=0.75):
        super().__init__()
        self.data_dim = data_dim
        self.in_len = in_len+(hop_len - in_len%hop_len)%hop_len
        self.out_len = out_len
        self.win_size = win_size
        self.filter = filter
        self.merge_size = merge_size
        self.keep_ratio = keepratio

        # The padding operation to handle invisible sgemnet length
        self.in_seg_num, self.out_seg_num = self.filter.generate_period_in_out_len(self.in_len, self.out_len)

        self.embed = nn.Sequential(nn.Linear(self.in_seg_num, d_model),
                                   nn.BatchNorm1d(d_model))
        
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model),
                                  nn.GELU(),
                                  nn.Linear(d_model, d_model))

        self.linear_final = nn.Linear(d_model, self.out_seg_num)
        self.dropout = nn.Dropout(p=dropout)
        

    def forward(self, x_seq):
        x_seq_embed = self.filter.input_rearrange(x_seq)
        x_seq_embed = self.embed(x_seq_embed)
        hidden = self.MLP1(x_seq_embed)
        hidden = x_seq_embed + self.dropout(hidden)
        predict_y = self.linear_final(hidden)
        predict_y = self.filter.output_rearrange(predict_y, x_seq.shape) 
        return predict_y