import torch
import statsmodels.tsa.seasonal
import cyl1tf
import hashlib
import json
import os
import multiprocessing as mul
import numpy as np
from torch import nn
from einops import rearrange
from concurrent.futures import ThreadPoolExecutor
from functools import partial

class STFT():
    def __init__(self, win_size, hop_len, keepratio=1.0, config_name = None):
        super(STFT, self).__init__()
        self.win_size = win_size
        self.hop_len = hop_len
        self.keepratio = keepratio

    def generate_trend_in_out_len(self, in_len, out_len):
        trend_in_len = (in_len + self.win_size // 2 * 2 - self.win_size) // self.hop_len + 1
        in_nodisclo_len = (in_len + self.win_size // 2 - self.win_size) // self.hop_len + 1
        trend_out_len = (in_len + out_len + self.win_size // 2 * 2 - self.win_size) // self.hop_len + 1 - in_nodisclo_len
        return trend_in_len, trend_out_len

    def generate_period_in_out_len(self, in_len, out_len):
        period_in_len = (in_len + self.win_size // 2 * 2 - self.win_size) // self.hop_len + 1
        in_nodisclo_len = (in_len + self.win_size // 2 - self.win_size) // self.hop_len + 1
        period_out_len = (in_len + out_len + self.win_size // 2 * 2 - self.win_size) // self.hop_len + 1 - in_nodisclo_len
        return period_in_len, period_out_len
    
    def transform(self, x, center = True):
        batch, in_len, ts_dim = x.shape
        x_series = rearrange(x, 'b input_len d -> (b d) input_len')
        x_stft = torch.stft(x_series, n_fft=self.win_size, hop_length=self.hop_len, center=center, return_complex=False)
        x_stft = rearrange(x_stft, '(b d) channel seg_num spec_phase-> b d channel seg_num spec_phase', b=batch)
        return x_stft[:, :, 0, :, 0], x_stft[:, :, 1:1+int(self.keepratio*self.win_size // 2), :, :]

    def smooth_transform(self, history_pad, x, center = False):
        batch, in_len, ts_dim = history_pad.shape
        left_pad_len = self.win_size - self.hop_len + (in_len + self.win_size//2 - self.win_size)%self.hop_len
        total = torch.cat((history_pad[:,-left_pad_len:,:], x, x[:,-self.win_size//2-1:-1,:].flip(dims=[1])), dim=1)
        total = rearrange(total, 'b input_len d -> (b d) input_len')
        x_stft = torch.stft(total, n_fft=self.win_size, hop_length=self.hop_len, center=center, return_complex=False)
        x_stft = rearrange(x_stft, '(b d) channel seg_num spec_phase-> b d channel seg_num spec_phase', b=batch)
        return x_stft[:, :, 0, :, 0], x_stft[:, :, 1:1+int(self.keepratio*self.win_size // 2), :, :]

    def inverse_transform(self, outputs_trend, outputs_period, out_len, center = True):
        outputs_trend = torch.stack((outputs_trend,torch.zeros_like(outputs_trend)),dim=-1).unsqueeze(2)
        outputs = torch.cat((outputs_trend,outputs_period), dim=2)
        if self.keepratio < 1.0:
            remainder = self.win_size//2 - int(self.keepratio*self.win_size // 2)
            outputs = torch.cat((outputs,torch.zeros_like(outputs_trend).repeat_interleave(remainder, dim=2)), dim=2)
        batch, ts_dim, channel, seg_num, spec_phase = outputs.shape
        x_stft_series = rearrange(outputs, 'b d channel seg_num spec_phase -> (b d) channel seg_num spec_phase')
        x = torch.istft(torch.complex(x_stft_series[:,:,:,0], x_stft_series[:,:,:,1]), n_fft=self.win_size, hop_length=self.hop_len, center=center, return_complex=False)
        x = rearrange(x, '(b d) input_len -> b input_len d', b=batch)[:,-out_len:,:]
        return x
    
    def input_rearrange(self, x_seq):
        x_seq = rearrange(x_seq, "batch ndim nchannel seq_len real_imag->(batch ndim nchannel real_imag) seq_len")
        return x_seq

    def output_rearrange(self, x_seq, shape):
        batch_size, ndim, nchannel = shape[0], shape[1], shape[2]
        x_seq = rearrange(x_seq, "(batch ndim nchannel real_imag) seq_len -> batch ndim nchannel seq_len real_imag", batch=batch_size, ndim=ndim, nchannel=nchannel, real_imag=2)
        return x_seq

class MA():
    def __init__(self, win_size, hop_len = None, keepratio = None, config_name = None):
        super(MA).__init__()
        self.win_size = win_size
        self.ma_filter = nn.AvgPool1d(kernel_size=self.win_size, stride = 1, padding=0)
    
    def generate_trend_in_out_len(self, in_len, out_len):
        trend_in_len = in_len - self.win_size + 1
        trend_out_len = out_len
        return trend_in_len, trend_out_len

    def generate_period_in_out_len(self, in_len, out_len):
        period_in_len = in_len - self.win_size + 1
        period_out_len = out_len
        return period_in_len, period_out_len

    def transform(self, x):
        batch, in_len, ts_dim = x.shape
        input = x.permute((0,2,1))
        x_ma = self.ma_filter(input)
        return x_ma, input[:, :, -(in_len-self.win_size+1):] - x_ma

    def smooth_transform(self, history_pad, x):
        batch, in_len, ts_dim = x.shape
        total = torch.cat((history_pad[:,-(self.win_size-1):,:], x), dim=1)
        total = total.permute((0,2,1))
        x_ma = self.ma_filter(total)
        return x_ma, total[:, :, -in_len:] - x_ma

    def inverse_transform(self, outputs_trend, outputs_period, out_len = None):
        predict = outputs_trend + outputs_period
        predict = predict.permute((0,2,1))
        return predict

    def input_rearrange(self, x_seq):
        x_seq = rearrange(x_seq, "batch ndim seq_len->(batch ndim) seq_len")
        return x_seq

    def output_rearrange(self, x_seq, shape):
        batch_size, ndim = shape[0], shape[1]
        x_seq = rearrange(x_seq, "(batch ndim) seq_len -> batch ndim seq_len", batch=batch_size, ndim=ndim)
        return x_seq

class HP():
    def __init__(self, win_size = None, hop_len = None, keepratio = None, config_name = None):
        super(HP).__init__()
        self.lamb = 10000
    
    def generate_trend_in_out_len(self, in_len, out_len):
        trend_in_len = in_len
        trend_out_len = out_len
        return trend_in_len, trend_out_len

    def generate_period_in_out_len(self, in_len, out_len):
        period_in_len = in_len
        period_out_len = out_len
        return period_in_len, period_out_len

    def transform(self, x):
        batch, in_len, ts_dim = x.shape
        input = x.permute((0,2,1))
        diff_mat = (torch.diag(torch.ones(in_len)) - 2 * torch.diag(torch.ones(in_len-1), 1) + torch.diag(torch.ones(in_len-2), 2))[:in_len-2, :]
        smooth_mat = torch.inverse((torch.eye(in_len,) + self.lamb * torch.matmul(diff_mat.t(), diff_mat))).to(input.device)
        x_hp = torch.matmul(input, smooth_mat.t())
        return x_hp, input - x_hp

    def smooth_transform(self, history_pad, x):
        return self.transform(x)

    def inverse_transform(self, outputs_trend, outputs_period, out_len = None):
        predict = outputs_trend + outputs_period
        predict = predict.permute((0,2,1))
        return predict

    def input_rearrange(self, x_seq):
        x_seq = rearrange(x_seq, "batch ndim seq_len->(batch ndim) seq_len")
        return x_seq

    def output_rearrange(self, x_seq, shape):
        batch_size, ndim = shape[0], shape[1]
        x_seq = rearrange(x_seq, "(batch ndim) seq_len -> batch ndim seq_len", batch=batch_size, ndim=ndim)
        return x_seq

class STL():
    def __init__(self, win_size, hop_len = None, keepratio = None, config_name = None):
        super(STL).__init__()
        self.win_size = win_size
        self.path = "hash/" + config_name + "_hashdict_stl.npz"

        self.solver = statsmodels.tsa.seasonal.STL
        # self.pool = mul.Pool(30)
    
    def fit_data(self, data):
        result = self.solver(data, period = self.win_size, low_pass_jump=2).fit()
        return result.trend

    def generate_trend_in_out_len(self, in_len, out_len):
        trend_in_len = in_len
        trend_out_len = out_len
        return trend_in_len, trend_out_len

    def generate_period_in_out_len(self, in_len, out_len):
        period_in_len = in_len
        period_out_len = out_len
        return period_in_len, period_out_len

    def transform(self, x):
        batch, in_len, ts_dim = x.shape
        input = x.permute((0,2,1))

        if input.device.type == "cpu":
            input_numpy = input.contiguous().view(-1, in_len).numpy()
        else:
            input_numpy = input.contiguous().view(-1, in_len).to("cpu").numpy()
        
        results = list(map(self.fit_data, input_numpy))
        x_stl = torch.tensor(np.array(results), dtype=torch.float32).to(x.device).view(batch, ts_dim, in_len)
        return x_stl, input - x_stl

    def smooth_transform(self, history_pad, x):
        return self.transform(x)

    def inverse_transform(self, outputs_trend, outputs_period, out_len = None):
        predict = outputs_trend + outputs_period
        predict = predict.permute((0,2,1))
        return predict

    def input_rearrange(self, x_seq):
        x_seq = rearrange(x_seq, "batch ndim seq_len->(batch ndim) seq_len")
        return x_seq

    def output_rearrange(self, x_seq, shape):
        batch_size, ndim = shape[0], shape[1]
        x_seq = rearrange(x_seq, "(batch ndim) seq_len -> batch ndim seq_len", batch=batch_size, ndim=ndim)
        return x_seq


    def __init__(self, win_size, hop_len, keepratio=1.0, config_name = None):
        self.win_size = win_size
        self.hop_len = hop_len

    def generate_trend_in_out_len(self, in_len, out_len):
        trend_in_len = in_len // 2 * 2 + 1
        trend_out_len = out_len // 2 * 2 + 1
        return trend_in_len, trend_out_len

    def generate_period_in_out_len(self, in_len, out_len):
        return 0, 0
    
    def transform(self, x, center = True):
        batch, in_len, ts_dim = x.shape
        x_series = rearrange(x, 'b input_len d -> b d input_len')
        x_fft = torch.fft.fft(x_series, dim=-1)[:, :, :(in_len//2)+1]
        x_fft = torch.cat((torch.real(x_fft), torch.imag(x_fft)[:, :, 1:]), dim=-1)
        return x_fft, None

    def smooth_transform(self, history_pad, x, center = False):
        return self.transform(x)

    def inverse_transform(self, outputs_trend, outputs_period, out_len, center = True):
        batch, ts_dim, fft_len, spec_phase = outputs_trend.shape
        x_fft_series = rearrange(outputs_trend, 'b d (fft_len spec_phase) -> b d fft_len spec_phase', spec_phase=2)
        x = torch.fft.ifft(torch.complex(x_fft_series[:,:,:,0], x_fft_series[:,:,:,1]), dim=-1)
        x = rearrange(x, 'b d input_len -> b input_len d')[:,-out_len:,:]
        return x
    
    def input_rearrange(self, x_seq):
        x_seq = rearrange(x_seq, "batch ndim fft_len real_imag->(batch ndim real_imag) fft_len")
        return x_seq

    def output_rearrange(self, x_seq, shape):
        batch_size, ndim = shape[0], shape[1]
        x_seq = rearrange(x_seq, "(batch ndim real_imag) fft_len -> batch ndim fft_len real_imag", batch=batch_size, ndim=ndim, real_imag=2)
        return x_seq