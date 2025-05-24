import os
import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from joblib import Parallel, delayed
from torch.utils.data import Dataset, DataLoader

from utils.tools import StandardScaler

import warnings
warnings.filterwarnings('ignore')

class Dataset_MTS(Dataset):
    def __init__(self, root_path, data_path='ETTh1.csv', flag='train', size=None, 
                  data_split = [0.7, 0.1, 0.2], scale=True, scale_statistic=None, use_gmm=True, n_components=3, data_dim=7, sample_len=1):
        # size [seq_len, label_len, pred_len]
        # info
        self.in_len = size[0]
        self.out_len = size[1]
        self.pred_len = (size[2] if size[2] is not None else self.out_len)
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.scale = scale

        self.use_gmm = use_gmm
        self.n_components = n_components
        self.data_dim = data_dim
        self.sample_len = sample_len
        self.root_path = root_path
        self.data_path = data_path
        self.data_split = data_split
        self.scale_statistic = scale_statistic
        self.__read_data__()

    def train_gmm(self, row):
        gmm = GaussianMixture(n_components=self.n_components, random_state=0)
        gmm.fit(row.reshape(-1, 1))
        return gmm

    def __gmm_process__(self, ):
        save_path = os.path.join(self.root_path, self.data_path.split('.')[0] + "_in{0}_gmm_comp{1}.pkl".format(self.in_len,self.n_components))
        if not os.path.exists(save_path):
            tmp = pd.DataFrame(self.data_x)
            tmp_list = []
            for j in range(0,self.in_len,self.sample_len):
                tmp_list.append(tmp.shift(j, axis=0).loc[self.in_len//self.sample_len*self.sample_len:].values)
            win_data = np.stack(tmp_list,axis=-1)
            
            # diff = win_data[:,:,1:]-win_data[:,:,:-1]
            # upper_bound = np.quantile(np.abs(diff),0.95, axis=-1, keepdims=True)
            # win_data = np.concatenate([win_data[:,:,0][:,:,np.newaxis],win_data[:,:,:-1]+diff.clip(-upper_bound,upper_bound)],axis=-1)
            
            win_data = (win_data - np.mean(win_data, axis=-1, keepdims=True)) / (
                np.std(win_data, axis=-1, keepdims=True) + 1e-5)
            self.gmm_models = Parallel(n_jobs=-1)(delayed(self.train_gmm)(row) for row in win_data.transpose((1,0,2)).reshape(self.data_dim,-1))
            model_dict = {i:self.gmm_models[i] for i in range(len(self.gmm_models))}
            joblib.dump(model_dict, save_path)
        else:
            self.gmm_models = list(joblib.load(save_path).values())

        self.mean = np.array([g.means_ for g in self.gmm_models]).squeeze(-1)
        self.variance = np.array([g.covariances_ for g in self.gmm_models]).squeeze(-1).squeeze(-1)
        self.weights = np.array([g.weights_ for g in self.gmm_models])

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        if (self.data_split[0] > 1):
            train_num = self.data_split[0]; val_num = self.data_split[1]; test_num = self.data_split[2];
        else:
            train_num = int(len(df_raw)*self.data_split[0])
            test_num = int(len(df_raw)*self.data_split[2])
            val_num = len(df_raw) - train_num - test_num
        border1s = [0, train_num - self.in_len, train_num + val_num - self.in_len]
        border2s = [train_num, train_num+val_num, train_num + val_num + test_num]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            if self.scale_statistic is None:
                self.scaler = StandardScaler()
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
            else:
                self.scaler = StandardScaler(mean = self.scale_statistic['mean'], std = self.scale_statistic['std'])
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.use_gmm:
            self.__gmm_process__()

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.in_len
        r_begin = s_end
        r_end = r_begin + (self.pred_len if self.set_type==2 else self.out_len)

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y
    
    def __len__(self):
        return len(self.data_x) - self.in_len- self.out_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)