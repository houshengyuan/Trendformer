import torch.nn.functional

from data.data_loader import Dataset_MTS
from exp.exp_basic import Exp_Basic
from models.trend_former import Trendformer
from models.patchTST import PatchTST_Trendformer
from models.transformer import Transformer_Trendformer
from models.MLP import MLP_Trendformer

from models.loss import BCE_Trend_MSE_Period, MSE_Trend_Period
from torch.distributions import  Normal

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from einops import rearrange

import inspect 

import os
import time
import json
import pickle

import warnings
warnings.filterwarnings('ignore')

class Exp_trendformer(Exp_Basic):
    def __init__(self, args):
        super(Exp_trendformer, self).__init__(args)
    
    def _build_model(self):        
        model_dict = {"iTransformer": Trendformer,
        "PatchTST": PatchTST_Trendformer,
        "MLP": MLP_Trendformer,
        "Transformer": Transformer_Trendformer,
        }
        params = (
            self.args.data_dim,
            self.args.in_len,
            self.args.out_len,
            self.args.use_filter,
            self.args.win_size,
            self.args.merge_size,
            self.args.hop_len,
            self.args.filter,
            self.args.d_model,
            self.args.d_ff,
            self.args.n_heads,
            self.args.e_layers,
            self.args.dropout,
            self.args.keepratio,
            self.args.use_gmm,
            self.args.n_components,
            self.args.data,
        )
        
        model = model_dict[self.args.backbone](*params).float()   
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model 

    def _get_data(self, flag):
        args = self.args

        if flag == 'test':
            shuffle_flag = False; drop_last = False; batch_size = args.batch_size;
        else:
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size;
        data_set = Dataset_MTS(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.in_len, args.out_len, args.pred_len],
            data_split = args.data_split,
            use_gmm = args.use_gmm,
            n_components=args.n_components,
            data_dim=args.data_dim,
            sample_len=args.sample_len,
        )

        self.mean = torch.tensor(getattr(data_set, "mean"), dtype=torch.float32) if hasattr(data_set, "mean") else (self.mean if hasattr(self, "mean") else None)
        self.variance = torch.tensor(getattr(data_set, "variance"), dtype=torch.float32) if hasattr(data_set, "variance") and not hasattr(self, "variance") else (self.variance if hasattr(self, "variance") else None)
        self.weight = torch.tensor(getattr(data_set, "weights"), dtype=torch.float32) if hasattr(data_set, "weights") and not hasattr(self, "weights") else (self.weight if hasattr(self, "weight") else None)
        
        if self.mean is not None and self.variance is not None:
            self.mean = self.mean.to(self.device)
            self.variance = self.variance.to(self.device)
            if self.args.use_multi_gpu and self.args.use_gpu:
                self.model.module.init_gmm(nn.Parameter(self.mean.unsqueeze(1).unsqueeze(0), requires_grad=False), nn.Parameter(self.variance.unsqueeze(1).unsqueeze(0), requires_grad=False))
            else:
                self.model.init_gmm(nn.Parameter(self.mean.unsqueeze(1).unsqueeze(0), requires_grad=False), nn.Parameter(self.variance.unsqueeze(1).unsqueeze(0), requires_grad=False))
        
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self, eval=False):
        train_criterion = {
            "bce_plus_mse": BCE_Trend_MSE_Period,
            "double_mse": MSE_Trend_Period,
            "mse": nn.MSELoss,
            "bce": nn.BCELoss
        }
        if not eval:
            criterion = train_criterion["bce_plus_mse" if self.args.use_gmm and self.args.use_filter else "double_mse" if self.args.use_filter else "bce" if self.args.use_gmm else "mse"]()
        else:
            criterion = nn.MSELoss()
        return criterion

    def pred_analysis(self, pred, true, train_data):
        #len*var
        for i in range(pred.shape[1],6):
            plt.figure(figsize=(30, 10))

            line1, = plt.plot(list(range(1, train_data.shape[0] + 1)), train_data[:,i], color='black')
            line2, = plt.plot(list(range(train_data.shape[0], train_data.shape[0] + pred.shape[0] + 1)), np.concatenate((np.array(train_data[-1,i])[None], pred[:,i]),axis=0), color='red', linestyle='dashed')
            line3, = plt.plot(list(range(train_data.shape[0], train_data.shape[0] + true.shape[0] + 1)), np.concatenate((np.array(train_data[-1,i])[None], true[:,i]),axis=0), color='green', linestyle='dashed')

            plt.legend(handles=[line1, line2, line3], labels=["input", "pred", "true"], loc="upper right", fontsize=6)
            plt.xlabel("time stamp")
            plt.ylabel("time seq value")
            plt.title("var{0}".format(i))
            plt.savefig("re{0}.jpg".format(i))
    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "args.json"), 'w') as f:
            json.dump(vars(self.args), f, indent=True)
        scale_statistic = {'mean': train_data.scaler.mean, 'std': train_data.scaler.std}
        with open(os.path.join(path, "scale_statistic.pkl"), 'wb') as f:
            pickle.dump(scale_statistic, f)
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, warm_train=self.args.warm_train, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion_train = self._select_criterion(eval=False)
        criterion_eval = self._select_criterion(eval=True)

        for epoch in range(self.args.train_epochs):
            time_now = time.time()
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    i, batch_x, batch_y, False, mean=self.mean, variance=self.variance)
                
                loss = criterion_train(pred, true)
                train_loss.append(loss.item())

                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()
                torch.cuda.empty_cache()
                 
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            print(torch.cuda.max_memory_allocated()/(1024**2),flush=True)
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion_eval, mode="valid")
            test_loss = self.vali(test_data, test_loader, criterion_eval, mode="test")
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss),flush=True)
            
            early_stopping(vali_loss, self.model, path, epoch+1)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)

        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        state_dict = self.model.module.state_dict() if isinstance(self.model, DataParallel) else self.model.state_dict()
        torch.save(state_dict, path+'/'+'checkpoint.pth')
        
        return self.model

    def vali(self, vali_data, vali_loader, criterion, mode):
        self.model.eval()

        instance_num = 0
        metrics_all = []
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(vali_loader):
                pred,true = self._process_one_batch(
                        i, batch_x, batch_y, False, mean=self.mean, variance=self.variance)

                batch_size = pred.shape[0]
                instance_num += batch_size
                if mode=="test" and self.args.pred_len is not None:
                    batch_metric = criterion(pred[:,:self.args.pred_len,:], true) * batch_size
                else:
                    batch_metric = criterion(pred, true) * batch_size
                metrics_all.append(batch_metric)

        metrics_all = torch.tensor(metrics_all)
        metrics_mean = metrics_all.sum() / instance_num

        self.model.train()
        return float(metrics_mean.detach().cpu())
    
    def test(self, setting, save_pred = False, inverse = False):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        metrics_all = []
        instance_num = 0

        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(test_loader):
                pred, true = self._process_one_batch(
                    i, batch_x, batch_y, True, mean=self.mean, variance=self.variance)
                batch_size = pred.shape[0]
                instance_num += batch_size
                
                if self.args.pred_len is not None:
                    batch_metric = np.array(metric(pred[:,:true.shape[1],:].detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                else:
                    batch_metric = np.array(metric(pred.detach().cpu().numpy(),true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)

                if save_pred:
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())

        metrics_all = np.stack(metrics_all, axis = 0)
        metrics_mean = metrics_all.sum(axis = 0) / instance_num

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metrics_mean
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        if (save_pred):
            preds = np.concatenate(preds, axis = 0)
            trues = np.concatenate(trues, axis = 0)
            np.save(folder_path+'pred.npy', preds)
            np.save(folder_path+'true.npy', trues)

        return

    def padding_layer(self, x):
        if self.args.in_len%self.args.hop_len==0:
            return x
        pad_len = (self.args.hop_len - self.args.in_len % self.args.hop_len) % self.args.hop_len
        x = rearrange(x, "batch seq_len var -> batch var seq_len")
        replicate = nn.ReplicationPad1d((pad_len,0))
        x = replicate(x)
        x = rearrange(x, "batch var seq_len -> batch seq_len var")
        return x

    def _process_one_batch(self, i, batch_x, batch_y, eval = False, mean = None, variance = None, weights = None):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x = self.padding_layer(batch_x)
        len_x, len_y = batch_x.shape[1], batch_y.shape[1]
        setattr(self.model, "trend_eval", self.model.only_trend and eval)

        output, label = self.model(batch_x, batch_y)
        return output, label

    def eval(self, setting, save_pred = False, inverse = False):
        #evaluate a saved model
        args = self.args
        data_set = Dataset_MTS(
            root_path=args.root_path,
            data_path=args.data_path,
            flag='test',
            size=[args.in_len, args.out_len, args.pred_len],
            data_split = args.data_split,
            scale = True,
            scale_statistic = args.scale_statistic,
            use_gmm = args.use_gmm,
            n_components=args.n_components,
            data_dim=args.data_dim,
            sample_len=args.sample_len,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False)
        
        self.model.eval()
        
        preds = []
        trues = []
        metrics_all = []
        instance_num = 0
        
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(data_loader):
                pred, true = self._process_one_batch(
                    i, batch_x, batch_y, True, mean=self.mean, variance=self.variance)
                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)
                if (save_pred):
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())

        metrics_all = np.stack(metrics_all, axis = 0)
        metrics_mean = metrics_all.sum(axis = 0) / instance_num

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metrics_mean
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        if (save_pred):
            preds = np.concatenate(preds, axis = 0)
            trues = np.concatenate(trues, axis = 0)
            np.save(folder_path+'pred.npy', preds)
            np.save(folder_path+'true.npy', trues)

        return mae, mse, rmse, mape, mspe
