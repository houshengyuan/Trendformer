import argparse
import os
import torch
import inspect

from exp.exp_trendformer import Exp_trendformer
from utils.tools import string_split

seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='TrendFormer')

parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
parser.add_argument('--root_path', type=str, default='./datasets/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')  
parser.add_argument('--data_split', type=str, default='0.7,0.1,0.2',help='train/val/test split, can be ratio or number')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location to store model checkpoints')

parser.add_argument('--in_len', type=int, default=96, help='input MTS length (T)')
parser.add_argument('--out_len', type=int, default=24, help='output MTS length (\tau)')
parser.add_argument('--pred_len', type=int, default=None, help='length of prediction if None take out_len by default')
parser.add_argument('--win_size', type=int, default=6, help='segment window length (L_seg)')
parser.add_argument('--merge_size', type=int, default=1, help='size for segment merge')
parser.add_argument('--hop_len', type=int, default=3, help='num of fft/window size in STFT')

parser.add_argument('--backbone', type=str, default="iTransformer", help='The backbone of our architecture')
parser.add_argument('--data_dim', type=int, default=7, help='Number of dimensions of the MTS data (D)')
parser.add_argument('--d_model', type=int, default=256, help='dimension of hidden states (d_model)')
parser.add_argument('--d_ff', type=int, default=512, help='dimension of MLP in transformer')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers (N)')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
parser.add_argument('--keepratio', type=float, default=1, help='keepratio')
parser.add_argument('--n_components', type=int, default=3, help='component number of GMM')
parser.add_argument('--sample_len', type=int, default=1, help='stride of history sliding window sample when constructing the histogram')

parser.add_argument('--use_filter', action='store_true', help='whether to use filter to decouple trend and priodicity for prediction', default=False)
parser.add_argument('--use_gmm', action='store_true', help='whether to use GMM to deal with series', default=False)
parser.add_argument('--filter', type=str, help='the name of filter', default='stft')

parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--warm_train', type=int, default=0, help='ignore validation metrics before some epoch')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer initial learning rate')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--loss_bound', type=float, default=0.8, help='loss bound for early stopping')

parser.add_argument('--save_pred', action='store_true', help='whether to save the predicted future MTS', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]
    print(args.gpu)

data_parser = {
    'ETTh1':{'data':'ETTh1.csv', 'data_dim':7, 'split':[12*30*24, 4*30*24, 4*30*24]},
    'ETTh2':{'data':'ETTh2.csv', 'data_dim':7, 'split':[12*30*24, 4*30*24, 4*30*24]},
    'ETTm1':{'data':'ETTm1.csv', 'data_dim':7, 'split':[4*12*30*24, 4*4*30*24, 4*4*30*24]},
    'ETTm2':{'data':'ETTm2.csv', 'data_dim':7, 'split':[4*12*30*24, 4*4*30*24, 4*4*30*24]},
    'WTH':{'data':'weather.csv', 'data_dim':21, 'split':[0.7, 0.1, 0.2]},#[28*30*24, 10*30*24, 10*30*24]},
    'ECL':{'data':'ECL.csv', 'data_dim':321, 'split':[0.7, 0.1, 0.2]},#[15*30*24, 3*30*24, 4*30*24]},
    'ILI':{'data':'national_illness.csv', 'data_dim':7, 'split':[0.7, 0.1, 0.2]},
    'Traffic':{'data':'traffic.csv', 'data_dim':862, 'split':[0.7, 0.1, 0.2]},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.data_dim = data_info['data_dim']
    args.data_split = data_info['split']
else:
    args.data_split = string_split(args.data_split)

print('Args in experiment:')
print(args)

Exp = Exp_trendformer

for ii in range(1, args.itr + 1):
    setting = 'Trendformer_{}_{}_il{}_ol{}_pred{}_win{}{}{}_merge{}_hop{}_dm{}_hid{}_nh{}_el{}_keep{}_lr{}_bsz{}{}{}_drop{}_itr{}'.format(args.backbone, args.data,
                args.in_len, args.out_len, args.pred_len, args.win_size, "_filter" if args.use_filter else "", "_"+args.filter if args.use_filter else "", args.merge_size, args.hop_len,
                args.d_model, args.d_ff, args.n_heads, args.e_layers, args.keepratio, args.learning_rate, args.batch_size, "_gmm" if args.use_gmm else "", "_comp"+str(args.n_components) if args.use_gmm else "", args.dropout, ii)

    exp = Exp(args) # set experiments
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, args.save_pred)

    torch.cuda.empty_cache()