#!/bin/bash

export NCCL_P2P_DISABLE=1

python main_trendformer.py --data ETTm2 \
--in_len 336 --out_len 96 --pred_len 96 --win_size 12 --hop_len 6 \
--learning_rate 2e-3 --itr 5 --keepratio 1 --e_layers 1 --d_model 64 --d_ff 128 \
--dropout 0.1 --batch_size 128 --n_components 3 --patience 10 --train_epochs 10 \
--use_gmm --use_filter --filter stft

python main_trendformer.py --data ETTm2 \
--in_len 336 --out_len 192 --pred_len 192 --win_size 12 --hop_len 6 \
--learning_rate 1e-3 --itr 5 --keepratio 1 --e_layers 1 --d_model 128 --d_ff 256 \
--dropout 0.3 --batch_size 128 --n_components 3 --patience 10 --train_epochs 10 \
--use_gmm --use_filter --filter stft

python main_trendformer.py --data ETTm2 \
--in_len 336 --out_len 336 --pred_len 336 --win_size 12 --hop_len 3 \
--learning_rate 1e-3 --itr 5 --keepratio 1 --e_layers 1 --d_model 64 --d_ff 128 \
--dropout 0.2 --batch_size 128 --n_components 3 --patience 10 --train_epochs 10 \
--use_gmm --use_filter --filter stft

python main_trendformer.py --data ETTm2 \
--in_len 720 --out_len 720 --pred_len 720 --win_size 24 --hop_len 3 \
--learning_rate 5e-4 --itr 5 --keepratio 1 --e_layers 1 --d_model 64 --d_ff 128 \
--dropout 0.2 --batch_size 128 --n_components 3 --patience 10 --train_epochs 10 \
--use_gmm --use_filter --filter stft
