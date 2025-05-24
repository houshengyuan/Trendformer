#!/bin/bash

export NCCL_P2P_DISABLE=1

for pred in 96 192 336 720
do
python main_trendformer.py --data ETTh2 \
--in_len 720 --out_len 720 --pred_len $pred --win_size 24 --hop_len 6 \
--learning_rate 5e-5 --itr 5 --e_layers 1 --batch_size 128 --dropout 0.3 \
--d_model 128 --d_ff 256 --patience 20 --train_epochs 20 --n_components 4 \
--use_filter --filter stft --use_gmm
done