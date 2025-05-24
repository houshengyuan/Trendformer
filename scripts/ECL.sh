#!/bin/bash

export NCCL_P2P_DISABLE=1

for pred in 96 192 336 720
do
python main_trendformer.py --data ECL \
--in_len 720 --out_len 720 --pred_len $pred --win_size 12 --hop_len 3 --merge_size 2 \
--learning_rate 0.001 --itr 5 --e_layers 2 --d_model 512 --d_ff 512 --n_components 2 \
--dropout 0.1 --batch_size 16 --train_epochs 20 --patience 20 --use_filter --filter stft --use_gmm
done