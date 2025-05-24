#!/bin/bash

export NCCL_P2P_DISABLE=1

for pred in 96 192 336 720
do
python main_trendformer.py --data ETTh1 \
--in_len 336 --out_len 720 --pred_len $pred --win_size 24 --hop_len 12 \
--learning_rate 0.002 --itr 5 --e_layers 1 --batch_size 128 --dropout 0.3 \
--d_model 16 --d_ff 32 --train_epochs 20 --patience 20 --warm_train 5 --n_components 1 \
--use_filter --filter stft --use_gmm
done
