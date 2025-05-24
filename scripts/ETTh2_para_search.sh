#!/bin/bash

export NCCL_P2P_DISABLE=1

for win in 24 12
do
for hoplen in 12 6 3
do
for comp in 1 2 3 4
do
for lr in 2e-4 1e-4 5e-5 2e-5 1e-5
do
for drop in 0.1 0.2 0.3
do
python main_trendformer.py --data ETTh2 \
--in_len 720 --out_len 720 --pred_len 720 --win_size $win --hop_len $hoplen \
--learning_rate $lr --itr 1 --e_layers 1 --batch_size 128 --dropout $drop \
--d_model 64 --d_ff 128 --patience 20 --train_epochs 20 --n_components $comp \
--use_filter --filter stft --use_gmm
done
done
done
done
done