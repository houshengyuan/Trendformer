#!/bin/bash

export NCCL_P2P_DISABLE=1

for win in 12 24
do
for hop in 12 6 3
do
for comp in 1 2 3
do
for lr in 1e-2 5e-3 2e-3 1e-3 5e-4 2e-4
do
for drop in 0.1 0.2 0.3
do
python main_trendformer.py --data ETTm2 \
--in_len 720 --out_len 720 --pred_len 720 --win_size $win --hop_len $hop \
--learning_rate $lr --itr 1 --keepratio 1 --e_layers 1 --d_model 32 --d_ff 64 \
--dropout $drop --batch_size 128 --n_components $comp --patience 10 --train_epochs 10 \
--use_gmm --use_filter --filter stft
done
done
done
done
done