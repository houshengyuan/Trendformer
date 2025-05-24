#!/bin/bash

export NCCL_P2P_DISABLE=1

for win in 24 12
do
for hoplen in 12 6 3
do
for lr in 2e-2 1e-2 5e-3 2e-3 1e-3 5e-4
do
for comp in 1 2 3 4
do
python main_trendformer.py --data ETTm1 \
--in_len 192 --out_len 336 --pred_len 336 --win_size $win --hop_len $hoplen --merge_size 1 \
--learning_rate $lr --itr 1 --keepratio 1 --e_layers 1 --d_model 128 --d_ff 256 \
--dropout 0.2 --batch_size 128 --n_components $comp --train_epochs 20 --patience 20 \
--use_filter --filter stft --use_gmm
done
done
done
done