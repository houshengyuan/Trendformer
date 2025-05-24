#!/bin/bash

export NCCL_P2P_DISABLE=1

for win in 24 12
do
for hop in 12 6 3
do
for comp in 1 2 3 4
do
for lr in 5e-2 2e-2 1e-2 5e-3 2e-3 1e-3 5e-4 2e-4 1e-4
do
python main_trendformer.py --data Traffic \
--in_len 720 --out_len 720 --pred_len 720 --win_size $win --hop_len $hop \
--learning_rate $lr --itr 1 --e_layers 1 --d_model 128 --d_ff 256 \
--dropout 0.1 --batch_size 16 --patience 3 --train_epochs 10 \
--use_filter --filter stft --use_gmm --n_components $comp
done
done
done
done