#!/bin/bash

export NCCL_P2P_DISABLE=1

for win in 24 12
do
for hoplen in 12 6 3
do
for comp in 1 2 3 4
do
for lr in 1e-3 5e-4 2e-4 1e-4 5e-5
do
python main_trendformer.py --data WTH \
--in_len 720 --out_len 720 --pred_len 720 --win_size $win --hop_len $hoplen --e_layers 1 \
--learning_rate $lr --itr 1 --dropout 0.2 --batch_size 16 --d_model 128 --d_ff 256 --n_components $comp \
--train_epochs 20 --patience 20 --use_gmm --use_filter --filter stft
done
done
done
done