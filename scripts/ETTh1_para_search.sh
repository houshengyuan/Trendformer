#!/bin/bash

export NCCL_P2P_DISABLE=1

for comp in 1 2 3
do
for lr in 1e-2 5e-3 2e-3 1e-3 5e-4
do
for drop in 0.1 0.2 0.3
do
python main_trendformer.py --data ETTh1 \
--in_len 336 --out_len 720 --pred_len 720 --win_size 24 --hop_len 6 \
--learning_rate $lr --itr 1 --e_layers 1 --batch_size 128 --dropout $drop \
--d_model 16 --d_ff 64 --train_epochs 20 --patience 20 --warm_train 5 --n_components $comp \
--use_filter --filter stft --use_gmm
done
done
done
