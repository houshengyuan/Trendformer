#!/bin/bash

export NCCL_P2P_DISABLE=1

for win in 12 24
do
for hop in 3 6
do
for comp in 1 2 3
do
for lr in 1e-2 2e-3 1e-3 5e-4
do
python main_trendformer.py --data ECL \
--in_len 720 --out_len 720 --pred_len 720 --win_size $win --hop_len $hop --merge_size 2 \
--learning_rate $lr --itr 1 --e_layers 2 --d_model 512 --d_ff 512 \
--dropout 0.1 --batch_size 16 --n_components $comp  --sample_len 2 \
--train_epochs 20 --patience 20 --use_filter --use_gmm --filter stft
done
done
done
done