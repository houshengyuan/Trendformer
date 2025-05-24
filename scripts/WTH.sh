#!/bin/bash

export NCCL_P2P_DISABLE=1

for pred in 96 192 336 720
do
python main_trendformer.py --data WTH \
--in_len 720 --out_len 720 --pred_len $pred --win_size 12 --hop_len 6 --e_layers 1 \
--learning_rate 2e-4 --itr 5 --keepratio 1 --dropout 0.3 --batch_size 16 --d_model 128 --d_ff 256 --n_components 4 \
--train_epochs 20 --patience 20 --use_gmm --use_filter --filter stft
done