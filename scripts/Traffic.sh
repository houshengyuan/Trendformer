#!/bin/bash

export NCCL_P2P_DISABLE=1

for pred in 96 192 336
do
python main_trendformer.py --data Traffic \
--in_len 720 --out_len 720 --pred_len $pred --win_size 12 --hop_len 6 \
--learning_rate 2e-3 --itr 5 --e_layers 1 --d_model 128 --d_ff 256 \
--dropout 0.3 --batch_size 16 --patience 3 --train_epochs 10 --use_filter --filter stft --use_gmm --n_components 1
done