#!/bin/bash

export NCCL_P2P_DISABLE=1

for pred in 60 48 36 24
do
python main_trendformer.py --data ILI \
--in_len 104 --out_len 60 --pred_len $pred --merge_size 1 \
--learning_rate 0.05 --itr 5 --keepratio 1 --e_layers 1 --d_model 16 --d_ff 32 \
--dropout 0.1 --batch_size 16 --train_epochs 100 --patience 100 --warm_train 20 --n_components 3 --use_gmm
done