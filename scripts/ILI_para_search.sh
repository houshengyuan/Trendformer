#!/bin/bash

export NCCL_P2P_DISABLE=1

for inlen in 144 104
do
for lr in 5e-2 2e-2 1e-2 5e-3 2e-3 1e-3
do
for drop in 0.1 0.2 0.3
do
for comp in 4 3 2 1
do
python main_trendformer.py --data ILI \
--in_len $inlen --out_len 60 --pred_len 60 --merge_size 1 \
--learning_rate $lr --itr 2 --keepratio 1 --e_layers 1 --d_model 16 --d_ff 32 \
--dropout $drop --batch_size 16 --train_epochs 100 --patience 100 --warm_train 20 --n_components $comp --use_gmm
done
done
done
done