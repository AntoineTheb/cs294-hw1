#!/bin/bash
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 1 -l 2 \
-s 32 -b 50000 -lr 0.005 --exp_name hc_best
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 1 -l 2 \
-s 32 -b 50000 -lr 0.005 -rtg --exp_name hc_best_rtg
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 1 -l 2 \
-s 32 -b 50000 -lr 0.005 --nn_baseline --exp_name hc_best_baseline
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 1 -l 2 \
-s 32 -b 50000 -lr 0.005 -rtg --nn_baseline --exp_name hc_best_rtg_baseline
