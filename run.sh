#! /bin/bash
set -e

if [ "$1" = "amazon670k" ]; then
    echo "Start training, dataset is" $1

    CUDA_VISIBLE_DEVICES=2 python src/main.py --lr 1e-4 --epoch 5 --dataset amazon670k --swa --swa_warmup 4 --swa_step 3000 --batch 4 --max_len 64 --eval_step 3000 --group_y_candidate_num 2000 --group_y_candidate_topk 75 --valid  --hidden_dim 256 --group_y_group 0
elif [ "$1" = "wiki500k" ]; then
    echo "Start training, dataset is" $1

    python src/main.py --lr 1e-4 --epoch 15 --dataset amazon670k --batch 16 --max_len 128
elif [ "$1" = "amazoncat13k" ]; then
    echo "Start training, dataset is" $1

    python src/main.py --lr 1e-4 --epoch 15 --dataset amazon670k --batch 16 --max_len 128
elif [ "$1" = "wiki31k" ]; then
    echo "Start training, dataset is" $1

    python src/main.py --lr 1e-4 --epoch 15 --dataset amazon670k --batch 16 --max_len 128
elif [ "$1" = "eurlex4k" ]; then
    echo "Start training, dataset is" $1

    python src/main.py --lr 1e-4 --epoch 15 --dataset amazon670k --batch 16 --max_len 128
fi