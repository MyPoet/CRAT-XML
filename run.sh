#! /bin/bash
set -e

if [ "$1" = "amazon670k" ]; then
    echo "Start training, dataset is" $1

    python src/main.py --lr 1e-4 --epoch 15 --dataset amazon670k --batch 16 --max_len 128 --valid
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