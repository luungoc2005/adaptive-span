#!/usr/bin/env bash

mkdir -p checkpoints

args="
--data data/reddit \
--tokenizer_vocab data/reddit/vocab/en-vocab.json \
--tokenizer_merges data/reddit/vocab/en-merges.txt \
--nlayers 12 \
--hid-sz 256 \
--inner-hid-sz 1024 \
--nheads 4 \
--attn-span 2048 \
--block-sz 256 \
--batch-sz 128 \
--lr 5e-4 \
--momentum 0 \
--dropout 0.1 \
--optim radam \
--lr-warmup 10000 \
--grad-clip 1.0 \
--niter 150 \
--nbatches 1000 \
--adapt-span \
--adapt-span-loss 0.000002 \
--adapt-span-cache \
--batch-split 2 \
--checkpoint checkpoints/reddit_small.pt
"


echo "Training ..."
# using the pytorch distributed launching
python3 main.py $args


echo "Evaluation ..."
# use a smaller batch size to reduce tokens without context and omitted tokens.
python3 main.py $args --full-eval-mode --batch-sz 32
