#!/usr/bin/env bash

mkdir -p checkpoints

args="
--data data/reddit \
--tokenizer_vocab data/reddit/vocab/en-vocab.json \
--tokenizer_merges data/reddit/vocab/en-merges.txt \
--nlayers 12 \
--hid-sz 512 \
--inner-hid-sz 2048 \
--nheads 8 \
--attn-span 1024 \
--block-sz 256 \
--batch-sz 144 \
--lr 2e-4 \
--momentum 0 \
--dropout 0.1 \
--optim radam \
--lr-warmup 12000 \
--grad-clip 1.0 \
--niter 250 \
--nbatches 1000 \
--adapt-span \
--adapt-span-loss 0.0000005 \
--adapt-span-cache \
--batch-split 3 \
--checkpoint checkpoints/reddit_small.pt
"


echo "Training ..."
# using the pytorch distributed launching
python3 main.py $args


echo "Evaluation ..."
# use a smaller batch size to reduce tokens without context and omitted tokens.
python3 main.py $args --full-eval-mode --batch-sz 32
