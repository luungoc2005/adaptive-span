#!/usr/bin/env bash

mkdir -p checkpoints

args="
--data data/reddit \
--tokenizer_vocab data/reddit/vocab/en-vocab.json \
--tokenizer_merges data/reddit/vocab/en-merges.txt \
--nlayers 2 \
--hid-sz 128 \
--inner-hid-sz 512 \
--nheads 4 \
--attn-span 512 \
--block-sz 512 \
--batch-sz 64 \
--optim radam \
--lr 3e-4 \
--momentum 0 \
--dropout .1 \
--lr-warmup 8000 \
--grad-clip 0.5 \
--niter 150 \
--nbatches 1000 \
--checkpoint checkpoints/reddit_small.pt
"


echo "Training ..."
# using the pytorch distributed launching
python3 main.py $args


echo "Evaluation ..."
# use a smaller batch size to reduce tokens without context and omitted tokens.
python3 main.py $args --full-eval-mode --batch-sz 32
