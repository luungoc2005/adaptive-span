#!/usr/bin/env bash

args="
--data data/reddit \
--tokenizer_vocab data/reddit/vocab/en-vocab.json \
--tokenizer_merges data/reddit/vocab/en-merges.txt \
--nlayers 12 \
--hid-sz 512 \
--inner-hid-sz 2048 \
--nheads 8 \
--attn-span 4096 \
--block-sz 512 \
--batch-sz 64 \
--lr 3e-4 \
--momentum 0 \
--dropout 0.15 \
--optim adamw \
--lr-warmup 32000 \
--grad-clip 1.0 \
--niter 250 \
--nbatches 1000 \
--adapt-span \
--adapt-span-loss 0.0000005 \
--adapt-span-cache \
--batch-split 2 \
--checkpoint checkpoints/reddit_small.pt
"


echo "Launching ..."
# using the pytorch distributed launching
python3 reddit_interactive.py $args
