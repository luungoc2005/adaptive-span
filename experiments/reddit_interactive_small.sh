#!/usr/bin/env bash

args="
--data data/reddit \
--tokenizer_vocab data/reddit/vocab/en-vocab.json \
--tokenizer_merges data/reddit/vocab/en-merges.txt \
--nlayers 8 \
--hid-sz 256 \
--inner-hid-sz 1024 \
--nheads 4 \
--attn-span 4096 \
--block-sz 256 \
--batch-sz 64 \
--optim radam \
--lr 3e-4 \
--momentum 0 \
--dropout .1 \
--lr-warmup 8000 \
--grad-clip 0.03 \
--niter 150 \
--nbatches 1000 \
--adapt-span \
--adapt-span-loss 0.000001 \
--adapt-span-cache
--batch-split 2 \
--checkpoint checkpoints/reddit_small.pt
"


echo "Launching ..."
# using the pytorch distributed launching
python3 reddit_interactive.py $args
