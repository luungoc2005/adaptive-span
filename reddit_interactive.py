import torch
import argparse
from os import path
from tokenizers import SentencePieceBPETokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--vocab_path', type=str, default='data/reddit/vocab')

args = parser.parse_args()

if __name__ == "__main__":
    pass