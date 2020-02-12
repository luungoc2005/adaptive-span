from os import path, listdir
from tokenizers import SentencePieceBPETokenizer, BPETokenizer

DATA_PATH = './data/reddit'
VOCAB_SIZE = 12000

tokenizer = SentencePieceBPETokenizer()

texts = [
    path.join(DATA_PATH, item)
    for item in listdir(DATA_PATH)
    if item.endswith('.txt')
]

tokenizer.train(texts, 
    vocab_size=VOCAB_SIZE, 
    min_frequency=10
)

SAVE_PATH = path.join(DATA_PATH, 'vocab')
if not path.isdir(SAVE_PATH):
    import os
    os.makedirs(SAVE_PATH)

tokenizer.save(SAVE_PATH, 'en')
