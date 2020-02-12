# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#!/usr/bin/env python3

import os
import torch


def _tokenize(text_path, tokenizer):
    """Tokenizes a text file."""
    if isinstance(text_path, list):
        ids = []
        for child_path in text_path:
            ids.extend(_tokenize(child_path, tokenizer).tolist())
        
        ids = torch.LongTensor(ids)

        return ids
    else:
        print('Tokenizing {}'.format(text_path))
        assert os.path.exists(text_path)

        # Assign to each token its identifier
        ids = []
        with open(text_path, 'r', encoding="utf8") as f:
            for line in f:
                enc = tokenizer.encode(line)
                ids.extend(enc.ids)

        ids = torch.LongTensor(ids)

        return ids


class RedditCorpus:
    def __init__(self, data_path=None, tokenizer_vocab_path=None, tokenizer_merges_path=None):
        from tokenizers import SentencePieceBPETokenizer

        if tokenizer_merges_path is None or tokenizer_vocab_path is None:
            raise ValueError('Tokenizer not found')
        
        self.tokenizer = SentencePieceBPETokenizer(
            tokenizer_vocab_path, tokenizer_merges_path
        )

        import math
        TOTAL_BATCHES = 7
        TRAIN_RANGE = max(1, math.floor(TOTAL_BATCHES * .7))
        VALID_RANGE = max(1, math.floor(TOTAL_BATCHES * .2))
        TEST_RANGE = max(1, math.floor(TOTAL_BATCHES * .1))

        start_ix = 0

        if data_path is not None:
            self.train = _tokenize(
                text_path=[
                    os.path.join(data_path, f'batch_{ix}.txt')
                    for ix in range(start_ix, start_ix+TRAIN_RANGE)
                ],
                tokenizer=self.tokenizer)
            start_ix += TRAIN_RANGE

            self.valid = _tokenize(
                text_path=[
                    os.path.join(data_path, f'batch_{ix}.txt')
                    for ix in range(start_ix, start_ix+VALID_RANGE)
                ],
                tokenizer=self.tokenizer)
            start_ix += VALID_RANGE

            self.test = _tokenize(
                text_path=[
                    os.path.join(data_path, f'batch_{ix}.txt')
                    for ix in range(start_ix, start_ix+TEST_RANGE)
                ],
                tokenizer=self.tokenizer)

    def state_dict(self):
        return {
            'train': self.train,
            'valid': self.valid,
            'test': self.test
        }

    def load_state_dict(self, state_dict):
        self.train = state_dict['train']
        self.valid = state_dict['valid']
        self.test = state_dict['test']

    @property
    def vocab_size(self):
        return self.tokenizer._tokenizer.get_vocab_size()


def _batchify(data_tensor, batch_size):
    nb_batches = data_tensor.size(0) // batch_size
    # trim away some tokens to make whole batches
    data_tensor = data_tensor.narrow(0, 0, nb_batches * batch_size)
    data_tensor = data_tensor.view(batch_size, -1).contiguous()
    return data_tensor


def _build_corpus(data_path, tokenizer_vocab_path, tokenizer_merges_path, env_params):
    # save the corpus to a file so that it's faster next time
    corpus_path = os.path.join(data_path, 'corpus.pt')
    if os.path.exists(corpus_path):
        print('Loading an existing corpus file from {}'.format(corpus_path))
        
        corpus = RedditCorpus(None, tokenizer_vocab_path, tokenizer_merges_path)
        corpus_checkpoint = torch.load(corpus_path)
        corpus.load_state_dict(corpus_checkpoint)
    else:
        print('Creating a corpus file at {}'.format(corpus_path))
        if env_params['distributed']:
            # only one process need to create a corpus file
            if env_params['rank'] == 0:
                corpus = RedditCorpus(data_path, tokenizer_vocab_path, tokenizer_merges_path)
                torch.save(corpus.state_dict(), corpus_path)
                # sync with other processes
                torch.distributed.broadcast(torch.zeros(1).cuda(), src=0)
            else:
                print('Waiting rank0 to create a corpus file.')
                # sync with rank0
                torch.distributed.broadcast(torch.zeros(1).cuda(), src=0)
                corpus = RedditCorpus(None, tokenizer_vocab_path, tokenizer_merges_path)
                corpus_checkpoint = torch.load(corpus_path)
                corpus.load_state_dict(corpus_checkpoint)
        else:
            corpus = RedditCorpus(data_path, tokenizer_vocab_path, tokenizer_merges_path)
            torch.save(corpus.state_dict(), corpus_path)
    return corpus

def _get_train_val_test_data(corpus, batch_size):
    return [
        _batchify(corpus.train, batch_size),
        _batchify(corpus.valid, batch_size),
        _batchify(corpus.test, batch_size)
    ]


def get_train_val_test_data(data_params, env_params, batch_size, device):
    corpus = _build_corpus(**data_params, env_params=env_params)
    data_params['vocab_size'] = corpus.vocab_size
    train_data, val_data, test_data = _get_train_val_test_data(
        corpus=corpus, batch_size=batch_size)

    if env_params['distributed']:
        # split the data into equal parts
        assert batch_size % env_params['world_size'] == 0
        device_batch_size = batch_size // env_params['world_size']
        slice_data = slice(
            device_batch_size * env_params['rank'],
            device_batch_size * (env_params['rank'] + 1))
        train_data = train_data[slice_data]
        val_data = val_data[slice_data]
        test_data = test_data[slice_data]

    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)
    return train_data, val_data, test_data
