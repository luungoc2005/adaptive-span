import torch
from os import path
from tokenizers import SentencePieceBPETokenizer

from config import PARAMS_CONFIG

from models import TransformerSeq

from utils import (
    get_params,
    load_checkpoint
)

LEADING_TEXT = """
[P1] There's a link to a (much) higher res version on the right for anyone who doesn't believe it.
How many exposures does a picture like that take? [SEP]  [P0] Two, and some really cheesy software that removes all DR from your pictures and makes them look like video games. 
Seriously, the beauty in great photography is often the high contrast, the differences between darks and lights, the shadow details... this [SEP]  [P1] Holy crap - that picture's going to shave its head, lunge at a car weilding an umbrella, and laugh maniacally as its children are forcefully removed? Damn.
Upon further consideration, I don't really get your simile. [SEP]  [P0] No. I'm saying that Britney Spears is really easy pop music that often sounds enjoyable upon first listen, but quickly reveals itself to be utterly shallow and almost completely worthless. [SEP]  [P1] Um ... yeah. I got that. Why does no one get my sense of humor? Sarcasm really doesn't translate on reddit. 
What you meant was, this photography is like Britney: popular, but insubstantial. [SEP]  [DOC_SEP]
"""

def launch(env_params,
           model_params,
           adapt_span_params,
           optim_params,
           data_params,
           trainer_params):
    device = env_params['device']
    
    model = TransformerSeq(
        vocab_size=data_params['vocab_size'], **model_params,
        adapt_span_params=adapt_span_params)

    model = model.to(device)

    
    
if __name__ == '__main__':
    launch(**get_params(params_config=PARAMS_CONFIG))
