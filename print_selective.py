
#!/usr/bin/env python

import argparse
import cPickle
import traceback
import logging
import time
import sys

import string
import os
import numpy as np
import codecs

from dialog_encdec import DialogEncoderDecoder 
from numpy_compat import argpartition
from state import prototype_state

logger = logging.getLogger(__name__)

def get_selective_bias(model, seq, verbose=False):    
    get_selective_fn = model.build_grab_selective_function()
    seq = model.words_to_indices(seq.split(), add_se=True)
    
    x_max_length = len(seq)
    x_cost_mask = np.ones((x_max_length, 1), dtype='float32')
    seq = np.array([seq], dtype='int32').T
    
    cost, hs = get_selective_fn(seq, x_max_length, x_cost_mask)
    return cost, hs

def parse_args():
    parser = argparse.ArgumentParser("Sample (with beam-search) from the session model")
       
    parser.add_argument("--ignore-unk",
            default=True, action="store_true",
            help="Ignore unknown words")
    
    parser.add_argument("model_prefix",
            help="Path to the model prefix (without _model.npz or _state.pkl)")

    parser.add_argument("--verbose",
            action="store_true", default=False,
            help="Be verbose")
    
    parser.add_argument("changes", nargs="?", default="", help="Changes to state")
    return parser.parse_args()

def main():
    args = parse_args()
    state = prototype_state()
   
    state_path = args.model_prefix + "_state.pkl"
    model_path = args.model_prefix + "_model.npz"

    with open(state_path) as src:
        state.update(cPickle.load(src)) 
    
    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
     
    model = DialogEncoderDecoder(state)
    if os.path.isfile(model_path):
        logger.debug("Loading previous model")
        model.load(model_path)
    else:
        raise Exception("Must specify a valid model path")

    cost, hs = get_selective_bias(model, "<s> next week , pretty much any day x </s>"
                                         "<s> i 'm off all week next week x </s>"
                                         "<s> monday or thursday are probs best xx </s> <s> </t>")
    np.savez('hs', hs)

if __name__ == "__main__":
    # Run with THEANO_FLAGS=mode=FAST_RUN,floatX=float32,allow_gc=True,scan.allow_gc=False,nvcc.flags=-use_fast_math python chat.py Model_Name
    main()


