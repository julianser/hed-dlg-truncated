#!/usr/bin/env python
__docformat__ = 'restructedtext en'
__authors__ = ("Julian Serban, Alessandro Sordoni")
__contact__ = "Julian Serban <julianserban@gmail.com>"

import argparse
import cPickle
import traceback
import itertools
import logging
import time
import sys
import search

import collections
import string
import os
import numpy
import codecs

import nltk
from random import randint

from dialog_encdec import DialogEncoderDecoder 
from numpy_compat import argpartition
from state import prototype_state

logger = logging.getLogger(__name__)

class Timer(object):
    def __init__(self):
        self.total = 0

    def start(self):
        self.start_time = time.time()

    def finish(self):
        self.total += time.time() - self.start_time

def sample(model, seqs=[[]], n_samples=1, beam_search=None, ignore_unk=False): 
    if beam_search:
        sentences = [] 
         
        seq = model.words_to_indices(seqs[0])
        gen_ids, gen_costs = beam_search.search(seq, n_samples, ignore_unk=ignore_unk) 
              
        for i in range(len(gen_ids)):
            sentence = model.indices_to_words(gen_ids[i])
            sentences.append(sentence)

        return sentences
    else:
        raise Exception("I don't know what to do")

def parse_args():
    parser = argparse.ArgumentParser("Sample (with beam-search) from the session model")
       
    parser.add_argument("--ignore-unk",
            default=True, action="store_true",
            help="Ignore unknown words")
    
    parser.add_argument("model_prefix",
            help="Path to the model prefix (without _model.npz or _state.pkl)")

    parser.add_argument("--normalize",
            action="store_true", default=False,
            help="Normalize log-prob with the word count")

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
    
    logger.info("This model uses " + model.decoder_bias_type + " bias type")

    beam_search = None
    sampler = None

    beam_search = search.BeamSearch(model)
    beam_search.compile()

    # Start chat loop    
    utterances = collections.deque()
    
    while (True):
       var = raw_input("User - ")

       while len(utterances) > 2:
           utterances.popleft()
         
       current_utterance = [ model.start_sym_sentence ] + var.split() + [ model.end_sym_sentence ]
       utterances.append(current_utterance)
         
       # Sample a random reply. To spicy it up, we could pick the longest reply or the reply with the fewest placeholders...
       seqs = list(itertools.chain(*utterances))

       sentences = sample(model, \
            seqs=[seqs], ignore_unk=args.ignore_unk, \
            beam_search=beam_search, n_samples=5)

       if len(sentences) == 0:
           raise ValueError("Generation error, no sentences were produced!")

       reply = " ".join(sentences[0]).encode('utf-8') 
       print "AI - ", reply
         
       utterances.append(sentences[0])

if __name__ == "__main__":
    # Run with THEANO_FLAGS=mode=FAST_RUN,floatX=float32,allow_gc=True,scan.allow_gc=False,nvcc.flags=-use_fast_math python chat.py Model_Name
    main()


