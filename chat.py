#!/usr/bin/env python
__docformat__ = 'restructedtext en'
__authors__ = ("Iulian Serban, Alessandro Sordoni")
__contact__ = "Iulian Serban <julianserban@gmail.com>"

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

import theano


logger = logging.getLogger(__name__)

class Timer(object):
    def __init__(self):
        self.total = 0

    def start(self):
        self.start_time = time.time()

    def finish(self):
        self.total += time.time() - self.start_time

def sample(model, seqs=[[]], n_samples=1, sampler=None, ignore_unk=False): 
    if sampler:
        context_samples, context_costs = sampler.sample(seqs,
                                            n_samples=n_samples,
                                            n_turns=1,
                                            ignore_unk=ignore_unk,
                                            verbose=True)


        return context_samples
    else:
        raise Exception("I don't know what to do")

def remove_speaker_tokens(s):
    s = s.replace('<first_speaker> ', '')
    s = s.replace('<second_speaker> ', '')
    s = s.replace('<third_speaker> ', '')
    s = s.replace('<minor_speaker> ', '')
    s = s.replace('<voice_over> ', '')
    s = s.replace('<off_screen> ', '')

    return s

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

    #sampler = search.RandomSampler(model)
    sampler = search.BeamSampler(model)

    # Start chat loop
    utterances = collections.deque()
    
    while (True):
       var = raw_input("User - ")

       # Increase number of utterances. We just set it to zero for simplicity so that model has no memory. 
       # But it works fine if we increase this number
       while len(utterances) > 0:
           utterances.popleft()
         
       current_utterance = [ model.end_sym_utterance ] + ['<first_speaker>'] + var.split() + [ model.end_sym_utterance ]
       utterances.append(current_utterance)
         
       #TODO Sample a random reply. To spice it up, we could pick the longest reply or the reply with the fewest placeholders...
       seqs = list(itertools.chain(*utterances))

       #TODO Retrieve only replies which are generated for second speaker...
       sentences = sample(model, \
            seqs=[seqs], ignore_unk=args.ignore_unk, \
            sampler=sampler, n_samples=5)

       if len(sentences) == 0:
           raise ValueError("Generation error, no sentences were produced!")

       utterances.append(sentences[0][0].split())

       reply = sentences[0][0].encode('utf-8')
       print "AI - ", remove_speaker_tokens(reply)


if __name__ == "__main__":
    # Run with THEANO_FLAGS=mode=FAST_RUN,floatX=float32,allow_gc=True,scan.allow_gc=False,nvcc.flags=-use_fast_math python chat.py Model_Name

    # Models only run with float32
    assert(theano.config.floatX == 'float32')

    main()


