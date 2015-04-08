#!/usr/bin/env python

import argparse
import cPickle
import traceback
import logging
import time
import sys

import os
import numpy
import codecs
import search 

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

def sample(model, seqs=[[]], n_samples=1, sampler=None, beam_search=None, ignore_unk=False, normalize=False, alpha=1, verbose=False): 
    if beam_search:
        logger.info("Starting beam search : {} start sequences in total".format(len(seqs)))
         
        for idx, seq in enumerate(seqs):
            sentences = [] 
            
            seq = model.words_to_indices(seq, add_se=True)
            logger.info("Searching for {}".format(seq))
             
            gen_queries, gen_costs = beam_search.search(seq, n_samples, ignore_unk=ignore_unk) 
              
            for i in range(len(gen_queries)):
                query = model.indices_to_words(gen_queries[i])
                sentences.append(" ".join(query))

            for i in range(len(gen_costs)):   
                print "{}: {}".format(gen_costs[i], sentences[i].encode('utf-8'))
            
        return sentences, gen_costs, gen_queries 
    elif sampler:
        logger.info("Starting sampling")
        max_length = 40         
        print sampler(n_samples, max_length) 
    else:
        raise Exception("I don't know what to do")

def parse_args():
    parser = argparse.ArgumentParser("Sample (with beam-search) from the session model")
    
    parser.add_argument("--n-samples", 
            default="1", type=int, 
            help="Number of samples, if used with --beam-search, the size of the beam")
    
    parser.add_argument("--ignore-unk",
            default=True, action="store_true",
            help="Ignore unknown words")
    
    parser.add_argument("model_prefix",
            help="Path to the model prefix (without _model.npz or _state.pkl)")
    
    parser.add_argument("queries",
            help="File of input queries")

    parser.add_argument("--normalize",
            action="store_true", default=False,
            help="Normalize log-prob with the word count")
     
    parser.add_argument("--beam-search",
            action="store_true",
            help="Enable beam search")

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
    
    beam_search = None
    sampler = None

    if args.beam_search:
        beam_search = search.BeamSearch(model)
        beam_search.compile()
    else:
        sampler = model.build_sampling_function()
     
    seqs = [[]]
    if args.queries:
        lines = open(args.queries, "r").readlines()
        seqs = [x.strip().split() for x in lines]
     
    sample(model, seqs=seqs, ignore_unk=args.ignore_unk, sampler=sampler, beam_search=beam_search, n_samples=args.n_samples)

if __name__ == "__main__":
    main()
