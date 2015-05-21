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

from dialog_encdec import DialogEncoderDecoder 
from numpy_compat import argpartition
from state import * 
from data_iterator import get_test_iterator

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("Sample (with beam-search) from the session model")
    
    parser.add_argument("model_prefix",
            help="Path to the model prefix (without _model.npz or _state.pkl)")
    
    parser.add_argument("--test-path",
            type=str, help="File of test data")

    parser.add_argument("--exclude-sos", action="store_true",
                       help="Mask <s> from the cost computation")
    
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
    
    eval_batch = model.build_eval_function()
    
    if args.test_path:
        state['test_triples'] = args.test_path
    
    test_data = get_test_iterator(state)
    test_data.start()
    
    test_cost = 0
    test_done = 0    
    logger.debug("[TEST START]") 
            
    while True:
        batch = test_data.next()
        # Train finished
        if not batch:
            break
         
        logger.debug("[TEST] - Got batch %d,%d" % (batch['x'].shape[1], batch['max_length']))

        x_data = batch['x']
        x_data_reversed = batch['x_reversed']
        max_length = batch['max_length']
        x_cost_mask = batch['x_mask']
                
        # Hack to get rid of start of sentence token.
        if args.exclude_sos and model.sos_sym != -1:
            x_cost_mask[x_data == model.sos_sym] = 0
            batch['num_preds'] = numpy.sum(x_cost_mask)
        
        c, _ = eval_batch(x_data, x_data_reversed, max_length, x_cost_mask)
        
        if numpy.isinf(c) or numpy.isnan(c):
            assert False
         
        test_cost += c
        test_done += batch['num_preds']
     
    logger.debug("[TEST END]") 
    
    test_cost /= float(test_done) 
    print "** test entr. = %.4f, perpl. = %.4f" % (float(test_cost), numpy.exp(test_cost))

if __name__ == "__main__":
    main()
