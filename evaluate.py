#!/usr/bin/env python
"""

OBSOLETE! This script is not supported right now!

Evaluation script.

For paper submissions, this script should normally be run with flags and both with and without the flag --exclude-stop-words.

Run example:

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,allow_gc=True,scan.allow_gc=False,nvcc.flags=-use_fast_math python evaluate.py Output/1432724394.9_MovieScriptModel &> Test_Eval_Output.txt

"""

import argparse
import cPickle
import traceback
import logging
import time
import sys

import os
import numpy
import codecs
import math

from dialog_encdec import DialogEncoderDecoder 
from numpy_compat import argpartition
from state import * 
from data_iterator import get_test_iterator

import matplotlib
matplotlib.use('Agg')
import pylab

logger = logging.getLogger(__name__)

# List of all 77 English pronouns, all puntucation signs included in Movie-Scriptolog and other special tokens.
stopwords = "all another any anybody anyone anything both each each other either everybody everyone everything few he her hers herself him himself his I it its itself many me mine more most much myself neither no one nobody none nothing one one another other others ours ourselves several she some somebody someone something that their theirs them themselves these they this those us we what whatever which whichever who whoever whom whomever whose you your yours yourself yourselves . , ? ' - -- ! <unk> </s> <s>"

def parse_args():
    parser = argparse.ArgumentParser("Sample (with beam-search) from the session model")
    
    parser.add_argument("model_prefix",
            help="Path to the model prefix (without _model.npz or _state.pkl)")
    
    parser.add_argument("--test-path",
            type=str, help="File of test data")

    parser.add_argument("--exclude-stop-words", action="store_true",
                       help="Exclude stop words (English pronouns, puntucation signs and special tokens) from all metrics. These words make up approximate 48.37% of the training set, so removing them should focus the metrics on the topical content and ignore syntatic errors.")

    parser.add_argument("--document-ids",
            type=str, help="File containing document ids for each triple (one id per line, if there are multiple tabs the first entry will be taken as the doc id). If this is given the script will compute standard deviations across documents for all metrics. CURRENTLY NOT IMPLEMENTED.")

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
        state['test_dialogues'] = args.test_path

    # Initialize list of stopwords to remove
    if args.exclude_stop_words:
        logger.debug("Initializing stop-word list")
        stopwords_lowercase = stopwords.lower().split(' ')
        stopwords_indices = []
        for word in stopwords_lowercase:
            if word in model.str_to_idx:
                stopwords_indices.append(model.str_to_idx[word])

    test_data = get_test_iterator(state)
    test_data.start()

    # Load document ids
    if args.document_ids:
        labels_file = open(args.document_ids, 'r')
        labels_text = labels_file.readlines()
        document_ids = numpy.zeros((len(labels_text)), dtype='int32')
        for i in range(len(labels_text)):
            document_ids[i] = int(labels_text[i].split('\t')[0])

        unique_document_ids = numpy.unique(document_ids)
        
        assert(test_data.data_len == document_ids.shape[0])

    else:
        print 'Warning no file with document ids given... standard deviations cannot be computed.'
        document_ids = numpy.zeros((test_data.data_len), dtype='int32')
        unique_document_ids = numpy.unique(document_ids)

    # Variables to store test statistics
    test_cost = 0 # negative log-likelihood
    test_wordpreds_done = 0 # number of words in total

    # Number of triples in dataset
    test_data_len = test_data.data_len

    max_stored_len = 160 # Maximum number of tokens to store for dialogues with highest and lowest validation errors

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
        reset_mask = batch['x_reset']
        ran_cost_utterance = batch['ran_var_constutterance']
        ran_decoder_drop_mask = batch['ran_decoder_drop_mask']

        if args.exclude_stop_words:
            for word_index in stopwords_indices:
                x_cost_mask[x_data == word_index] = 0

        batch['num_preds'] = numpy.sum(x_cost_mask)

        c, _, c_list, _, _  = eval_batch(x_data, x_data_reversed, max_length, x_cost_mask, reset_mask, ran_cost_utterance, ran_decoder_drop_mask)

        c_list = c_list.reshape((batch['x'].shape[1],max_length-1), order=(1,0))
        c_list = numpy.sum(c_list, axis=1)     

        if numpy.isinf(c) or numpy.isnan(c):
            continue
        
        test_cost += c

        words_in_triples = numpy.sum(x_cost_mask, axis=0)

        if numpy.isinf(c) or numpy.isnan(c):
            continue

        if numpy.isinf(c) or numpy.isnan(c):
            continue


        test_wordpreds_done += batch['num_preds']
     
    logger.debug("[TEST END]") 

    print 'test_wordpreds_done (number of words) ', test_wordpreds_done
    test_cost /= test_wordpreds_done

    print "** test cost (NLL) = %.4f, test word-perplexity = %.4f " % (float(test_cost), float(math.exp(test_cost)))  

    logger.debug("All done, exiting...")

if __name__ == "__main__":
    main()
