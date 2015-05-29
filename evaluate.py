#!/usr/bin/env python
"""
Evaluation script.

For paper submissions, this script should normally be run with flags --exclude-sos --plot-graphs, and both with and without the flag --exclude-stop-words.
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

# List of all 77 English pronouns, all puntucation signs included in MovieTriples and other special tokens.
stopwords = "all another any anybody anyone anything both each each other either everybody everyone everything few he her hers herself him himself his I it its itself many me mine more most much myself neither no one nobody none nothing one one another other others ours ourselves several she some somebody someone something that their theirs them themselves these they this those us we what whatever which whichever who whoever whom whomever whose you your yours yourself yourselves . , ? ' - -- ! <unk> </s> <s>"

def parse_args():
    parser = argparse.ArgumentParser("Sample (with beam-search) from the session model")
    
    parser.add_argument("model_prefix",
            help="Path to the model prefix (without _model.npz or _state.pkl)")
    
    parser.add_argument("--test-path",
            type=str, help="File of test data")

    parser.add_argument("--exclude-sos", action="store_true",
                       help="Mask <s> from the cost computation")

    parser.add_argument("--plot-graphs", action="store_true",
                       help="Plots frequency graphs for word perplexity and pointwise mutual information")

    parser.add_argument("--exclude-stop-words", action="store_true",
                       help="Exclude stop words (English pronouns, puntucation signs and special tokens) from all metrics. These words make up approximate 48.37% of the training set, so removing them should focus the metrics on the topical content and ignore syntatic errors.")
  
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
    eval_misclass_batch = model.build_eval_misclassification_function()
    
    if args.test_path:
        state['test_triples'] = args.test_path

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
    
    # Variables to store test statistics
    test_cost = 0
    test_cost_first_utterances = 0
    test_cost_last_utterance_marginal = 0
    test_misclass = 0
    test_misclass_first_utterances = 0
    test_empirical_mutual_information = 0

    if model.bootstrap_from_semantic_information:
        test_semantic_cost = 0
        test_semantic_misclass = 0

    test_wordpreds_done = 0
    test_wordpreds_done_last_utterance = 0
    test_triples_done = 0

    # Prepare variables for plotting histogram over word-perplexities and mutual information
    test_data_len = test_data.data_len
    test_cost_list = numpy.zeros((test_data_len,))
    test_pmi_list = numpy.zeros((test_data_len,))


    # Prepare variables for printing the test examples the model performs best and worst on
    test_extrema_setsize = min(state['track_extrema_samples_count'], test_data_len)
    test_extrema_samples_to_print = min(state['print_extrema_samples_count'], test_extrema_setsize)

    test_lowest_costs = numpy.ones((test_extrema_setsize,))*1000
    test_lowest_triples = numpy.ones((test_extrema_setsize,state['seqlen']))*1000
    test_highest_costs = numpy.ones((test_extrema_setsize,))*(-1000)
    test_highest_triples = numpy.ones((test_extrema_setsize,state['seqlen']))*(-1000)

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
        x_semantic = batch['x_semantic']
        x_semantic_nonempty_indices = numpy.where(x_semantic >= 0)

        # Hack to get rid of start of sentence token.
        if args.exclude_sos and model.sos_sym != -1:
            x_cost_mask[x_data == model.sos_sym] = 0

        if args.exclude_stop_words:
            for word_index in stopwords_indices:
                x_cost_mask[x_data == word_index] = 0

        batch['num_preds'] = numpy.sum(x_cost_mask)

        c, c_list = eval_batch(x_data, x_data_reversed, max_length, x_cost_mask)
        
        c_list = c_list.reshape((batch['x'].shape[1],max_length), order=(1,0))
        c_list = numpy.sum(c_list, axis=1)
        
        words_in_triples = numpy.sum(x_cost_mask, axis=0)
        c_list = c_list / words_in_triples
        

        if numpy.isinf(c) or numpy.isnan(c):
            continue
        
        test_cost += c

        # Store test costs in list
        nxt =  min((test_triples_done+batch['x'].shape[1]), test_data_len)
        triples_in_batch = nxt-test_triples_done
        test_cost_list[(nxt-triples_in_batch):nxt] = numpy.exp(c_list[0:triples_in_batch])

        # Store best and worst test costs        
        con_costs = numpy.concatenate([test_lowest_costs, c_list[0:triples_in_batch]])
        con_triples = numpy.concatenate([test_lowest_triples, x_data[:, 0:triples_in_batch].T], axis=0)
        con_indices = con_costs.argsort()[0:test_extrema_setsize][::1]
        test_lowest_costs = con_costs[con_indices]
        test_lowest_triples = con_triples[con_indices]

        con_costs = numpy.concatenate([test_highest_costs, c_list[0:triples_in_batch]])
        con_triples = numpy.concatenate([test_highest_triples, x_data[:, 0:triples_in_batch].T], axis=0)
        con_indices = con_costs.argsort()[-test_extrema_setsize:][::-1]
        test_highest_costs = con_costs[con_indices]
        test_highest_triples = con_triples[con_indices]

        # Compute word-error rate
        miscl = eval_misclass_batch(x_data, x_data_reversed, max_length, x_cost_mask)
        if numpy.isinf(c) or numpy.isnan(c):
            continue

        test_misclass += miscl

        # Equations to compute empirical mutual information

        # Compute marginal log-likelihood of last utterance in triple:
        # We approximate it with the margina log-probabiltiy of the utterance being observed first in the triple
        x_data_last_utterance = batch['x_last_utterance']
        x_data_last_utterance_reversed = batch['x_last_utterance_reversed']
        x_cost_mask_last_utterance = batch['x_mask_last_utterance']
        x_start_of_last_utterance = batch['x_start_of_last_utterance']

        # Hack to get rid of start of sentence token.
        if args.exclude_sos and model.sos_sym != -1:
            x_cost_mask_last_utterance[x_data_last_utterance == model.sos_sym] = 0

        if args.exclude_stop_words:
            for word_index in stopwords_indices:
                x_cost_mask_last_utterance[x_data_last_utterance == word_index] = 0

        batch['num_preds_at_utterance'] = numpy.sum(x_cost_mask_last_utterance)

        marginal_last_utterance_loglikelihood, marginal_last_utterance_loglikelihood_list = eval_batch(x_data_last_utterance, x_data_last_utterance_reversed, max_length, x_cost_mask_last_utterance)

        marginal_last_utterance_loglikelihood_list = marginal_last_utterance_loglikelihood_list.reshape((batch['x'].shape[1],max_length), order=(1,0))
        marginal_last_utterance_loglikelihood_list = numpy.sum(marginal_last_utterance_loglikelihood_list, axis=1)

        # Compute marginal log-likelihood of first utterances in triple by masking the last utterance
        x_cost_mask_first_utterances = numpy.copy(x_cost_mask)
        for i in range(batch['x'].shape[1]):
            x_cost_mask_first_utterances[x_start_of_last_utterance[i]:max_length, i] = 0

        marginal_first_utterances_loglikelihood, marginal_first_utterances_loglikelihood_list = eval_batch(x_data, x_data_reversed, max_length, x_cost_mask_first_utterances)

        marginal_first_utterances_loglikelihood_list = marginal_first_utterances_loglikelihood_list.reshape((batch['x'].shape[1],max_length), order=(1,0))
        marginal_first_utterances_loglikelihood_list = numpy.sum(marginal_first_utterances_loglikelihood_list, axis=1)

        # Compute empirical mutual information and pointwise empirical mutual information
        test_empirical_mutual_information += -c + marginal_first_utterances_loglikelihood + marginal_last_utterance_loglikelihood
        test_pmi_list[(nxt-triples_in_batch):nxt] = (-c_list*words_in_triples + marginal_first_utterances_loglikelihood_list + marginal_last_utterance_loglikelihood_list)[0:triples_in_batch]

        # Store log P(U_1, U_2) cost computed during mutual information
        test_cost_first_utterances += marginal_first_utterances_loglikelihood

        # Store marginal log P(U_3)
        test_cost_last_utterance_marginal += marginal_last_utterance_loglikelihood


        # Compute word-error rate for first utterances
        miscl_first_utterances = eval_misclass_batch(x_data, x_data_reversed, max_length, x_cost_mask_first_utterances)
        test_misclass_first_utterances += miscl_first_utterances
        if numpy.isinf(c) or numpy.isnan(c):
            continue

        if model.bootstrap_from_semantic_information:
            # Compute cross-entropy error on predicting the semantic class and retrieve predictions
            sem_eval = eval_semantic_batch(x_data, x_data_reversed, max_length, x_cost_mask, x_semantic)

            # Evaluate only non-empty triples (empty triples are created to fill 
            #   the whole batch sometimes).
            sem_cost = sem_eval[0][-1, :, :]
            test_semantic_cost += numpy.sum(sem_cost[x_semantic_nonempty_indices])

            # Compute misclassified predictions on last timestep over all labels
            sem_preds = sem_eval[1][-1, :, :]
            sem_preds_misclass = len(numpy.where(((x_semantic-0.5)*(sem_preds-0.5))[x_semantic_nonempty_indices] < 0)[0])
            test_semantic_misclass += sem_preds_misclass


        test_wordpreds_done += batch['num_preds']
        test_wordpreds_done_last_utterance += batch['num_preds_at_utterance']
        test_triples_done += batch['num_triples']
     
    logger.debug("[TEST END]") 

    test_cost_last_utterance_marginal /= test_wordpreds_done_last_utterance
    test_cost_last_utterance = (test_cost - test_cost_first_utterances) / test_wordpreds_done_last_utterance
    test_cost /= test_wordpreds_done
    test_cost_first_utterances /= float(test_wordpreds_done - test_wordpreds_done_last_utterance)

    test_misclass_last_utterance = float(test_misclass - test_misclass_first_utterances) / float(test_wordpreds_done_last_utterance)
    test_misclass_first_utterances /= float(test_wordpreds_done - test_wordpreds_done_last_utterance)
    test_misclass /= float(test_wordpreds_done)
    test_empirical_mutual_information /= float(test_triples_done)

    if model.bootstrap_from_semantic_information:
        test_semantic_cost /= float(test_triples_done)
        test_semantic_misclass /= float(test_done_triples)
        print "** test semantic cost = %.4f, test semantic misclass error = %.4f" % (float(test_semantic_cost), float(test_semantic_misclass))

    print "** test cost (NLL) = %.4f, test word-perplexity = %.4f, test word-perplexity last utterance = %.4f, , test word-perplexity marginal last utterance = %.4f, test mean word-error = %.4f, test mean word-error last utterance = %.4f, test emp. mutual information = %.4f" % (float(test_cost), float(math.exp(test_cost)), float(math.exp(test_cost_last_utterance)), float(math.exp(test_cost_last_utterance_marginal)), float(test_misclass), float(test_misclass_last_utterance), test_empirical_mutual_information)

    # Plot histogram over test costs
    if args.plot_graphs:
        try:
            pylab.figure()
            bins = range(0, 50, 1)
            pylab.hist(test_cost_list, normed=1, histtype='bar')
            pylab.savefig(model.state['save_dir'] + '/' + model.state['run_id'] + "_" + model.state['prefix'] + 'Test_WordPerplexities.png')
        except:
            pass

    # Print 5 of 10% test samples with highest log-likelihood
    if args.plot_graphs:
        print " highest word log-likelihood test samples: " 
        numpy.random.shuffle(test_lowest_triples)
        for i in range(test_extrema_samples_to_print):
            print "      Sample: {}".format(" ".join(model.indices_to_words(numpy.ravel(test_lowest_triples[i,:]))))

        print " lowest word log-likelihood test samples: " 
        numpy.random.shuffle(test_highest_triples)
        for i in range(test_extrema_samples_to_print):
            print "      Sample: {}".format(" ".join(model.indices_to_words(numpy.ravel(test_highest_triples[i,:]))))


    # Plot histogram over empirical pointwise mutual informations
    if args.plot_graphs:
        try:
            pylab.figure()
            bins = range(0, 100, 1)
            pylab.hist(test_pmi_list, normed=1, histtype='bar')
            pylab.savefig(model.state['save_dir'] + '/' + model.state['run_id'] + "_" + model.state['prefix'] + 'Test_PMI.png')
        except:
            pass

    logger.debug("All done, exiting...")

if __name__ == "__main__":
    main()
