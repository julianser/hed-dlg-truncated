#!/usr/bin/env python
"""
Evaluation script.

For paper submissions, this script should normally be run with flags --plot-graphs, and both with and without the flag --exclude-stop-words.


Run example:

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,allow_gc=True,scan.allow_gc=False,nvcc.flags=-use_fast_math python evaluate.py --plot-graphs Output/1432724394.9_MovieScriptModel --document_ids Data/Test_Shuffled_Dataset_Labels.txt &> Test_Eval_Output.txt

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

    parser.add_argument("--plot-graphs", action="store_true",
                       help="Plots frequency graphs for word perplexity and pointwise mutual information")

    parser.add_argument("--exclude-stop-words", action="store_true",
                       help="Exclude stop words (English pronouns, puntucation signs and special tokens) from all metrics. These words make up approximate 48.37% of the training set, so removing them should focus the metrics on the topical content and ignore syntatic errors.")

    parser.add_argument("--document-ids",
            type=str, help="File containing document ids for each triple (one id per line, if there are multiple tabs the first entry will be taken as the doc id). If this is given the script will compute standard deviations across documents for all metrics.")

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
    test_cost_first_utterances = 0 # marginal negative log-likelihood of first two utterances
    test_cost_last_utterance_marginal = 0 # marginal (approximate) negative log-likelihood of last utterances
    test_misclass = 0 # misclassification error-rate
    test_misclass_first_utterances = 0 # misclassification error-rate of first two utterances
    test_empirical_mutual_information = 0  # empirical mutual information between first two utterances and third utterance, where the marginal P(U_3) is approximated by P(U_3, empty, empty).

    if model.bootstrap_from_semantic_information:
        test_semantic_cost = 0
        test_semantic_misclass = 0

    test_wordpreds_done = 0 # number of words in total
    test_wordpreds_done_last_utterance = 0 # number of words in last utterances
    test_triples_done = 0 # number of triples evaluated

    # Variables to compute negative log-likelihood and empirical mutual information per genre
    compute_genre_specific_metrics = False
    if hasattr(model, 'semantic_information_dim'):
        compute_genre_specific_metrics = True
        test_cost_per_genre = numpy.zeros((model.semantic_information_dim, 1), dtype='float32')
        test_mi_per_genre = numpy.zeros((model.semantic_information_dim, 1), dtype='float32')
        test_wordpreds_done_per_genre = numpy.zeros((model.semantic_information_dim, 1), dtype='float32')
        test_triples_done_per_genre = numpy.zeros((model.semantic_information_dim, 1), dtype='float32')

    # Number of triples in dataset
    test_data_len = test_data.data_len

    # Correspond to the same variables as above, but now for each triple.
    # e.g. test_cost_list is a numpy array with the negative log-likelihood for each triple in the test set
    test_cost_list = numpy.zeros((test_data_len,))
    test_pmi_list = numpy.zeros((test_data_len,))
    test_cost_last_utterance_marginal_list = numpy.zeros((test_data_len,))
    test_misclass_list = numpy.zeros((test_data_len,))
    test_misclass_last_utterance_list = numpy.zeros((test_data_len,))

    # Array containing number of words in each triple
    words_in_triples_list = numpy.zeros((test_data_len,))

    # Array containing number of words in last utterance of each triple
    words_in_last_utterance_list = numpy.zeros((test_data_len,))

    # Prepare variables for printing the test examples the model performs best and worst on
    test_extrema_setsize = min(state['track_extrema_samples_count'], test_data_len)
    test_extrema_samples_to_print = min(state['print_extrema_samples_count'], test_extrema_setsize)

    max_stored_len = 160 # Maximum number of tokens to store for dialogues with highest and lowest validation errors
    test_lowest_costs = numpy.ones((test_extrema_setsize,))*1000
    test_lowest_triples = numpy.ones((test_extrema_setsize,max_stored_len))*1000
    test_highest_costs = numpy.ones((test_extrema_setsize,))*(-1000)
    test_highest_triples = numpy.ones((test_extrema_setsize,max_stored_len))*(-1000)

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

        if args.exclude_stop_words:
            for word_index in stopwords_indices:
                x_cost_mask[x_data == word_index] = 0

        batch['num_preds'] = numpy.sum(x_cost_mask)

        c, c_list = eval_batch(x_data, x_data_reversed, max_length, x_cost_mask, x_semantic)
        
        c_list = c_list.reshape((batch['x'].shape[1],max_length), order=(1,0))
        c_list = numpy.sum(c_list, axis=1)
       

        # Compute genre specific stats...
        if compute_genre_specific_metrics:
            non_nan_entries = numpy.array(c_list >= 0, dtype=int)
            c_list[numpy.where(non_nan_entries==0)] = 0
            test_cost_per_genre += (numpy.asmatrix(non_nan_entries*c_list) * numpy.asmatrix(x_semantic)).T
            test_wordpreds_done_per_genre += (numpy.asmatrix(non_nan_entries*numpy.sum(x_cost_mask, axis=0)) * numpy.asmatrix(x_semantic)).T

        if numpy.isinf(c) or numpy.isnan(c):
            continue
        
        test_cost += c

        # Store test costs in list
        nxt =  min((test_triples_done+batch['x'].shape[1]), test_data_len)
        triples_in_batch = nxt-test_triples_done

        words_in_triples = numpy.sum(x_cost_mask, axis=0)
        words_in_triples_list[(nxt-triples_in_batch):nxt] = words_in_triples[0:triples_in_batch]

        # We don't need to normalzie by the number of words... not if we're computing standard deviations at least...
        test_cost_list[(nxt-triples_in_batch):nxt] = c_list[0:triples_in_batch]

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
        miscl, miscl_list = eval_misclass_batch(x_data, x_data_reversed, max_length, x_cost_mask, x_semantic)
        if numpy.isinf(c) or numpy.isnan(c):
            continue

        test_misclass += miscl

        # Store misclassification errors in list
        miscl_list = miscl_list.reshape((batch['x'].shape[1],max_length), order=(1,0))
        miscl_list = numpy.sum(miscl_list, axis=1)
        test_misclass_list[(nxt-triples_in_batch):nxt] = miscl_list[0:triples_in_batch]

        # Equations to compute empirical mutual information

        # Compute marginal log-likelihood of last utterance in triple:
        # We approximate it with the margina log-probabiltiy of the utterance being observed first in the triple
        x_data_last_utterance = batch['x_last_utterance']
        x_data_last_utterance_reversed = batch['x_last_utterance_reversed']
        x_cost_mask_last_utterance = batch['x_mask_last_utterance']
        x_start_of_last_utterance = batch['x_start_of_last_utterance']

        if args.exclude_stop_words:
            for word_index in stopwords_indices:
                x_cost_mask_last_utterance[x_data_last_utterance == word_index] = 0


        words_in_last_utterance = numpy.sum(x_cost_mask_last_utterance, axis=0)
        words_in_last_utterance_list[(nxt-triples_in_batch):nxt] = words_in_last_utterance[0:triples_in_batch]

        batch['num_preds_at_utterance'] = numpy.sum(x_cost_mask_last_utterance)

        marginal_last_utterance_loglikelihood, marginal_last_utterance_loglikelihood_list = eval_batch(x_data_last_utterance, x_data_last_utterance_reversed, max_length, x_cost_mask_last_utterance, x_semantic)

        marginal_last_utterance_loglikelihood_list = marginal_last_utterance_loglikelihood_list.reshape((batch['x'].shape[1],max_length), order=(1,0))
        marginal_last_utterance_loglikelihood_list = numpy.sum(marginal_last_utterance_loglikelihood_list, axis=1)
        test_cost_last_utterance_marginal_list[(nxt-triples_in_batch):nxt] = marginal_last_utterance_loglikelihood_list[0:triples_in_batch]

        # Compute marginal log-likelihood of first utterances in triple by masking the last utterance
        x_cost_mask_first_utterances = numpy.copy(x_cost_mask)
        for i in range(batch['x'].shape[1]):
            x_cost_mask_first_utterances[x_start_of_last_utterance[i]:max_length, i] = 0

        marginal_first_utterances_loglikelihood, marginal_first_utterances_loglikelihood_list = eval_batch(x_data, x_data_reversed, max_length, x_cost_mask_first_utterances, x_semantic)

        marginal_first_utterances_loglikelihood_list = marginal_first_utterances_loglikelihood_list.reshape((batch['x'].shape[1],max_length), order=(1,0))
        marginal_first_utterances_loglikelihood_list = numpy.sum(marginal_first_utterances_loglikelihood_list, axis=1)

        # Compute empirical mutual information and pointwise empirical mutual information
        test_empirical_mutual_information += -c + marginal_first_utterances_loglikelihood + marginal_last_utterance_loglikelihood
        test_pmi_list[(nxt-triples_in_batch):nxt] = (-c_list*words_in_triples + marginal_first_utterances_loglikelihood_list + marginal_last_utterance_loglikelihood_list)[0:triples_in_batch]

        # Compute genre specific stats...
        if compute_genre_specific_metrics:
            if triples_in_batch==batch['x'].shape[1]:
                mi_list = (-c_list*words_in_triples + marginal_first_utterances_loglikelihood_list + marginal_last_utterance_loglikelihood_list)[0:triples_in_batch]
                non_nan_entries = numpy.array(mi_list >= 0, dtype=int)*numpy.array(mi_list != numpy.nan, dtype=int)
                test_mi_per_genre += (numpy.asmatrix(non_nan_entries*mi_list) * numpy.asmatrix(x_semantic)).T
                test_triples_done_per_genre += numpy.reshape(numpy.sum(x_semantic, axis=0), test_triples_done_per_genre.shape)

        # Store log P(U_1, U_2) cost computed during mutual information
        test_cost_first_utterances += marginal_first_utterances_loglikelihood

        # Store marginal log P(U_3)
        test_cost_last_utterance_marginal += marginal_last_utterance_loglikelihood


        # Compute word-error rate for first utterances
        miscl_first_utterances, miscl_first_utterances_list = eval_misclass_batch(x_data, x_data_reversed, max_length, x_cost_mask_first_utterances, x_semantic)
        test_misclass_first_utterances += miscl_first_utterances
        if numpy.isinf(c) or numpy.isnan(c):
            continue

        # Store misclassification for last utterance
        miscl_first_utterances_list = miscl_first_utterances_list.reshape((batch['x'].shape[1],max_length), order=(1,0))
        miscl_first_utterances_list = numpy.sum(miscl_first_utterances_list, axis=1)

        miscl_last_utterance_list = miscl_list - miscl_first_utterances_list

        test_misclass_last_utterance_list[(nxt-triples_in_batch):nxt] = miscl_last_utterance_list[0:triples_in_batch]


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

    print "** test cost (NLL) = %.4f, test word-perplexity = %.4f, test word-perplexity last utterance = %.4f, test word-perplexity marginal last utterance = %.4f, test mean word-error = %.4f, test mean word-error last utterance = %.4f, test emp. mutual information = %.4f" % (float(test_cost), float(math.exp(test_cost)), float(math.exp(test_cost_last_utterance)), float(math.exp(test_cost_last_utterance_marginal)), float(test_misclass), float(test_misclass_last_utterance), test_empirical_mutual_information)

    if compute_genre_specific_metrics:
        print '** test perplexity per genre', numpy.exp(test_cost_per_genre/test_wordpreds_done_per_genre)
        print '** test_mi_per_genre', test_mi_per_genre

        print '** words per genre', test_wordpreds_done_per_genre




    # Plot histogram over test costs
    if args.plot_graphs:
        try:
            pylab.figure()
            bins = range(0, 50, 1)
            pylab.hist(numpy.exp(test_cost_list), normed=1, histtype='bar')
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

    # To estimate the standard deviations, we assume that triples across documents (movies) are independent.
    # We compute the mean metric for each document, and then the variance between documents.
    # We then use the between document variance to compute the:
    # Let m be a metric:
    # Var[m] = Var[1/(words in total) \sum_d \sum_i m_{di}]
    #        = Var[1/(words in total) \sum_d (words in doc d)/(words in doc d) \sum_i m_{di}]
    #        = \sum_d (words in doc d)^2/(words in total)^2 Var [ 1/(words in doc d) \sum_i ]
    #        = \sum_d (words in doc d)^2/(words in total)^2 sigma^2
    #
    # where sigma^2 is the variance computed for the means across documents.

    # negative log-likelihood for each document (movie)
    per_document_test_cost = numpy.zeros((len(unique_document_ids)), dtype='float32')
    # negative log-likelihood for last utterance for each document (movie)
    per_document_test_cost_last_utterance = numpy.zeros((len(unique_document_ids)), dtype='float32')
    # misclassification error for each document (movie)
    per_document_test_misclass = numpy.zeros((len(unique_document_ids)), dtype='float32')
    # misclassification error for last utterance for each document (movie)
    per_document_test_misclass_last_utterance = numpy.zeros((len(unique_document_ids)), dtype='float32')


    # Compute standard deviations based on means across documents (sigma^2 above)
    all_words_squared = 0 # \sum_d (words in doc d)^2
    all_words_in_last_utterance_squared = 0 # \sum_d (words in last utterance of doc d)^2
    for doc_id in range(len(unique_document_ids)):
        doc_indices = numpy.where(document_ids == unique_document_ids[doc_id])

        per_document_test_cost[doc_id] = numpy.sum(test_cost_list[doc_indices]) / numpy.sum(words_in_triples_list[doc_indices])
        per_document_test_cost_last_utterance[doc_id] = numpy.sum(test_cost_last_utterance_marginal_list[doc_indices]) / numpy.sum(words_in_last_utterance_list[doc_indices])

        per_document_test_misclass[doc_id] = numpy.sum(test_misclass_list[doc_indices]) / numpy.sum(words_in_triples_list[doc_indices])
        per_document_test_misclass_last_utterance[doc_id] = numpy.sum(test_misclass_last_utterance_list[doc_indices]) / numpy.sum(words_in_last_utterance_list[doc_indices])

        all_words_squared += float(numpy.sum(words_in_triples_list[doc_indices]))**2
        all_words_in_last_utterance_squared += float(numpy.sum(words_in_last_utterance_list[doc_indices]))**2

    # Sanity check that all documents are being used in the standard deviation calculations
    assert(numpy.sum(words_in_triples_list) == test_wordpreds_done)
    assert(numpy.sum(words_in_last_utterance_list) == test_wordpreds_done_last_utterance)

    # Compute final standard deviation equation and print the standard deviations
    per_document_test_cost_variance = numpy.var(per_document_test_cost) * float(all_words_squared) / float(test_wordpreds_done**2)
    per_document_test_cost_last_utterance_variance = numpy.var(per_document_test_cost_last_utterance) * float(all_words_in_last_utterance_squared) / float(test_wordpreds_done_last_utterance**2)
    per_document_test_misclass_variance = numpy.var(per_document_test_misclass) * float(all_words_squared) / float(test_wordpreds_done**2)
    per_document_test_misclass_last_utterance_variance = numpy.var(per_document_test_misclass_last_utterance) * float(all_words_in_last_utterance_squared) / float(test_wordpreds_done_last_utterance**2)

    print 'Standard deviations:'
    print "** test cost (NLL) = ", math.sqrt(per_document_test_cost_variance)
    print "** test perplexity (NLL) = ", math.sqrt((math.exp(per_document_test_cost_variance) - 1)*math.exp(2*test_cost+per_document_test_cost_variance))

    print "** test cost last utterance (NLL) = ", math.sqrt(per_document_test_cost_last_utterance_variance)
    print "** test perplexity last utterance  (NLL) = ", math.sqrt((math.exp(per_document_test_cost_last_utterance_variance) - 1)*math.exp(2*test_cost+per_document_test_cost_last_utterance_variance))

    print "** test word-error = ", math.sqrt(per_document_test_misclass_variance)
    print "** test last utterance word-error = ", math.sqrt(per_document_test_misclass_last_utterance_variance)

    logger.debug("All done, exiting...")

if __name__ == "__main__":
    main()
