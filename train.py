# -*- coding: utf-8 -*-
#!/usr/bin/env python

from data_iterator import *
from state import *
from dialog_encdec import *
from utils import *
from evaluation import *

import time
import traceback
import os.path
import sys
import argparse
import cPickle
import logging
import search
import pprint
import numpy
import collections
import signal
import math


import matplotlib
matplotlib.use('Agg')
import pylab


class Unbuffered:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)
logger = logging.getLogger(__name__)

### Unique RUN_ID for this execution
RUN_ID = str(time.time())

### Additional measures can be set here
measures = ["train_cost", "train_misclass", "valid_cost", "valid_misclass", "valid_emi", "valid_bleu_n_1", "valid_bleu_n_2", "valid_bleu_n_3", "valid_bleu_n_4", 'valid_jaccard', 'valid_recall_at_1', 'valid_recall_at_5', 'valid_mrr_at_5', 'tfidf_cs_at_1', 'tfidf_cs_at_5']

def init_timings():
    timings = {}
    for m in measures:
        timings[m] = []
    return timings

def save(model, timings):
    print "Saving the model..."

    # ignore keyboard interrupt while saving
    start = time.time()
    s = signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    model.save(model.state['save_dir'] + '/' + model.state['run_id'] + "_" + model.state['prefix'] + 'model.npz')
    cPickle.dump(model.state, open(model.state['save_dir'] + '/' +  model.state['run_id'] + "_" + model.state['prefix'] + 'state.pkl', 'w'))
    numpy.savez(model.state['save_dir'] + '/' + model.state['run_id'] + "_" + model.state['prefix'] + 'timing.npz', **timings)
    signal.signal(signal.SIGINT, s)
    
    print "Model saved, took {}".format(time.time() - start)

def load(model, filename):
    print "Loading the model..."

    # ignore keyboard interrupt while saving
    start = time.time()
    s = signal.signal(signal.SIGINT, signal.SIG_IGN)
    model.load(filename)
    signal.signal(signal.SIGINT, s)

    print "Model loaded, took {}".format(time.time() - start)

def main(args):     
    logging.basicConfig(level = logging.DEBUG,
                        format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s")
     
    state = eval(args.prototype)() 
    timings = init_timings() 
    
    
    if args.resume != "":
        logger.debug("Resuming %s" % args.resume)
        
        state_file = args.resume + '_state.pkl'
        timings_file = args.resume + '_timing.npz'
        
        if os.path.isfile(state_file) and os.path.isfile(timings_file):
            logger.debug("Loading previous state")
            
            state = cPickle.load(open(state_file, 'r'))
            timings = dict(numpy.load(open(timings_file, 'r')))
            for x, y in timings.items():
                timings[x] = list(y)
        else:
            raise Exception("Cannot resume, cannot find files!")

    logger.debug("State:\n{}".format(pprint.pformat(state)))
    logger.debug("Timings:\n{}".format(pprint.pformat(timings)))
 
    if args.force_train_all_wordemb == True:
        state['fix_pretrained_word_embeddings'] = False

    model = DialogEncoderDecoder(state)
    rng = model.rng 

    if args.resume != "":
        filename = args.resume + '_model.npz'
        if os.path.isfile(filename):
            logger.debug("Loading previous model")
            load(model, filename)
        else:
            raise Exception("Cannot resume, cannot find model file!")
        
        if 'run_id' not in model.state:
            raise Exception('Backward compatibility not ensured! (need run_id in state)')           

    else:
        # assign new run_id key
        model.state['run_id'] = RUN_ID

    logger.debug("Compile trainer")
    if not state["use_nce"]:
        logger.debug("Training with exact log-likelihood")
        train_batch = model.build_train_function()
    else:
        logger.debug("Training with noise contrastive estimation")
        train_batch = model.build_nce_function()

    if model.bootstrap_from_semantic_information:
        eval_semantic_batch = model.build_semantic_eval_function()

    eval_batch = model.build_eval_function()
    eval_misclass_batch = model.build_eval_misclassification_function()

    random_sampler = search.RandomSampler(model)
    beam_sampler = search.BeamSampler(model) 

    logger.debug("Load data")
    train_data, \
    valid_data, \
    test_data = get_batch_iterator(rng, state)
    
    train_data.start()
    
    # Build the data structures for Bleu evaluation
    if 'bleu_evaluation' in state:
        bleu_eval_n_1 = BleuEvaluator(n=1)
        bleu_eval_n_2 = BleuEvaluator(n=2)
        bleu_eval_n_3 = BleuEvaluator(n=3)
        bleu_eval_n_4 = BleuEvaluator(n=4)
        jaccard_eval = JaccardEvaluator()
        recall_at_1_eval = RecallEvaluator(n=1)
        recall_at_5_eval = RecallEvaluator(n=5)
        mrr_at_5_eval = MRREvaluator(n=5)
        tfidf_cs_at_1_eval = TFIDF_CS_Evaluator(model, train_data.data_len, 1)
        tfidf_cs_at_5_eval = TFIDF_CS_Evaluator(model, train_data.data_len, 5)

        samples = open(state['bleu_evaluation'], 'r').readlines() 
        n = state['bleu_context_length']
        
        contexts = []
        targets = []
        for x in samples:        
            sentences = x.strip().split('\t')
            assert len(sentences) > n
            contexts.append(sentences[:n])
            targets.append(sentences[n:])

    # Start looping through the dataset
    step = 0
    patience = state['patience'] 
    start_time = time.time()
     
    train_cost = 0
    train_misclass = 0
    train_done = 0
    ex_done = 0
     
    while (step < state['loop_iters'] and
            (time.time() - start_time)/60. < state['time_stop'] and
            patience >= 0):

        # Sample stuff
        if step % 200 == 0:
            for param in model.params:
                print "%s = %.4f" % (param.name, numpy.sum(param.get_value() ** 2) ** 0.5)

            samples, costs = random_sampler.sample([[]], n_samples=1, n_turns=3)
            print "Sampled : {}".format(samples[0])

        # Training phase
        batch = train_data.next() 

        # Train finished
        if not batch:
            # Restart training
            logger.debug("Got None...")
            break
        
        logger.debug("[TRAIN] - Got batch %d,%d" % (batch['x'].shape[1], batch['max_length']))
        
        x_data = batch['x']
        x_data_reversed = batch['x_reversed']
        max_length = batch['max_length']
        x_cost_mask = batch['x_mask']
        x_semantic = batch['x_semantic']

        if state['use_nce']:
            y_neg = rng.choice(size=(10, max_length, x_data.shape[1]), a=model.idim, p=model.noise_probs).astype('int32')
            c = train_batch(x_data, x_data_reversed, y_neg, max_length, x_cost_mask, x_semantic)
        else:
            c = train_batch(x_data, x_data_reversed, max_length, x_cost_mask, x_semantic)

        if numpy.isinf(c) or numpy.isnan(c):
            logger.warn("Got NaN cost .. skipping")
            continue

        train_cost += c

        # Compute word-error rate
        miscl = eval_misclass_batch(x_data, x_data_reversed, max_length, x_cost_mask)
        if numpy.isinf(c) or numpy.isnan(c):
            logger.warn("Got NaN misclassification .. skipping")
            continue

        train_misclass += miscl

        train_done += batch['num_preds']

        this_time = time.time()
        if step % state['train_freq'] == 0:
            elapsed = this_time - start_time
            h, m, s = ConvertTimedelta(this_time - start_time)
            print ".. %.2d:%.2d:%.2d %4d mb # %d bs %d maxl %d acc_cost = %.4f acc_word_perplexity = %.4f acc_mean_word_error = %.4f " % (h, m, s,\
                                                                 state['time_stop'] - (time.time() - start_time)/60.,\
                                                                 step, \
                                                                 batch['x'].shape[1], \
                                                                 batch['max_length'], \
                                                                 float(train_cost/train_done), \
                                                                 math.exp(float(train_cost/train_done)), \
                                                                 float(train_misclass)/float(train_done))

        if valid_data is not None and\
            step % state['valid_freq'] == 0 and step > 1:
                valid_data.start()
                valid_cost = 0
                valid_misclass = 0
                valid_empirical_mutual_information = 0
                if model.bootstrap_from_semantic_information:
                    valid_semantic_cost = 0
                    valid_semantic_misclass = 0

                valid_wordpreds_done = 0
                valid_triples_done = 0


                # Prepare variables for plotting histogram over word-perplexities and mutual information
                valid_data_len = valid_data.data_len
                valid_cost_list = numpy.zeros((valid_data_len,))
                valid_pmi_list = numpy.zeros((valid_data_len,))


                # Prepare variables for printing the training examples the model performs best and worst on
                valid_extrema_setsize = min(state['track_extrema_samples_count'], valid_data_len)
                valid_extrema_samples_to_print = min(state['print_extrema_samples_count'], valid_extrema_setsize)

                valid_lowest_costs = numpy.ones((valid_extrema_setsize,))*1000
                valid_lowest_triples = numpy.ones((valid_extrema_setsize,state['seqlen']))*1000
                valid_highest_costs = numpy.ones((valid_extrema_setsize,))*(-1000)
                valid_highest_triples = numpy.ones((valid_extrema_setsize,state['seqlen']))*(-1000)


                logger.debug("[VALIDATION START]") 
                
                while True:
                    batch = valid_data.next()
                    # Train finished
                    if not batch:
                        break
                     
                    logger.debug("[VALID] - Got batch %d,%d" % (batch['x'].shape[1], batch['max_length']))
        
                    x_data = batch['x']
                    x_data_reversed = batch['x_reversed']
                    max_length = batch['max_length']
                    x_cost_mask = batch['x_mask']
                    x_semantic = batch['x_semantic']
                    x_semantic_nonempty_indices = numpy.where(x_semantic >= 0)

                    c, c_list = eval_batch(x_data, x_data_reversed, max_length, x_cost_mask)

                    c_list = c_list.reshape((batch['x'].shape[1],max_length), order=(1,0))
                    c_list = numpy.sum(c_list, axis=1)
                    
                    words_in_triples = numpy.sum(x_cost_mask, axis=0)
                    c_list = c_list / words_in_triples
                    

                    if numpy.isinf(c) or numpy.isnan(c):
                        continue
                    
                    valid_cost += c

                    # Store validation costs in list
                    nxt =  min((valid_triples_done+batch['x'].shape[1]), valid_data_len)
                    triples_in_batch = nxt-valid_triples_done
                    valid_cost_list[(nxt-triples_in_batch):nxt] = numpy.exp(c_list[0:triples_in_batch])

                    # Store best and worst validation costs                    
                    con_costs = np.concatenate([valid_lowest_costs, c_list[0:triples_in_batch]])
                    con_triples = np.concatenate([valid_lowest_triples, x_data[:, 0:triples_in_batch].T], axis=0)
                    con_indices = con_costs.argsort()[0:valid_extrema_setsize][::1]
                    valid_lowest_costs = con_costs[con_indices]
                    valid_lowest_triples = con_triples[con_indices]

                    con_costs = np.concatenate([valid_highest_costs, c_list[0:triples_in_batch]])
                    con_triples = np.concatenate([valid_highest_triples, x_data[:, 0:triples_in_batch].T], axis=0)
                    con_indices = con_costs.argsort()[-valid_extrema_setsize:][::-1]
                    valid_highest_costs = con_costs[con_indices]
                    valid_highest_triples = con_triples[con_indices]

                    # Compute word-error rate
                    miscl = eval_misclass_batch(x_data, x_data_reversed, max_length, x_cost_mask)
                    if numpy.isinf(c) or numpy.isnan(c):
                        continue

                    valid_misclass += miscl

                    # Compute empirical mutual information
                    if state['compute_mutual_information'] == True:
                        # Compute marginal log-likelihood of last utterance in triple:
                        # We approximate it with the margina log-probabiltiy of the utterance being observed first in the triple
                        x_data_last_utterance = batch['x_last_utterance']
                        x_data_last_utterance_reversed = batch['x_last_utterance_reversed']
                        x_cost_mask_last_utterance = batch['x_mask_last_utterance']
                        x_start_of_last_utterance = batch['x_start_of_last_utterance']

                        marginal_last_utterance_loglikelihood, marginal_last_utterance_loglikelihood_list = eval_batch(x_data_last_utterance, x_data_last_utterance_reversed, max_length, x_cost_mask_last_utterance)

                        marginal_last_utterance_loglikelihood_list = marginal_last_utterance_loglikelihood_list.reshape((batch['x'].shape[1],max_length), order=(1,0))
                        marginal_last_utterance_loglikelihood_list = numpy.sum(marginal_last_utterance_loglikelihood_list, axis=1)
                        # If we wanted to normalize histogram plots by utterance length, we should enable this:
                        #words_in_last_utterance = numpy.sum(x_cost_mask_last_utterance, axis=0)
                        #marginal_last_utterance_loglikelihood_list = marginal_last_utterance_loglikelihood_list / words_in_last_utterance

                        # Compute marginal log-likelihood of first utterances in triple by masking the last utterance
                        x_cost_mask_first_utterances = numpy.copy(x_cost_mask)
                        for i in range(batch['x'].shape[1]):
                            x_cost_mask_first_utterances[x_start_of_last_utterance[i]:max_length, i] = 0

                        marginal_first_utterances_loglikelihood, marginal_first_utterances_loglikelihood_list = eval_batch(x_data, x_data_reversed, max_length, x_cost_mask_first_utterances)

                        marginal_first_utterances_loglikelihood_list = marginal_first_utterances_loglikelihood_list.reshape((batch['x'].shape[1],max_length), order=(1,0))
                        marginal_first_utterances_loglikelihood_list = numpy.sum(marginal_first_utterances_loglikelihood_list, axis=1)

                        # If we wanted to normalize histogram plots by utterance length, we should enable this:
                        #words_in_first_utterances = numpy.sum(x_cost_mask_first_utterances, axis=0)
                        #marginal_first_utterances_loglikelihood_list = marginal_first_utterances_loglikelihood_list / words_in_first_utterances

                        # Compute empirical mutual information and pointwise empirical mutual information
                        valid_empirical_mutual_information += -c + marginal_first_utterances_loglikelihood + marginal_last_utterance_loglikelihood
                        valid_pmi_list[(nxt-triples_in_batch):nxt] = (-c_list*words_in_triples + marginal_first_utterances_loglikelihood_list + marginal_last_utterance_loglikelihood_list)[0:triples_in_batch]

                    if model.bootstrap_from_semantic_information:
                        # Compute cross-entropy error on predicting the semantic class and retrieve predictions
                        sem_eval = eval_semantic_batch(x_data, x_data_reversed, max_length, x_cost_mask, x_semantic)

                        # Evaluate only non-empty triples (empty triples are created to fill 
                        #   the whole batch sometimes).
                        sem_cost = sem_eval[0][-1, :, :]
                        valid_semantic_cost += numpy.sum(sem_cost[x_semantic_nonempty_indices])

                        # Compute misclassified predictions on last timestep over all labels
                        sem_preds = sem_eval[1][-1, :, :]
                        sem_preds_misclass = len(numpy.where(((x_semantic-0.5)*(sem_preds-0.5))[x_semantic_nonempty_indices] < 0)[0])


                        valid_semantic_misclass += sem_preds_misclass


                    valid_wordpreds_done += batch['num_preds']

                    # TODO: Fix this. For the last validation / test batch, the last triples are empty but still counted here...
                    valid_triples_done += batch['x'].shape[1]


                logger.debug("[VALIDATION END]") 
                 
                valid_cost /= valid_wordpreds_done
                valid_misclass /= float(valid_wordpreds_done)
                valid_empirical_mutual_information /= float(valid_triples_done)

                if len(timings["valid_cost"]) == 0 or valid_cost < numpy.min(timings["valid_cost"]):
                    patience = state['patience']
                    # Saving model if decrease in validation cost
                    save(model, timings)
                elif valid_cost >= timings["valid_cost"][-1] * state['cost_threshold']:
                    patience -= 1

                if model.bootstrap_from_semantic_information:
                    valid_semantic_cost /= float(valid_triples_done)
                    valid_semantic_misclass /= float(valid_triples_done)
                    print "** valid semantic cost = %.4f, valid semantic misclass error = %.4f" % (float(valid_semantic_cost), float(valid_semantic_misclass))

                print "** valid cost (NLL) = %.4f, valid word-perplexity = %.4f, valid mean word-error = %.4f, valid emp. mutual information = %.4f, patience = %d" % (float(valid_cost), float(math.exp(valid_cost)), float(valid_misclass), valid_empirical_mutual_information, patience)

                timings["train_cost"].append(train_cost/train_done)
                timings["train_misclass"].append(float(train_misclass)/float(train_done))
                timings["valid_cost"].append(valid_cost)
                timings["valid_misclass"].append(valid_misclass)
                timings["valid_emi"].append(valid_empirical_mutual_information)

                # Reset train cost, train misclass and train done
                train_cost = 0
                train_misclass = 0
                train_done = 0

                # Plot histogram over validation costs
                try:
                    pylab.figure()
                    bins = range(0, 50, 1)
                    pylab.hist(valid_cost_list, normed=1, histtype='bar')
                    pylab.savefig(model.state['save_dir'] + '/' + model.state['run_id'] + "_" + model.state['prefix'] + 'Valid_WordPerplexities_'+ str(step) + '.png')
                except:
                    pass


                # Print 5 of 10% validation samples with highest log-likelihood
                if state['track_extrema_validation_samples']==True:
                    print " highest word log-likelihood valid samples: " 
                    np.random.shuffle(valid_lowest_triples)
                    for i in range(valid_extrema_samples_to_print):
                        print "      Sample: {}".format(" ".join(model.indices_to_words(numpy.ravel(valid_lowest_triples[i,:]))))

                    print " lowest word log-likelihood valid samples: " 
                    np.random.shuffle(valid_highest_triples)
                    for i in range(valid_extrema_samples_to_print):
                        print "      Sample: {}".format(" ".join(model.indices_to_words(numpy.ravel(valid_highest_triples[i,:]))))

                # Plot histogram over empirical pointwise mutual informations
                if state['compute_mutual_information'] == True:
                    try:
                        pylab.figure()
                        bins = range(0, 100, 1)
                        pylab.hist(valid_pmi_list, normed=1, histtype='bar')
                        pylab.savefig(model.state['save_dir'] + '/' + model.state['run_id'] + "_" + model.state['prefix'] + 'Valid_PMI_'+ str(step) + '.png')
                    except:
                        pass


        if 'bleu_evaluation' in state and \
            step % state['valid_freq'] == 0 and step > 1:
            # Compute samples with beam search
            logger.debug("Executing beam search to get targets for bleu, jaccard etc.")
            samples, costs = beam_sampler.sample(contexts, n_samples=5, ignore_unk=True)
            logger.debug("Finished beam search.")

            # Save beam search samples to file
            logger.debug("Saving beam search samples to file.")
            f = open(model.state['save_dir'] + '/' + model.state['run_id'] + "_" + model.state['prefix'] + 'BeamSamples', 'w')
            for ps in samples:
                for i in range(len(ps)):
                    f.write(ps[i] + '\t')
                f.write('\n')

            logger.debug("Finished saving beam search samples.")


            assert len(samples) == len(contexts)
            #print 'samples', samples
             
            # Bleu evaluation
            bleu_n_1 = bleu_eval_n_1.evaluate(samples, targets)
            print "** bleu score (n=1) = %.4f " % bleu_n_1[0] 
            timings["valid_bleu_n_1"].append(bleu_n_1[0])

            bleu_n_2 = bleu_eval_n_2.evaluate(samples, targets)
            print "** bleu score (n=2) = %.4f " % bleu_n_2[0] 
            timings["valid_bleu_n_2"].append(bleu_n_2[0])

            bleu_n_3 = bleu_eval_n_3.evaluate(samples, targets)
            print "** bleu score (n=3) = %.4f " % bleu_n_3[0] 
            timings["valid_bleu_n_3"].append(bleu_n_3[0])

            bleu_n_4 = bleu_eval_n_4.evaluate(samples, targets)
            print "** bleu score (n=4) = %.4f " % bleu_n_4[0] 
            timings["valid_bleu_n_4"].append(bleu_n_4[0])

            # Jaccard evaluation
            jaccard = jaccard_eval.evaluate(samples, targets)
            print "** jaccard score = %.4f " % jaccard
            timings["valid_jaccard"].append(jaccard)

            # Recall evaluation
            recall_at_1 = recall_at_1_eval.evaluate(samples, targets)
            print "** recall@1 score = %.4f " % recall_at_1
            timings["valid_recall_at_1"].append(recall_at_1)

            recall_at_5 = recall_at_5_eval.evaluate(samples, targets)
            print "** recall@5 score = %.4f " % recall_at_5
            timings["valid_recall_at_5"].append(recall_at_5)

            # MRR evaluation (equivalent to mean average precision)
            mrr_at_5 = mrr_at_5_eval.evaluate(samples, targets)
            print "** mrr@5 score = %.4f " % mrr_at_5
            timings["valid_mrr_at_5"].append(mrr_at_5)

            # TF-IDF cosine similarity evaluation
            tfidf_cs_at_1 = tfidf_cs_at_1_eval.evaluate(samples, targets)
            print "** tfidf-cs@1 score = %.4f " % tfidf_cs_at_1
            timings["tfidf_cs_at_1"].append(tfidf_cs_at_1)

            tfidf_cs_at_5 = tfidf_cs_at_5_eval.evaluate(samples, targets)
            print "** tfidf-cs@5 score = %.4f " % tfidf_cs_at_5
            timings["tfidf_cs_at_5"].append(tfidf_cs_at_5)

        step += 1

    logger.debug("All done, exiting...")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default="", help="Resume training from that state")
    parser.add_argument("--force_train_all_wordemb", action='store_true', help="If true, will force the model to train all word embeddings in the encoder. This switch can be used to fine-tune a model which was trained with fixed (pretrained)  encoder word embeddings.")

    parser.add_argument("--prototype", type=str, help="Use the prototype", default='prototype_state')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Models only run with float32
    assert(theano.config.floatX == 'float32')

    args = parse_args()
    main(args)
