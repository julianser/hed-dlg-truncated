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

def save(model, timings, post_fix = ''):
    print "Saving the model..."

    # ignore keyboard interrupt while saving
    start = time.time()
    s = signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    model.save(model.state['save_dir'] + '/' + model.state['run_id'] + "_" + model.state['prefix'] + post_fix + 'model.npz')
    cPickle.dump(model.state, open(model.state['save_dir'] + '/' +  model.state['run_id'] + "_" + model.state['prefix'] + post_fix + 'state.pkl', 'w'))
    numpy.savez(model.state['save_dir'] + '/' + model.state['run_id'] + "_" + model.state['prefix'] + post_fix + 'timing.npz', **timings)
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

    eval_batch = model.build_eval_function()

    random_sampler = search.RandomSampler(model)
    beam_sampler = search.BeamSampler(model) 

    logger.debug("Load data")
    train_data, \
    valid_data, = get_train_iterator(state)
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
    
    start_validation = False

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
        x_semantic = batch['x_semantic']
        x_reset = batch['x_reset']
        if state['use_nce']:
            y_neg = rng.choice(size=(10, max_length, x_data.shape[1]), a=model.idim, p=model.noise_probs).astype('int32')
            c = train_batch(x_data, x_data_reversed, y_neg, max_length, x_cost_mask, x_semantic, x_reset)
        else:
            c = train_batch(x_data, x_data_reversed, max_length, x_cost_mask, x_semantic, x_reset)

        if numpy.isinf(c) or numpy.isnan(c):
            logger.warn("Got NaN cost .. skipping")
            continue

        train_cost += c

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
                start_validation = True

        # Only start validation loop once it's time to validate and once all previous batches have been reset
        if start_validation and\
            numpy.sum(numpy.abs(x_reset)) < 1:
                start_validation = False
                valid_data.start()
                valid_cost = 0
                valid_wordpreds_done = 0
                valid_dialogues_done = 0


                # Prepare variables for plotting histogram over word-perplexities and mutual information
                valid_data_len = valid_data.data_len
                valid_cost_list = numpy.zeros((valid_data_len,))
                valid_pmi_list = numpy.zeros((valid_data_len,))

                # Prepare variables for printing the training examples the model performs best and worst on
                valid_extrema_setsize = min(state['track_extrema_samples_count'], valid_data_len)
                valid_extrema_samples_to_print = min(state['print_extrema_samples_count'], valid_extrema_setsize)

                max_stored_len = 160 # Maximum number of tokens to store for dialogues with highest and lowest validation errors
                valid_lowest_costs = numpy.ones((valid_extrema_setsize,))*1000
                valid_lowest_dialogues = numpy.ones((valid_extrema_setsize,max_stored_len))*1000
                valid_highest_costs = numpy.ones((valid_extrema_setsize,))*(-1000)
                valid_highest_dialogues = numpy.ones((valid_extrema_setsize,max_stored_len))*(-1000)

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

                    x_reset = batch['x_reset']

                    c, c_list = eval_batch(x_data, x_data_reversed, max_length, x_cost_mask, x_semantic, x_reset)

                    # Rehape into matrix, where rows are validation samples and columns are tokens
                    # Note that we use max_length-1 because we don't get a cost for the first token
                    # (the first token is always assumed to be eos)
                    c_list = c_list.reshape((batch['x'].shape[1],max_length-1), order=(1,0))
                    c_list = numpy.sum(c_list, axis=1)
                    
                    words_in_dialogues = numpy.sum(x_cost_mask, axis=0)
                    c_list = c_list / words_in_dialogues
                    

                    if numpy.isinf(c) or numpy.isnan(c):
                        continue
                    
                    valid_cost += c

                    valid_wordpreds_done += batch['num_preds']
                    valid_dialogues_done += batch['num_dialogues']


                logger.debug("[VALIDATION END]") 
                 
                valid_cost /= valid_wordpreds_done

                if len(timings["valid_cost"]) == 0 or valid_cost < numpy.min(timings["valid_cost"]):
                    patience = state['patience']
                    # Saving model if decrease in validation cost
                    save(model, timings)
                elif valid_cost >= timings["valid_cost"][-1] * state['cost_threshold']:
                    patience -= 1

                if args.save_every_valid_iteration:
                    save(model, timings, '_' + str(step) + '_')



                print "** valid cost (NLL) = %.4f, valid word-perplexity = %.4f, patience = %d" % (float(valid_cost), float(math.exp(valid_cost)), patience)

                timings["train_cost"].append(train_cost/train_done)
                timings["valid_cost"].append(valid_cost)

                # Reset train cost, train misclass and train done
                train_cost = 0
                train_done = 0

        step += 1

    logger.debug("All done, exiting...")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default="", help="Resume training from that state")
    parser.add_argument("--force_train_all_wordemb", action='store_true', help="If true, will force the model to train all word embeddings in the encoder. This switch can be used to fine-tune a model which was trained with fixed (pretrained)  encoder word embeddings.")
    parser.add_argument("--save_every_valid_iteration", action='store_true', help="If true, will save a copy of the model at every validation iteration.")

    parser.add_argument("--prototype", type=str, help="Use the prototype", default='prototype_state')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Models only run with float32
    assert(theano.config.floatX == 'float32')

    args = parse_args()
    main(args)
