import numpy as np
import theano
import theano.tensor as T
import sys, getopt
import logging

from state import *
from utils import *
from SS_dataset import *

import itertools
import sys
import pickle
import random
import datetime

logger = logging.getLogger(__name__)

def create_padded_batch(state, x):
    mx = state['seqlen']
    n = state['bs'] 
    
    X = numpy.zeros((mx, n), dtype='int32')
    Xmask = numpy.zeros((mx, n), dtype='float32') 

    # Variables to store last utterance (used to compute mutual information metric)
    X_last_utterance = numpy.zeros((mx, n), dtype='int32')
    Xmask_last_utterance = numpy.zeros((mx, n), dtype='float32') 
    X_start_of_last_utterance = numpy.zeros((n), dtype='int32') 


    # Fill X and Xmask
    # Keep track of number of predictions and maximum triple length
    num_preds = 0
    max_length = 0
    for idx in xrange(len(x[0])):
        # Insert sequence idx in a column of matrix X
        triple_length = len(x[0][idx])

        # Fiddle-it if it is too long ..
        if mx < triple_length: 
            continue

        X[:triple_length, idx] = x[0][idx][:triple_length]

        max_length = max(max_length, triple_length)

        # Set the number of predictions == sum(Xmask), for cost purposes
        num_preds += triple_length
        
        # Mark the end of phrase
        if len(x[0][idx]) < mx:
            X[triple_length:, idx] = state['eos_sym']

        # Initialize Xmask column with ones in all positions that
        # were just set in X
        Xmask[:triple_length, idx] = 1.

        # Find start of last utterance
        eos_indices = numpy.where(X[:, idx] == state['eos_sym'])[0]
        assert (len(eos_indices) > 2)
        start_of_last_utterance = eos_indices[1]+1
        X_start_of_last_utterance[idx] = start_of_last_utterance
        X_last_utterance[0:(triple_length-start_of_last_utterance), idx] = X[start_of_last_utterance:triple_length, idx]
        Xmask_last_utterance[0:(triple_length-start_of_last_utterance), idx] = Xmask[start_of_last_utterance:triple_length, idx]
     




     
    assert num_preds == numpy.sum(Xmask)
    return {'x': X, 'x_mask': Xmask, 'x_last_utterance': X_last_utterance, 'x_mask_last_utterance': Xmask_last_utterance, 'x_start_of_last_utterance': X_start_of_last_utterance, 'num_preds': num_preds, 'max_length': max_length}

def get_batch_iterator(rng, state):
    class Iterator(SSIterator):
        def __init__(self, *args, **kwargs):
            SSIterator.__init__(self, rng, *args, **kwargs)
            self.batch_iter = None
    
        def get_homogenous_batch_iter(self, batch_size = -1):
            while True:
                k_batches = state['sort_k_batches']
                batch_size = self.batch_size if (batch_size == -1) else batch_size 
               
                data = []
                for k in range(k_batches):
                    batch = SSIterator.next(self)
                    if batch:
                        data.append(batch)
                
                if not len(data):
                    return
                
                x = numpy.asarray(list(itertools.chain(*data)))
                lens = numpy.asarray([map(len, x)])
                order = numpy.argsort(lens.max(axis=0))
                 
                for k in range(len(data)):
                    indices = order[k * batch_size:(k + 1) * batch_size]
                    batch = create_padded_batch(state, [x[indices]])
                    if batch:
                        yield batch
        
        def start(self):
            SSIterator.start(self)
            self.batch_iter = None

        def next(self, batch_size = -1):
            """ 
            We can specify a batch size,
            independent of the object initialization. 
            """
            if not self.batch_iter:
                self.batch_iter = self.get_homogenous_batch_iter(batch_size)
            try:
                batch = next(self.batch_iter)
            except StopIteration:
                return None
            return batch

    train_data = Iterator(
        batch_size=int(state['bs']),
        triple_file=state['train_triples'],
        queue_size=100,
        use_infinite_loop=True,
        max_len=state['seqlen']) 
     
    valid_data = Iterator(
        batch_size=int(state['bs']),
        triple_file=state['valid_triples'],
        use_infinite_loop=False,
        queue_size=100,
        max_len=state['seqlen'])
    
    test_data = Iterator(
        batch_size=int(state['bs']),
        triple_file=state['test_triples'],
        use_infinite_loop=False,
        queue_size=100,
        max_len=state['seqlen'])

    return train_data, valid_data, test_data
