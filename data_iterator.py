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

    # Variable to store each utterance in reverse form (for bidirectional RNNs)
    X_reversed = numpy.zeros((mx, n), dtype='int32')

    # Variables to store last utterance (for computing mutual information metric)
    X_last_utterance = numpy.zeros((mx, n), dtype='int32')
    X_last_utterance_reversed = numpy.zeros((mx, n), dtype='int32')
    Xmask_last_utterance = numpy.zeros((mx, n), dtype='float32')
    X_start_of_last_utterance = numpy.zeros((n), dtype='int32') 

    # Fill X and Xmask
    # Keep track of number of predictions and maximum triple length
    num_preds = 0
    num_preds_last_utterance = 0
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
        # were just set in X. 
        # Note: if we need mask to depend on tokens inside X, then we need to 
        # create a corresponding mask for X_reversed and send it further in the model
        Xmask[:triple_length, idx] = 1.

        # Reverse all utterances
        sos_indices = numpy.where(X[:, idx] == state['sos_sym'])[0]
        eos_indices = numpy.where(X[:, idx] == state['eos_sym'])[0]
        X_reversed[:triple_length, idx] = x[0][idx][:triple_length]
        prev_eos_index = -1
        for eos_index in eos_indices:
            X_reversed[(prev_eos_index+2):eos_index, idx] = (X_reversed[(prev_eos_index+2):eos_index, idx])[::-1]
            prev_eos_index = eos_index
            if prev_eos_index > triple_length:
                break

        # Find start of last utterance and store the utterance
        assert (len(eos_indices) >= len(sos_indices))

        if len(sos_indices) > 0: # Check that dialogue is not empty
            start_of_last_utterance = sos_indices[-1]
        else: # If it is empty, then we define last utterance to start at the beginning
            start_of_last_utterance = 0

        num_preds_last_utterance += triple_length - start_of_last_utterance

        X_start_of_last_utterance[idx] = start_of_last_utterance
        X_last_utterance[0:(triple_length-start_of_last_utterance), idx] = X[start_of_last_utterance:triple_length, idx]
        Xmask_last_utterance[0:(triple_length-start_of_last_utterance), idx] = Xmask[start_of_last_utterance:triple_length, idx]


        # Store also the last utterance in reverse
        X_last_utterance_reversed[0:(triple_length-start_of_last_utterance), idx] = numpy.copy(X_last_utterance[0:(triple_length-start_of_last_utterance), idx])
        X_last_utterance_reversed[1:(triple_length-start_of_last_utterance-1), idx] = (X_last_utterance_reversed[1:(triple_length-start_of_last_utterance-1), idx])[::-1]
     
    assert num_preds == numpy.sum(Xmask)
    
    return {'x': X,                                                 \
            'x_reversed': X_reversed,                               \
            'x_mask': Xmask,                                        \
            'x_last_utterance': X_last_utterance,                   \
            'x_last_utterance_reversed': X_last_utterance_reversed, \
            'x_mask_last_utterance': Xmask_last_utterance,          \
            'x_start_of_last_utterance': X_start_of_last_utterance, \
            'num_preds': num_preds,                                 \
            'num_preds_at_utterance': num_preds_last_utterance,    \
            'num_triples': len(x[0]),                               \
            'max_length': max_length                                \
           }

class Iterator(SSIterator):
    def __init__(self, triple_file, batch_size, **kwargs):
        SSIterator.__init__(self, triple_file, batch_size,                   \
                            semantic_file=kwargs.pop('semantic_file', None), \
                            max_len=kwargs.pop('max_len', -1),               \
                            use_infinite_loop=kwargs.pop('use_infinite_loop', False))
        # TODO: max_len should be handled here and SSIterator should zip semantic_data and 
        # data. 
        self.k_batches = kwargs.pop('sort_k_batches', 20)
        # TODO: For backward compatibility. This should be removed in future versions
        # i.e. remove all the x_reversed computations in the model itself.
        self.state = kwargs.pop('state', None)
        # ---------------- 
        self.batch_iter = None

    def get_homogenous_batch_iter(self, batch_size = -1):
        while True:
            batch_size = self.batch_size if (batch_size == -1) else batch_size 
           
            data = []
            for k in range(self.k_batches):
                batch = SSIterator.next(self)
                if batch:
                    data.append(batch)
            
            if not len(data):
                return
            
            number_of_batches = len(data)
            data = list(itertools.chain.from_iterable(data))

            # Split list of words from the triple index
            data_x = []
            data_semantic = []
            for i in range(len(data)):
                data_x.append(data[i][0])
                data_semantic.append(data[i][1])

            x = numpy.asarray(list(itertools.chain(data_x)))
            x_semantic = numpy.asarray(list(itertools.chain(data_semantic)))

            lens = numpy.asarray([map(len, x)])
            order = numpy.argsort(lens.max(axis=0))
                 
            for k in range(number_of_batches):
                indices = order[k * batch_size:(k + 1) * batch_size]
                batch = create_padded_batch(self.state, [x[indices]])

                # Add semantic information to batch; take care to fill with -1 (=n/a) whenever the batch is filled with empty triples
                if 'semantic_information_dim' in self.state:
                    batch['x_semantic'] = - numpy.ones((self.state['bs'], self.state['semantic_information_dim'])).astype('int32')
                    batch['x_semantic'][0:len(indices), :] = numpy.asarray(list(itertools.chain(x_semantic[indices]))).astype('int32')
                else:
                    batch['x_semantic'] = None

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

def get_train_iterator(state):
    semantic_train_path = None
    semantic_valid_path = None
    
    if 'train_semantic' in state:
        assert state['valid_semantic']
        semantic_train_path = state['train_semantic']
        semantic_valid_path = state['valid_semantic']
    
    train_data = Iterator(
        state['train_triples'],
        int(state['bs']),
        state=state,
        seed=state['seed'],
        semantic_file=semantic_train_path,
        use_infinite_loop=True,
        max_len=state['seqlen']) 
     
    valid_data = Iterator(
        state['valid_triples'],
        int(state['bs']),
        state=state,
        seed=state['seed'],
        semantic_file=semantic_valid_path,
        use_infinite_loop=False,
        max_len=state['seqlen'])
    return train_data, valid_data 

def get_test_iterator(state):
    assert 'test_triples' in state
    test_path = state.get('test_triples')
    semantic_test_path = state.get('test_semantic', None)

    test_data = Iterator(
        test_path,
        int(state['bs']), 
        state=state,
        seed=state['seed'],
        semantic_file=semantic_test_path,
        use_infinite_loop=False,
        max_len=state['seqlen'])
    return test_data
