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
import math
import copy

logger = logging.getLogger(__name__)

def create_padded_batch(state, x):
    # Find max length in batch
    mx = 0
    for idx in xrange(len(x[0])):
        mx = max(mx, len(x[0][idx]))

    # Take into account that sometimes we need to add the end-of-utterance symbol at the start
    mx += 1

    n = state['bs'] 
    
    X = numpy.zeros((mx, n), dtype='int32')
    Xmask = numpy.zeros((mx, n), dtype='float32') 

    # Variable to store each utterance in reverse form (for bidirectional RNNs)
    X_reversed = numpy.zeros((mx, n), dtype='int32')

    # Fill X and Xmask
    # Keep track of number of predictions and maximum dialogue length
    num_preds = 0
    max_length = 0
    for idx in xrange(len(x[0])):
        # Insert sequence idx in a column of matrix X
        dialogue_length = len(x[0][idx])

        # Fiddle-it if it is too long ..
        if mx < dialogue_length: 
            continue

        # Make sure end-of-utterance symbol is at beginning of dialogue.
        # This will force model to generate first utterance too
        if not x[0][idx][0] == state['eos_sym']:
            X[:dialogue_length+1, idx] = [state['eos_sym']] + x[0][idx][:dialogue_length]
            dialogue_length = dialogue_length + 1
        else:
            X[:dialogue_length, idx] = x[0][idx][:dialogue_length]

        max_length = max(max_length, dialogue_length)

        # Set the number of predictions == sum(Xmask), for cost purposes, minus one (to exclude first eos symbol)
        num_preds += dialogue_length - 1
        
        # Mark the end of phrase
        if len(x[0][idx]) < mx:
            X[dialogue_length:, idx] = state['eos_sym']

        # Initialize Xmask column with ones in all positions that
        # were just set in X (except for first eos symbol, because we are not evaluating this). 
        # Note: if we need mask to depend on tokens inside X, then we need to 
        # create a corresponding mask for X_reversed and send it further in the model
        Xmask[0:dialogue_length, idx] = 1.

        # Reverse all utterances
        eos_indices = numpy.where(X[:, idx] == state['eos_sym'])[0]
        X_reversed[:, idx] = X[:, idx]
        prev_eos_index = -1
        for eos_index in eos_indices:
            X_reversed[(prev_eos_index+1):eos_index, idx] = (X_reversed[(prev_eos_index+1):eos_index, idx])[::-1]
            prev_eos_index = eos_index
            if prev_eos_index > dialogue_length:
                break

    assert num_preds == numpy.sum(Xmask) - numpy.sum(Xmask[0, :])
    
    return {'x': X,                                                 \
            'x_reversed': X_reversed,                               \
            'x_mask': Xmask,                                        \
            'num_preds': num_preds,                                 \
            'num_dialogues': len(x[0]),                               \
            'max_length': max_length                                \
           }

class Iterator(SSIterator):
    def __init__(self, dialogue_file, batch_size, **kwargs):
        SSIterator.__init__(self, dialogue_file, batch_size,                   \
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

            # Split list of words from the dialogue index
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
                full_batch = create_padded_batch(self.state, [x[indices]])

                # Add semantic information to batch; take care to fill with -1 (=n/a) whenever the batch is filled with empty dialogues
                if 'semantic_information_dim' in self.state:
                    full_batch['x_semantic'] = - numpy.ones((self.state['bs'], self.state['semantic_information_dim'])).astype('int32')
                    full_batch['x_semantic'][0:len(indices), :] = numpy.asarray(list(itertools.chain(x_semantic[indices]))).astype('int32')
                else:
                    full_batch['x_semantic'] = None

                # Then split batches to have size self.state.max_grad_steps
                splits = int(math.ceil(float(full_batch['max_length']) / float(self.state['max_grad_steps'])))
                batches = []
                for i in range(0, splits):
                    batch = copy.deepcopy(full_batch)
                    start_pos = self.state['max_grad_steps'] * i
                    if start_pos > 0:
                        start_pos = start_pos - 1

                    end_pos = min(full_batch['max_length'], self.state['max_grad_steps'] * (i + 1))
                    batch['x'] = full_batch['x'][start_pos:end_pos, :]
                    batch['x_reversed'] = full_batch['x_reversed'][start_pos:end_pos, :]
                    batch['x_mask'] = full_batch['x_mask'][start_pos:end_pos, :]


                    batch['max_length'] = end_pos - start_pos
                    batch['num_preds'] = numpy.sum(batch['x_mask']) - numpy.sum(batch['x_mask'][0,:])
                    # For each batch we compute the number of dialogues as a fraction of the full batch,
                    # that way, when we add them together, we get the total number of dialogues.
                    batch['num_dialogues'] = float(full_batch['num_dialogues']) / float(splits)
                    batch['x_reset'] = numpy.ones(self.state['bs'], dtype='float32')

                    batches.append(batch)

                if len(batches) > 0:
                    batches[len(batches)-1]['x_reset'] = numpy.zeros(self.state['bs'], dtype='float32')

                for batch in batches:
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
        state['train_dialogues'],
        int(state['bs']),
        state=state,
        seed=state['seed'],
        semantic_file=semantic_train_path,
        use_infinite_loop=True,
        max_len=-1) 
     
    valid_data = Iterator(
        state['valid_dialogues'],
        int(state['bs']),
        state=state,
        seed=state['seed'],
        semantic_file=semantic_valid_path,
        use_infinite_loop=False,
        max_len=-1)
    return train_data, valid_data 

def get_test_iterator(state):
    assert 'test_dialogues' in state
    test_path = state.get('test_dialogues')
    semantic_test_path = state.get('test_semantic', None)

    test_data = Iterator(
        test_path,
        int(state['bs']), 
        state=state,
        seed=state['seed'],
        semantic_file=semantic_test_path,
        use_infinite_loop=False,
        max_len=-1)
    return test_data
