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


def add_random_variables_to_batch(state, rng, batch, prev_batch, evaluate_mode):
    """
    This is a helper function, which adds random variables to a batch.
    We do it this way, because we want to avoid Theano's random sampling both to speed up and to avoid
    known Theano issues with sampling inside scan loops.

    The random variable 'ran_var_constutterance' is sampled from a standard Gaussian distribution, 
    which remains constant during each utterance (i.e. between end-of-utterance tokens).
    
    When not in evaluate mode, the random vector 'ran_decoder_drop_mask' is also sampled. 
    This variable represents the input tokens which are replaced by unk when given to 
    the decoder RNN. It is required for the noise addition trick used by Bowman et al. (2015).
    """

    # If none return none
    if not batch:
        return batch

    # Variable to store random vector sampled at the beginning of each utterance
    Ran_Var_ConstUtterance = numpy.zeros((batch['x'].shape[0], batch['x'].shape[1], state['latent_gaussian_per_utterance_dim']), dtype='float32')

    # Go through each sample, find end-of-utterance indices and sample random variables
    for idx in xrange(batch['x'].shape[1]):
        # Find end-of-utterance indices
        eos_indices = numpy.where(batch['x'][:, idx] == state['eos_sym'])[0].tolist()

        # Make sure we also sample at the beginning of the utterance, and that we stop appropriately at the end
        if len(eos_indices) > 0:
            if not eos_indices[0] == 0:
                eos_indices = [0] + eos_indices
            if not eos_indices[-1] == batch['x'].shape[0]:
                eos_indices = eos_indices + [batch['x'].shape[0]]
        else:
            eos_indices = [0] + [batch['x'].shape[0]]

        # Sample random variables using NumPy
        ran_vectors = rng.normal(loc=0, scale=1, size=(len(eos_indices), state['latent_gaussian_per_utterance_dim']))
        for i in range(len(eos_indices)-1):
            for j in range(eos_indices[i], eos_indices[i+1]):
                Ran_Var_ConstUtterance[j, idx, :] = ran_vectors[i, :]

        # If a previous batch is given, and the last utterance in the previous batch
        # overlaps with the first utterance in the current batch, then we need to copy over 
        # the random variables from the last utterance in the last batch to remain consistent.
        if prev_batch:
            if ('x_reset' in prev_batch) and (not numpy.sum(numpy.abs(prev_batch['x_reset'])) < 1) \
              and ('ran_var_constutterance' in prev_batch):
                prev_ran_vector = prev_batch['ran_var_constutterance'][-1,idx,:]
                if len(eos_indices) > 1:
                    for j in range(0, eos_indices[1]):
                        Ran_Var_ConstUtterance[j, idx, :] = prev_ran_vector
                else:
                    for j in range(0, batch['x'].shape[0]):
                        Ran_Var_ConstUtterance[j, idx, :] = prev_ran_vector

    # Add new random Gaussian variable to batch
    batch['ran_var_constutterance'] = Ran_Var_ConstUtterance

    # Create word drop mask based on 'decoder_drop_previous_input_tokens_rate' option:
    if evaluate_mode:
        batch['ran_decoder_drop_mask'] = numpy.ones((batch['x'].shape[0], batch['x'].shape[1]), dtype='float32')
    else:
        if state.get('decoder_drop_previous_input_tokens', False):
            ran_drop = rng.uniform(size=(batch['x'].shape[0], batch['x'].shape[1]))
            batch['ran_decoder_drop_mask'] = (ran_drop <= state['decoder_drop_previous_input_tokens_rate']).astype('float32')
        else:
            batch['ran_decoder_drop_mask'] = numpy.ones((batch['x'].shape[0], batch['x'].shape[1]), dtype='float32')


    return batch


def create_padded_batch(state, rng, x, force_end_of_utterance_token = False):
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

    # Fill X and Xmask.
    # Keep track of number of predictions and maximum dialogue length.
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

        # Keep track of longest dialogue
        max_length = max(max_length, dialogue_length)

        # Set the number of predictions == sum(Xmask), for cost purposes, minus one (to exclude first eos symbol)
        num_preds += dialogue_length - 1
        
        # Mark the end of phrase
        if len(x[0][idx]) < mx:
            if force_end_of_utterance_token:
                X[dialogue_length:, idx] = state['eos_sym']

        # Initialize Xmask column with ones in all positions that
        # were just set in X (except for first eos symbol, because we are not evaluating this). 
        # Note: if we need mask to depend on tokens inside X, then we need to 
        # create a corresponding mask for X_reversed and send it further in the model
        Xmask[0:dialogue_length, idx] = 1.

        # Reverse all utterances
        # TODO: For backward compatibility. This should be removed in future versions
        # i.e. move all the x_reversed computations to the model itself.
        eos_indices = numpy.where(X[:, idx] == state['eos_sym'])[0]
        X_reversed[:, idx] = X[:, idx]
        prev_eos_index = -1
        for eos_index in eos_indices:
            X_reversed[(prev_eos_index+1):eos_index, idx] = (X_reversed[(prev_eos_index+1):eos_index, idx])[::-1]
            prev_eos_index = eos_index
            if prev_eos_index > dialogue_length:
                break


    assert num_preds == numpy.sum(Xmask) - numpy.sum(Xmask[0, :])

    batch = {'x': X,                                                 \
             'x_reversed': X_reversed,                               \
             'x_mask': Xmask,                                        \
             'num_preds': num_preds,                                 \
             'num_dialogues': len(x[0]),                             \
             'max_length': max_length                                \
            }

    return batch

class Iterator(SSIterator):
    def __init__(self, dialogue_file, batch_size, **kwargs):
        SSIterator.__init__(self, dialogue_file, batch_size,                          \
                            seed=kwargs.pop('seed', 1234),                            \
                            max_len=kwargs.pop('max_len', -1),                        \
                            use_infinite_loop=kwargs.pop('use_infinite_loop', False))

        self.k_batches = kwargs.pop('sort_k_batches', 20)
        self.state = kwargs.pop('state', None)

        self.batch_iter = None
        self.rng = numpy.random.RandomState(self.state['seed'])

        # Keep track of previous batch, because this is needed to specify random variables
        self.prev_batch = None

        # Store whether the iterator operates in evaluate mode or not
        self.evaluate_mode = kwargs.pop('evaluate_mode', False)
        print 'Data Iterator Evaluate Mode: ', self.evaluate_mode

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
            for i in range(len(data)):
                data_x.append(data[i][0])

            x = numpy.asarray(list(itertools.chain(data_x)))

            lens = numpy.asarray([map(len, x)])
            order = numpy.argsort(lens.max(axis=0))
                 
            for k in range(number_of_batches):
                indices = order[k * batch_size:(k + 1) * batch_size]
                full_batch = create_padded_batch(self.state, self.rng, [x[indices]])

                # Then split batches to have size 'max_grad_steps'
                splits = int(math.ceil(float(full_batch['max_length']) / float(self.state['max_grad_steps'])))
                batches = []
                for i in range(0, splits):
                    batch = copy.deepcopy(full_batch)

                    # Retrieve start and end position (index) of current mini-batch
                    start_pos = self.state['max_grad_steps'] * i
                    if start_pos > 0:
                        start_pos = start_pos - 1

                    # We need to copy over the last token from each batch onto the next, 
                    # because this is what the model expects.
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
        # If there are no more batches in list, try to generate new batches
        if not self.batch_iter:
            self.batch_iter = self.get_homogenous_batch_iter(batch_size)

        try:
            # Retrieve next batch
            batch = next(self.batch_iter)

            # Add Gaussian random variables to batch. 
            # We add them separetly for each batch to save memory.
            # If we instead had added them to the full batch before splitting into mini-batches,
            # the random variables would take up several GBs for big batches and long documents.
            batch = add_random_variables_to_batch(self.state, self.rng, batch, self.prev_batch, self.evaluate_mode)
            # Keep track of last batch
            self.prev_batch = batch
        except StopIteration:
            return None
        return batch

def get_train_iterator(state):
    train_data = Iterator(
        state['train_dialogues'],
        int(state['bs']),
        state=state,
        seed=state['seed'],
        use_infinite_loop=True,
        max_len=-1,
        evaluate_mode=False)
     
    valid_data = Iterator(
        state['valid_dialogues'],
        int(state['bs']),
        state=state,
        seed=state['seed'],
        use_infinite_loop=False,
        max_len=-1,
        evaluate_mode=True)
    return train_data, valid_data 

def get_test_iterator(state):
    assert 'test_dialogues' in state
    test_path = state.get('test_dialogues')

    test_data = Iterator(
        test_path,
        int(state['bs']), 
        state=state,
        seed=state['seed'],
        use_infinite_loop=False,
        max_len=-1,
        evaluate_mode=True)
    return test_data
