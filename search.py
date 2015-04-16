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
from state import prototype_state

logger = logging.getLogger(__name__)

class BeamSearch(object):
    def __init__(self, model):
        self.model = model
        state = self.model.state
        self.unk_sym = self.model.unk_sym
        self.eos_sym = self.model.eos_sym
        self.qdim = self.model.qdim
        self.compiled = False
        self.sdim = self.model.sdim

    def compile(self):
        logger.debug("Compiling beam search functions")
        
        self.next_probs_predictor = self.model.build_next_probs_function()
        self.compute_encoding = self.model.build_encoder_function()
        self.compiled = True

    def search(self, context, beam_size=1, ignore_unk=False, \
               min_length=1, max_length=100, normalize_by_length=True, verbose=False):
        if not self.compiled:
            self.compile()

        # Convert to column vector
        context = numpy.array(context, dtype='int32')[:, None]
        prev_hd = numpy.zeros((beam_size, self.qdim), dtype='float32')
        prev_hs = numpy.zeros((beam_size, self.sdim), dtype='float32')
         
        # Compute the context encoding and get
        # the last hierarchical state
        h, hs = self.compute_encoding(context)
        prev_hs[:] = hs[-1]
         
        fin_beam_gen = []
        fin_beam_costs = []
         
        beam_gen = [[] for i in range(beam_size)] 
        costs = [0.0 for i in range(beam_size)]

        for k in range(max_length):
            if len(fin_beam_gen) >= beam_size:
                break
             
            if verbose:
                logger.info("Beam search at step %d" % k)
             
            prev_words = (numpy.array(map(lambda bg : bg[-1], beam_gen))
                    if k > 0
                    else numpy.zeros(beam_size, dtype="int32") + self.eos_sym)

            outputs, hd = self.next_probs_predictor(prev_hs, prev_words, prev_hd)
            log_probs = numpy.log(outputs)
             
            # Adjust log probs according to search restrictions
            if ignore_unk:
                log_probs[:, self.unk_sym] = -numpy.inf 
            if k <= min_length:
                log_probs[:, self.eos_sym] = -numpy.inf

            next_costs = numpy.array(costs)[:, None] - log_probs
            
            # Pick only on the first line (for the beginning of sampling)
            # This will avoid duplicate <s> token.
            if k == 0:
                flat_next_costs = next_costs[:1, :].flatten()
            else:
                # Set the next cost to infinite for finished sentences (they will be replaced)
                # by other sentences in the beam
                indices = [i for i, bg in enumerate(beam_gen) if bg[-1] == self.eos_sym]
                next_costs[indices, :] = numpy.inf 
                flat_next_costs = next_costs.flatten()
             
            best_costs_indices = argpartition(
                    flat_next_costs.flatten(),
                    beam_size)[:beam_size]            
             

            # Decypher flatten indices
            voc_size = log_probs.shape[1]
            trans_indices = best_costs_indices / voc_size
            word_indices = best_costs_indices % voc_size 
            costs = flat_next_costs[best_costs_indices]
             
            new_beam_gen = [[] for i in range(beam_size)] 
            new_costs = numpy.zeros(beam_size)
            new_prev_hd = numpy.zeros((beam_size, self.qdim), dtype="float32")
            
            for i, (orig_idx, next_word, next_cost) in enumerate(
                        zip(trans_indices, word_indices, costs)):
                new_beam_gen[i] = beam_gen[orig_idx] + [next_word]
                new_costs[i] = next_cost
                new_prev_hd[i] = hd[orig_idx]
            
            # Save the previous hidden states
            prev_hd = new_prev_hd
            beam_gen = new_beam_gen 
            costs = new_costs 

            for i in range(beam_size):
                # We finished sampling?
                if beam_gen[i][-1] == self.eos_sym:
                    if verbose:
                        logger.debug("Adding sentence {} from beam {}".format(new_beam_gen[i], i))
                     
                    # Add without start and end-of-sentence
                    fin_beam_gen.append(beam_gen[i]) 
                    if normalize_by_length:
                        fin_beam_costs.append(costs[i]/len(beam_gen[i]))
        
        # If we have not sampled anything
        # then force include stuff
        if len(fin_beam_gen) == 0:
            fin_beam_gen = beam_gen
            fin_beam_costs = [costs[i]/len(beam_gen[i]) for i in range(len(costs))]

        # Here we could have more than beam_size samples.
        # This is because we allow to sample beam_size terms
        # even if one sentence in the beam has been terminated </s>
        fin_beam_gen = numpy.array(fin_beam_gen)[numpy.argsort(fin_beam_costs)]
        fin_beam_costs = numpy.array(sorted(fin_beam_costs))
        
        return fin_beam_gen[:beam_size], fin_beam_costs[:beam_size]

class Sampler(object):
    """
    A simple sampler based on beam search
    """
    def __init__(self, model):
        # Compile beam search
        self.model = model
        self.beam_search = BeamSearch(model)
        self.beam_search.compile()

    def sample(self, contexts, n_samples=1, ignore_unk=False, verbose=False):
        if verbose:
            logger.info("Starting beam search : {} start sequences in total".format(len(contexts)))

        context_samples = []
        context_costs = []

        # Start loop for each sentence
        for context_id, context_sentences in enumerate(contexts):
            if verbose:
                logger.info("Searching for {}".format(context_sentences))

            # Convert contextes into list of ids
            joined_context = []
            for sentence in context_sentences:
                sentence_ids = self.model.words_to_indices(sentence.split())
                # Add sos and eos tokens
                joined_context += [self.model.sos_sym] + sentence_ids + [self.model.eos_sym]
            
            samples, costs = self.beam_search.search(joined_context, n_samples, ignore_unk=ignore_unk, verbose=verbose)
            # Convert back indices to list of words
            converted_samples = map(self.model.indices_to_words, samples, exclude_start_end=True)
            # Join the list of words
            converted_samples = map(' '.join, converted_samples)

            if verbose:
                for i in range(len(converted_samples)):
                    print "{}: {}".format(costs[i], converted_samples[i].encode('utf-8'))

            context_samples.append(converted_samples)
            context_costs.append(costs)

        return context_samples, context_costs
