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
        self.sdim = self.model.sdim

    def compile(self):
        logger.debug("Compiling beam search functions")
        self.next_probs_predictor = self.model.build_next_probs_function()
        self.compute_encoding = self.model.build_encoder_function()

    def search(self, seq, beam_size=1, ignore_unk=False, minlen=1, normalize_by_length=True, verbose=False):
        # Make seq a column vector
        seq = numpy.array(seq)
        
        if seq.ndim == 1:
            seq = numpy.array([seq], dtype='int32').T
        else:
            seq = seq.T

        assert seq.ndim == 2
        h, hs = self.compute_encoding(seq)
         
        # Initializing starting points with the last encoding of the sequence 
        prev_words = numpy.zeros((seq.shape[1],), dtype='int32') + self.eos_sym
        prev_hd = numpy.zeros((seq.shape[1], self.qdim), dtype='float32')
        prev_hs = numpy.zeros((seq.shape[1], self.sdim), dtype='float32')
         
        prev_hs[:] = hs[-1]
         
        fin_beam_gen = []
        fin_beam_costs = []

        beam_gen = [[]] 
        costs = [0.0]

        for k in range(100):
            if beam_size == 0:
                break
            
            if verbose:
                logger.info("Beam search at step %d" % k)
            
            prev_words = (numpy.array(map(lambda bg : bg[-1], beam_gen))
                    if k > 0
                    else numpy.zeros(1, dtype="int32") + self.eos_sym)
             
            prev_hd = prev_hd[:beam_size]
            prev_hs = prev_hs[:beam_size]

            assert prev_hs.shape[0] == prev_hd.shape[0]
            assert prev_words.shape[0] == prev_hs.shape[0]

            outputs, hd = self.next_probs_predictor(prev_hs, prev_words, prev_hd)
            log_probs = numpy.log(outputs)

            # Adjust log probs according to search restrictions
            if ignore_unk:
                log_probs[:, self.unk_sym] = -numpy.inf

            if k < minlen:
                log_probs[:, self.eos_sym] = -numpy.inf

            # Find the best options by calling argpartition of flatten array
            next_costs = numpy.array(costs)[:, None] - log_probs
            flat_next_costs = next_costs.flatten()
            best_costs_indices = argpartition(
                    flat_next_costs.flatten(),
                    beam_size)[:beam_size]

            # Decypher flatten indices
            voc_size = log_probs.shape[1]
            trans_indices = best_costs_indices / voc_size
            word_indices = best_costs_indices % voc_size
            costs = flat_next_costs[best_costs_indices]

            # Form a beam for the next iteration
            new_beam_gen = [[]] * beam_size 
            new_costs = numpy.zeros(beam_size)
            
            new_prev_hs = numpy.zeros((beam_size, self.sdim), dtype="float32")
            new_prev_hs[:] = hs[-1]
            new_prev_hd = numpy.zeros((beam_size, self.qdim), dtype="float32")
            
            for i, (orig_idx, next_word, next_cost) in enumerate(
                    zip(trans_indices, word_indices, costs)):
                
                new_beam_gen[i] = beam_gen[orig_idx] + [next_word]
                new_costs[i] = next_cost
                new_prev_hd[i] = hd[orig_idx]
             
            beam_gen = []
            costs = []
            indices = []

            for i in range(beam_size):
                # We finished sampling?
                if new_beam_gen[i][-1] != self.eos_sym:
                    beam_gen.append(new_beam_gen[i])
                    costs.append(new_costs[i])
                    indices.append(i)
                else:
                    beam_size -= 1
                    # Add without start and end-of-sentence
                    fin_beam_gen.append(new_beam_gen[i][1:-1])
                    if normalize_by_length:
                        fin_beam_costs.append(new_costs[i]/len(new_beam_gen[i]))

            # Filter out the finished states 
            prev_hd = new_prev_hd[indices]
            prev_hs = new_prev_hs[indices]
         
        fin_beam_gen = numpy.array(fin_beam_gen)[numpy.argsort(fin_beam_costs)]
        fin_beam_costs = numpy.array(sorted(fin_beam_costs))
         
        return fin_beam_gen, fin_beam_costs

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

            samples, costs = self.beam_search.search(joined_context, n_samples, ignore_unk=ignore_unk)
            # Convert back indices to list of words
            converted_samples = map(self.model.indices_to_words, samples)
            # Join the list of words
            converted_samples = map(' '.join, converted_samples)

            if verbose:
                for i in range(len(converted_samples)):
                    print "{}: {}".format(costs[i], converted_samples[i].encode('utf-8'))
            
            context_samples.append(converted_samples)
            context_costs.append(costs)
        
        return context_samples, context_costs
