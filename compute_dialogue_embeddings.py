#!/usr/bin/env python
"""
This script computes dialogue embeddings for dialogues found in a text file.
"""

#!/usr/bin/env python

import argparse
import cPickle
import traceback
import logging
import time
import sys
import math

import os
import numpy
import codecs
import search
import utils

from dialog_encdec import DialogEncoderDecoder
from numpy_compat import argpartition
from state import prototype_state

logger = logging.getLogger(__name__)

class Timer(object):
    def __init__(self):
        self.total = 0

    def start(self):
        self.start_time = time.time()

    def finish(self):
        self.total += time.time() - self.start_time

def parse_args():
    parser = argparse.ArgumentParser("Compute dialogue embeddings from model")

    parser.add_argument("model_prefix",
            help="Path to the model prefix (without _model.npz or _state.pkl)")

    parser.add_argument("dialogues",
            help="File of input dialogues (tab separated)")

    parser.add_argument("output",
            help="Output file")
    
    parser.add_argument("--verbose",
            action="store_true", default=False,
            help="Be verbose")

    parser.add_argument("--use-second-last-state",
            action="store_true", default=False,
            help="Outputs the second last dialogue encoder state instead of the last one")

    return parser.parse_args()

def compute_encodings(joined_contexts, model, model_compute_encoding, output_second_last_state = False):
    context = numpy.zeros((model.seqlen, len(joined_contexts)), dtype='int32')
    context_lengths = numpy.zeros(len(joined_contexts), dtype='int32')
    for idx in range(len(joined_contexts)):
        context_lengths[idx] = len(joined_contexts[idx])
        if context_lengths[idx] < model.seqlen:
            context[:context_lengths[idx], idx] = joined_contexts[idx]
        else:
            # If context is longer tha max context, truncate it and force the end-of-utterance token at the end
            context[:model.seqlen, idx] = joined_contexts[idx][0:model.seqlen]
            context[model.seqlen-1, idx] = model.eos_sym
            context_lengths[idx] = model.seqlen

    n_samples = len(joined_contexts)

    # Generate the reversed context
    reversed_context = numpy.copy(context)
    for idx in range(context.shape[1]):
        eos_indices = numpy.where(context[:, idx] == model.eos_sym)[0]
        prev_eos_index = -1
        for eos_index in eos_indices:
            reversed_context[(prev_eos_index+2):eos_index, idx] = (reversed_context[(prev_eos_index+2):eos_index, idx])[::-1]
            prev_eos_index = eos_index

    # Recompute hs only for those particular sentences
    # that met the end-of-sentence token

    encoder_states = model_compute_encoding(context, reversed_context, model.seqlen)
    hs = encoder_states[1]

    if output_second_last_state:
        second_last_hidden_state = numpy.zeros((hs.shape[1], hs.shape[2]), dtype='float64')
        for i in range(hs.shape[1]):
            second_last_hidden_state[i, :] = hs[context_lengths[i] - 1, i, :]
        return second_last_hidden_state
    else:
        return hs[-1, :, :]


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
    
    contexts = [[]]
    lines = open(args.dialogues, "r").readlines()
    if len(lines):
        contexts = [x.strip().split('\t') for x in lines]
   
    model_compute_encoding = model.build_encoder_function()
    dialogue_encodings = []

    # Start loop
    joined_contexts = []
    batch_index = 0
    batch_total = int(math.ceil(float(len(contexts)) / float(model.bs)))
    for context_id, context_sentences in enumerate(contexts):

        # Convert contextes into list of ids
        joined_context = []

        if len(context_sentences) == 0:
            joined_context = [model.eos_sym]
        else:
            for sentence in context_sentences:
                sentence_ids = model.words_to_indices(sentence.split())
                # Add sos and eos tokens
                joined_context += [model.sos_sym] + sentence_ids + [model.eos_sym]

        # HACK
        for i in range(0, 50):
            joined_context += [model.sos_sym] + [0] + [model.eos_sym]

        joined_contexts.append(joined_context)

        if len(joined_contexts) == model.bs:
            batch_index = batch_index + 1
            logger.debug("[COMPUTE] - Got batch %d / %d" % (batch_index, batch_total))
            encs = compute_encodings(joined_contexts, model, model_compute_encoding, args.use_second_last_state)
            for i in range(len(encs)):
                dialogue_encodings.append(encs[i])

            joined_contexts = []


    if len(joined_contexts) > 0:
        logger.debug("[COMPUTE] - Got batch %d / %d" % (batch_total, batch_total))
        encs = compute_encodings(joined_contexts, model, model_compute_encoding, args.use_second_last_state)
        for i in range(len(encs)):
            dialogue_encodings.append(encs[i])

    # Save encodings to disc
    cPickle.dump(dialogue_encodings, open(args.output + '.pkl', 'w'))

if __name__ == "__main__":
    main()

