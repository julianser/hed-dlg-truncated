"""
Takes as input a test file (pkl) and outputs two text files for testing.
The first is the test context file 'test_contexts.txt', which contains all the first n utterances.
The second is the test response (prediction) file 'test_responses.txt', which contains the remaining utterances.

The test context file can be given as input to the sample.py script to generate possible continuations of the dialogue. The resulting utterances can then be compared to the test prediction file.

Usage example:

    python create-text-file-for-tests.py <model> Test_SplitByDialogues.dialogues.pkl --utterances_to_predict 2

Usage example with truncated contexts:

    python create-text-file-for-tests.py <model> Test_SplitByDialogues.dialogues.pkl --utterances_to_predict 2 --max_words_in_context 300

NOTE: It's better to use the original dialogues in plain text for building the context/response pairs, since we can then avoid unknown tokens!

@author Iulian Vlad Serban
"""

import argparse
import cPickle
import traceback
import itertools
import logging
import time
import sys

import collections
import string
import os

from state import prototype_state

def indices_to_words(idx_to_str, seq):
    """
    Converts a list of words to a list
    of word ids. Use unk_sym if a word is not
    known.
    """
    r = ''
    for word_index in seq:
        if word_index > len(idx_to_str):
            raise ValueError('Word index is too large for the model vocabulary!')
        r += idx_to_str[word_index] + ' '

    return r.strip()

def parse_args():
    parser = argparse.ArgumentParser("Generate text file with test dialogues")
    
    parser.add_argument("model_prefix",
            help="Path to the model prefix (without _model.npz or _state.pkl)")

    parser.add_argument("test_file",
            help="Path to the test file (pickled list, with one dialogue per entry; or plain text file with one dialogue per line)")

    parser.add_argument("--utterances_to_predict",
            type=int, default=1,
            help="Number of utterances to predict")

    parser.add_argument("--max_words_in_context",
            type=int, default=-1,
            help="Number of words in context (if there are more, the beginning of the context will be truncated)")

    parser.add_argument("--leave_out_short_dialogues", action='store_true', help="If enabled, dialogues which have fewer than (1+utterances_to_predict) will be left out. This ensures that the model is always conditioning on some context.")



    return parser.parse_args()

def main():
    args = parse_args()

    # Load state file
    state = prototype_state()
    state_path = args.model_prefix + "_state.pkl"
    with open(state_path) as src:
        state.update(cPickle.load(src))

    # Load dictionary

    # Load dictionaries to convert str to idx and vice-versa
    raw_dict = cPickle.load(open(state['dictionary'], 'r'))

    str_to_idx = dict([(tok, tok_id) for tok, tok_id, _, _ in raw_dict])
    idx_to_str = dict([(tok_id, tok) for tok, tok_id, freq, _ in raw_dict])


    assert len(args.test_file) > 3
    test_contexts = ''
    test_responses = ''
    utterances_to_predict = args.utterances_to_predict
    assert args.utterances_to_predict > 0

    # Is it a pickle file? Then process using model dictionaries..
    if args.test_file[len(args.test_file)-4:len(args.test_file)] == '.pkl':
        test_dialogues = cPickle.load(open(args.test_file, 'r'))
        for test_dialogueid,test_dialogue in enumerate(test_dialogues):
            if test_dialogueid % 100 == 0:
                print 'test_dialogue', test_dialogueid

            utterances = []
            current_utterance = []
            for word in test_dialogue:
                current_utterance += [word]
                if word == state['eos_sym']:
                    utterances += [current_utterance]
                    current_utterance = []

            if args.leave_out_short_dialogues:
                if len(utterances) <= utterances_to_predict+1:
                    continue

            context_utterances = []
            prediction_utterances = []
            for utteranceid, utterance in enumerate(utterances):
                if utteranceid >= len(utterances) - utterances_to_predict:
                    prediction_utterances += utterance
                else:
                    context_utterances += utterance

            if args.max_words_in_context > 0:
                while len(context_utterances) > args.max_words_in_context:
                    del context_utterances[0]


            test_contexts += indices_to_words(idx_to_str, context_utterances) + '\n'
            test_responses += indices_to_words(idx_to_str, prediction_utterances) + '\n'

    else: # Assume it's a text file

        test_dialogues = [[]]
        lines = open(args.test_file, "r").readlines()
        if len(lines):
            test_dialogues = [x.strip() for x in lines]

        for test_dialogueid,test_dialogue in enumerate(test_dialogues):
            if test_dialogueid % 100 == 0:
                print 'test_dialogue', test_dialogueid

            utterances = []
            current_utterance = []
            for word in test_dialogue.split():
                current_utterance += [word]
                if word == state['end_sym_utterance']:
                    utterances += [current_utterance]
                    current_utterance = []

            if args.leave_out_short_dialogues:
                if len(utterances) <= utterances_to_predict+1:
                    continue

            context_utterances = []
            prediction_utterances = []
            for utteranceid, utterance in enumerate(utterances):
                if utteranceid >= len(utterances) - utterances_to_predict:
                    prediction_utterances += utterance
                else:
                    context_utterances += utterance

            if args.max_words_in_context > 0:
                while len(context_utterances) > args.max_words_in_context:
                    del context_utterances[0]


            test_contexts += ' '.join(context_utterances) + '\n'
            test_responses += ' '.join(prediction_utterances) + '\n'


    print('Writing to files...')
    f = open('test_contexts.txt','w')
    f.write(test_contexts)
    f.close()

    f = open('test_responses.txt','w')
    f.write(test_responses)
    f.close()

    print('All done!')

if __name__ == "__main__":
    main()

