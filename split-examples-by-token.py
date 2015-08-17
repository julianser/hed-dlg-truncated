"""
Takes as input a binarized dialogue corpus, splits the examples by a certain token and shuffles it

Example run:

   python split-examples-by-token.py Training.dialogues.pkl 2 Training_SplitByDialogues.dialogues --join_last_two_examples

@author Iulian Vlad Serban
"""

import collections
import numpy
import operator
import os
import sys
import logging
import cPickle

from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('text2dict')

def safe_pickle(obj, filename):
    if os.path.isfile(filename):
        logger.info("Overwriting %s." % filename)
    else:
        logger.info("Saving to %s." % filename)
    
    with open(filename, 'wb') as f:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)

# Thanks to Emile on Stackoverflow:
# http://stackoverflow.com/questions/4322705/split-a-list-into-nested-lists-on-a-value

def _itersplit(l, splitters):
    current = []
    for item in l:
        if item in splitters:
            yield current
            current = []
        else:
            current.append(item)
    yield current

def magicsplit(l, *splitters):
    return [subl for subl in _itersplit(l, splitters) if subl]

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="Binarized dialogue corpus (pkl file)")
parser.add_argument("token_id", type=int, help="Token index to split examples by (e.g. to split by end-of-dialogue set this to 2)")
parser.add_argument("--join_last_two_examples",
            action="store_true", default=False,
            help="If on, will join the last two splits generated from each example. This is useful to handle empty or very short last samples")

parser.add_argument("output", type=str, help="Filename of processed binarized dialogue corpus (pkl file)")
args = parser.parse_args()

if not os.path.isfile(args.input):
    raise Exception("Input file not found!")

logger.info("Loading dialogue corpus")
data = cPickle.load(open(args.input, 'r'))
data_len = len(data)

logger.info('Corpus loaded... Data len is %d' % data_len)

logger.info("Splitting corpus examples by token id... ")
processed_binarized_corpus = []
for i in range(data_len):
    logger.info('    Example %d ' % i)
    new_examples = magicsplit(data[i], int(args.token_id))

    # If option is specified, we append the last new example to the second last one
    if args.join_last_two_examples and len(new_examples) > 1:
        new_examples[len(new_examples)-2] += new_examples[len(new_examples)-1]
        del new_examples[len(new_examples)-1]

    for new_example in new_examples:
        processed_binarized_corpus.append(new_example + [int(args.token_id)])

logger.info('New data len is %d' % len(processed_binarized_corpus))

logger.info("Reshuffling corpus.")
rng = numpy.random.RandomState(13248)
rng.shuffle(processed_binarized_corpus)

logger.info("Saving corpus.")
safe_pickle(processed_binarized_corpus, args.output + ".pkl")

logger.info("Corpus saved. All done!")
