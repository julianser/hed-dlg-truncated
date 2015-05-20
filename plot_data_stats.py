"""
This script will plot a word rank vs word frequency graph 
and a word triple rank versus word triple frequency graph
for a set of corpus samples and model generated samples.

This can be used to compare the word coverage of the
generated samples with the real corpus samples.


Run it with the following three parameters: 
   dataset_name: name of dataset to use in plot labels
   complete_dataset_filename: path to the complete corpus (tab-separated text file)
   generatedsamples_filename: path to the generated samples (tab-separated text file)

@author Iulian Vlad Serban
"""

import collections
import numpy
import operator
import os
import sys
import logging
import cPickle
import itertools
from collections import Counter

import cPickle
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("dataset_name", type=str, help="Name of dataset to use in plots")
parser.add_argument("complete_dataset_filename", type=str, help="Complete dataset filename (tab-separated text file)")
parser.add_argument("generatedsamples_filename", type=str, help="Generated dataset filename (tab-separated text file)")


args = parser.parse_args()

def getFrequenciesFromTextFile(filename):
	word_counter = Counter()

	for line in open(filename, 'r'):
	    s = [x for x in line.strip().split()]
	    word_counter.update(s) 

	total_freq = sum(word_counter.values())

	vocab_count = word_counter.most_common()

        print '400 most common words in ' + str(filename)
        print word_counter.most_common(400)


	# Add special tokens to the vocabulary
	vocab = {'<unk>': 0, '<s>': 1, '</s>': 2}
	for i, (word, count) in enumerate(vocab_count):
	    vocab[word] = i + 3


	freqs = collections.defaultdict(lambda: 0)
	df = collections.defaultdict(lambda: 0)

	for line, triple in enumerate(open(filename, 'r')):
	    triple_lst = []

	    utterances = triple.split('\t')
	    for i, utterance in enumerate(utterances):
		
		utterance_lst = []
		for word in utterance.strip().split():
		    word_id = vocab.get(word, 0)
		    utterance_lst.append(word_id)
		    freqs[word_id] += 1
		
		# Here, we filter out unknown triple text and empty triples
		# i.e. <s> </s> or <s> 0 </s>
		if utterance_lst != [0] and len(utterance_lst):
		    triple_lst.append([1] + utterance_lst + [2]) 
		    freqs[1] += 1
		    freqs[2] += 1
		    df[1] += 1
		    df[2] += 1

	    unique_word_indices = []
	    for i in range(len(triple_lst)):
	        for word_id in triple_lst[i]:
	            unique_word_indices.append(word_id)

	    unique_word_indices = set(unique_word_indices)
	    for word_id in unique_word_indices:
	        df[word_id] += 1

	return np.asarray(freqs.values(), dtype='float32'), np.asarray(df.values(), dtype='float32')


# Get frequencies for dataset
word_freq_data, doc_freq_data_short = getFrequenciesFromTextFile(args.complete_dataset_filename)
word_freq_data = np.sort(word_freq_data) / np.sum(word_freq_data)
doc_freq_data_short = doc_freq_data_short / np.sum(doc_freq_data_short)
doc_freq_data = np.zeros((word_freq_data.shape[0]))
doc_freq_data[0:doc_freq_data_short.shape[0]] = doc_freq_data_short[:]
doc_freq_data = np.sort(doc_freq_data)

# Get frequencies from samples
word_freq_generated, doc_freq_generated_short = getFrequenciesFromTextFile(args.generatedsamples_filename)
word_freq_generated = np.sort(word_freq_generated) / np.sum(word_freq_generated)
doc_freq_generated_short = doc_freq_generated_short / np.sum(doc_freq_generated_short)
doc_freq_generated = np.zeros((word_freq_data.shape[0]))
doc_freq_generated[0:doc_freq_generated_short.shape[0]] = doc_freq_generated_short[:]
doc_freq_generated = np.sort(doc_freq_generated)

# Plot word rank versus (normalized) word frequency
plt.plot(range(0,len(word_freq_data)), word_freq_data[::-1], '-', lw=2)
plt.plot(range(0,len(word_freq_generated)), word_freq_generated[::-1], '-', lw=2)
legend(['Data Samples', "Generated Samples"])
plt.title(args.dataset_name + ': Word rank versus word frequency')
plt.xlabel('Word rank')
plt.ylabel('Word frequency (normalized)')
plt.xscale('log')
plt.yscale('log')
plt.show()

# Plot word triple rank versus (normalized) word triple rank
plt.plot(range(0,len(doc_freq_data)), doc_freq_data[::-1], '-', lw=2)
plt.plot(range(0,len(doc_freq_generated)), doc_freq_generated[::-1], '-', lw=2)
legend(['Data Samples', "Generated Samples"])
plt.title(args.dataset_name + ': Word triple rank versus word triple frequency')
plt.xlabel('Word triple rank')
plt.ylabel('Word triple frequency (normalized)')
plt.xscale('log')
plt.show()

