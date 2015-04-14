"""
Compute n-letter-gram Jaccard similarity.
"""
__docformat__ = 'restructedtext en'
__authors__ = ("Alessandro Sordoni")
__contact__ = "Alessandro Sordoni <sordonia@iro.umontreal>"

import sys
import math
import copy
import re
import operator
import collections
import numpy

def count_letter_ngram(sentence, n=3):
    local_counts = set()
    for k in range(len(sentence.strip()) - n + 1): 
        local_counts.add(sentence[k:k+n])
    return local_counts

class Jaccard:
    """
    Jaccard n-letter-gram similarity.
    Use: 
    >>> j = Jaccard()
    >>> j.update("i have it", "i have is")
    >>> print j.compute()
    0.75
    >>> j.reset()
    """
    def __init__(self, n=3):
        self.n = n
        self.statistics = []
       	
    def aggregate(self):
        if len(self.statistics) == 0:
            return numpy.zeros((1,)) 	
        stat_matrix = numpy.array(self.statistics)
        return numpy.sum(stat_matrix, axis=0)

    def update(self, candidate, ref):
        stats = numpy.zeros((1,))	
         
        cand_ngrams = count_letter_ngram(candidate, self.n)
        ref_ngrams = count_letter_ngram(ref, self.n)
        stats[0] = float(len(cand_ngrams & ref_ngrams)) / len(cand_ngrams | ref_ngrams)
        self.statistics.append(stats)
	
    def compute(self):
        stats = self.aggregate()
        return stats[0]
	
    def reset(self):
        self.statistics = []
