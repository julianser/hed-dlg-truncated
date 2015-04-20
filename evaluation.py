"""
Computes BLEU@n / Jaccard. 
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

def get_ref_length(ref_lens, candidate_len, method='closest'):
    if method == 'closest':
        len_diff = [(x, numpy.abs(x - candidate_len)) for x in ref_lens]
        min_len = sorted(len_diff, key=operator.itemgetter(1))[0][0] 
    elif method == 'shortest':
        min_len = min(ref_lens)
    elif method == 'average':
        min_len = float(sum(ref_lens))/len(ref_lens)
    return min_len 

def normalize(sentence):
    return sentence.strip().split()

def count_ngrams(sentences, n=4):
    global_counts = {} 
    for sentence in sentences:
        local_counts = {}
        list_len = len(sentence)	
         
        for k in xrange(1, n + 1):
            for i in range(list_len - k + 1):
                ngram = tuple(sentence[i:i+k])
                local_counts[ngram] = local_counts.get(ngram, 0) + 1
	    	
		### Store maximum occurrence; useful for multireference bleu
		for ngram, count in local_counts.items():
			global_counts[ngram] = max(global_counts.get(ngram, 0), count)
	return global_counts

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
        return numpy.mean(stat_matrix)

    def update(self, candidate, ref):
        stats = numpy.zeros((1,))	
         
        cand_ngrams = count_letter_ngram(candidate, self.n)
        ref_ngrams = count_letter_ngram(ref, self.n)
        stats[0] = float(len(cand_ngrams & ref_ngrams)) / len(cand_ngrams | ref_ngrams)
        self.statistics.append(stats)
	
    def compute(self):
        stats = self.aggregate()
        #return stats[0]
        return stats
	
    def reset(self):
        self.statistics = []

class JaccardEvaluator(object):
    """ Jaccard evaluator
    """
    def __init__(self):
        self.jaccard = Jaccard()

    def evaluate(self, prediction, target):
        if len(target) != len(prediction):
            raise ValueError('Target and predictions length mismatch!')
        
        # Assume ordered list and take only the first one
        if isinstance(prediction[0], list):
            prediction = [x[0] for x in prediction]
         
        self.jaccard.reset()
        for ts, ps in zip(target, prediction):
            self.jaccard.update(ps, *ts)
        return self.jaccard.compute()

class Bleu:
    """
    Bleu score. 
    Use: 
    >>> b = Bleu()
    >>> b.update("i have this", "i have this :)", "oh my my") # multi-references
    >>> b.compute()
    >>> b.reset()
    """
    def __init__(self, n=4):
        # Statistics are
        # - 1-gramcount,
        # - 2-gramcount,
        # - 3-gramcount,
        # - 4-gramcount,
        # - 1-grammatch,
        # - 2-grammatch,
        # - 3-grammatch,
        # - 4-grammatch,
        # - reflen
        self.n = n
        self.statistics = []
       	
    def aggregate(self):
        if len(self.statistics) == 0:
            return numpy.zeros((2 * self.n + 1,)) 	
        stat_matrix = numpy.array(self.statistics)
        return numpy.sum(stat_matrix, axis=0)

    def update(self, candidate, *refs):
        refs = [normalize(ref) for ref in refs]
        candidate = normalize(candidate)
        
        stats = numpy.zeros((2 * self.n + 1,))	
        stats[-1] = get_ref_length(map(len, refs), len(candidate))
         
        cand_ngram_counts = count_ngrams([candidate], self.n)
        refs_ngram_counts = count_ngrams(refs, self.n)

        for ngram, count in cand_ngram_counts.items():
            stats[len(ngram) + self.n - 1] += min(count, refs_ngram_counts.get(ngram, 0)) 
        for k in xrange(1, self.n + 1):
            stats[k - 1] = max(len(candidate) - k + 1, 0)
        self.statistics.append(stats)
	
    def compute(self, smoothing=0, length_penalty=1):
        precs = numpy.zeros((self.n + 1,))
        stats = self.aggregate()
        log_bleu = 0.
        
        for k in range(self.n):
            correct = float(stats[self.n + k] + smoothing)
            if correct == 0.:
                return 0., precs
            total = float(stats[k] + 2*smoothing)
            precs[k] = numpy.log(correct) - numpy.log(total)
            log_bleu += precs[k]
	        
        log_bleu /= float(self.n)
        stats[-1] = stats[-1] * length_penalty
        log_bleu += min(0, 1 - float(stats[0]/stats[-1]))
        return numpy.exp(log_bleu), numpy.exp(precs) 
	
    def reset(self):
        self.statistics = []

class BleuEvaluator(object):
    """ Bleu evaluator
    """
    def __init__(self):
        self.bleu = Bleu()

    def evaluate(self, prediction, target):
        if len(target) != len(prediction):
            raise ValueError('Target and predictions length mismatch!')
        
        # Assume ordered list and take only the first one
        if isinstance(prediction[0], list):
            prediction = [x[0] for x in prediction]
         
        self.bleu.reset()
        for ts, ps in zip(target, prediction):
            self.bleu.update(ps, *ts)
        return self.bleu.compute()



class Recall:
    """
    Evaluate mean recall at utterance level.
    Use: 
    >>> r = Recall()
    >>> r.update("i have it", ["i have is", "i have some"])
    >>> r.update("i have it", ["i have is", "i have it"])
    >>> print r.compute()
    0.5
    >>> r.reset()
    """
    def __init__(self, n):
        self.n = n
        self.statistics = []
       	
    def aggregate(self):
        if len(self.statistics) == 0:
            return numpy.zeros((1,)) 	
        stat_matrix = numpy.array(self.statistics)
        return numpy.mean(stat_matrix)

    def update(self, candidates, ref):
        stats = numpy.zeros((1,))	
        
        for candidate in candidates:
            if candidate == ref:
                stats[0] = 1
                self.statistics.append(stats)
                break
	
    def compute(self):
        stats = self.aggregate()
        return stats
	
    def reset(self):
        self.statistics = []

class RecallEvaluator(object):
    """ Recall evaluator
    """
    def __init__(self, n=5):
        self.recall = Recall(n)
        self.n = n

    def evaluate(self, prediction, target):
        if len(target) != len(prediction):
            raise ValueError('Target and predictions length mismatch!')

        self.recall.reset()
        for ts, ps in zip(target, prediction):
            assert(len(ps) >= self.n)
            self.recall.update(ps[0:self.n], *ts)

        return self.recall.compute()


class MRR:
    """
    Evaluate mean reciprocal rank.
    Use: 
    >>> r = MRR()
    >>> r.update("i have it", ["i have is", "i have some"])
    >>> r.update("i have it", ["i have is", "i have it"])
    >>> print r.compute()
    0.25
    >>> r.reset()
    """
    def __init__(self, n):
        self.n = n
        self.statistics = []
       	
    def aggregate(self):
        if len(self.statistics) == 0:
            return numpy.zeros((1,)) 	
        stat_matrix = numpy.array(self.statistics)
        return numpy.mean(stat_matrix)

    def update(self, candidates, ref):
        stats = numpy.zeros((1,))	
        
        for index in range(len(candidates)):
            if candidates[index] == ref:
                stats[0] = 1/(index+1)
                self.statistics.append(stats)
                break
	
    def compute(self):
        stats = self.aggregate()
        return stats
	
    def reset(self):
        self.statistics = []

class MRREvaluator(object):
    """ Mean reciprocal rank evaluator
    """
    def __init__(self, n=5):
        self.mrr = MRR(n)
        self.n = n

    def evaluate(self, prediction, target):
        if len(target) != len(prediction):
            raise ValueError('Target and predictions length mismatch!')

        self.mrr.reset()
        for ts, ps in zip(target, prediction):
            assert(len(ps) >= self.n)
            self.mrr.update(ps[0:self.n], *ts)

        return self.mrr.compute()
