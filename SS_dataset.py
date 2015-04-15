import numpy as np
import os, gc
import cPickle
import copy
import logging

import threading
import Queue

import collections

logger = logging.getLogger(__name__)
np.random.seed(1234)

class SSFetcher(threading.Thread):
    def __init__(self, parent):
        threading.Thread.__init__(self)
        self.parent = parent
        self.indexes = np.arange(parent.data_len)

    def run(self):
        diter = self.parent
        # Shuffle with parents random generator
        self.parent.rng.shuffle(self.indexes)

        offset = 0
        # Take groups of 10000 triples and group by length
        while not diter.exit_flag:
            last_batch = False
            triples = []

            while len(triples) < diter.batch_size:
                if offset == diter.data_len:
                    if not diter.use_infinite_loop:
                        last_batch = True
                        break
                    else:
                        # Infinite loop here, we reshuffle the indexes
                        # and reset the offset
                        np.random.shuffle(self.indexes)
                        offset = 0

                index = self.indexes[offset]
                s = diter.data[index]
                offset += 1

                # Append only if it is shorter than max_len
                if len(s) <= diter.max_len:
                    triples.append(s)

            if len(triples):
                diter.queue.put(triples)

            if last_batch:
                diter.queue.put(None)
                return

class SSIterator(object):
    def __init__(self,
                 rng,
                 batch_size,
                 triple_file=None,
                 dtype="int32",
                 can_fit=False,
                 queue_size=100,
                 cache_size=100,
                 shuffle=True,
                 use_infinite_loop=True,
                 max_len=1000):

        args = locals()
        args.pop("self")
        self.__dict__.update(args)
        self.rng = rng
        self.load_files()
        self.exit_flag = False

    def load_files(self):
        self.data = cPickle.load(open(self.triple_file, 'r'))
        self.data_len = len(self.data)
        logger.debug('Data len is %d' % self.data_len)

    def start(self):
        self.exit_flag = False
        self.queue = Queue.Queue(maxsize=self.queue_size)
        self.gather = SSFetcher(self)
        self.gather.daemon = True
        self.gather.start()

    def __del__(self):
        if hasattr(self, 'gather'):
            self.gather.exitFlag = True
            self.gather.join()

    def __iter__(self):
        return self

    def next(self):
        if self.exit_flag:
            return None
        
        batch = self.queue.get()
        if not batch:
            self.exit_flag = True
        return batch

if __name__ == '__main__':
    """ debug """
    import sys
    
    iterator = SSIterator(100, triple_file=sys.argv[1], use_infinite_loop=False)
    iterator.start()
    
    _cpt = 0
    while True:
        batch = iterator.next()
        if batch is None:
            break
        _cpt += 1
     
    print "Read %d batches" % _cpt
