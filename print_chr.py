import pprint
import sys
import cPickle
import numpy

print "{}".format(pprint.pformat(cPickle.load(open(sys.argv[1], 'r'))))
