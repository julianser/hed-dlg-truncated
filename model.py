import logging
import numpy
import theano
logger = logging.getLogger(__name__)

class Model(object):
    def __init__(self):
        self.floatX = theano.config.floatX
        # Parameters of the model
        self.params = []
    
    def save(self, filename):
        """
        Save the model to file `filename`
        """
        vals = dict([(x.name, x.get_value()) for x in self.params])
        numpy.savez(filename, **vals)

    def load(self, filename):
        """
        Load the model.
        """
        vals = numpy.load(filename)
        for p in self.params:
            if p.name in vals:
                logger.debug('Loading {} of {}'.format(p.name, p.get_value(borrow=True).shape))
                if p.get_value().shape != vals[p.name].shape:
                    raise Exception('Shape mismatch: {} != {} for {}'.format(p.get_value().shape, vals[p.name].shape, p.name))
                p.set_value(vals[p.name])
            else:
                logger.error('No parameter {} given: default initialization used'.format(p.name))
                unknown = set(vals.keys()) - {p.name for p in self.params}
                if len(unknown):
                    logger.error('Unknown parameters {} given'.format(unknown))
