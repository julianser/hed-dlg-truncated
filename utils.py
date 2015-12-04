import numpy
import adam
import theano
import theano.tensor as T
from collections import OrderedDict

PRINT_VARS = True

def DPrint(name, var):
    if PRINT_VARS is False:
        return var

    return theano.printing.Print(name)(var)

def sharedX(value, name=None, borrow=False, dtype=None):
    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared(theano._asarray(value, dtype=dtype),
                         name=name,
                         borrow=borrow)

def Adam(grads, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
    return adam.Adam(grads, lr, b1, b2, e)

def Adagrad(grads, lr):
    updates = OrderedDict()
    for param in grads.keys():
        # sum_square_grad := \sum g^2
        sum_square_grad = sharedX(param.get_value() * 0.)
        if param.name is not None:
            sum_square_grad.name = 'sum_square_grad_' + param.name

        # Accumulate gradient
        new_sum_squared_grad = sum_square_grad + T.sqr(grads[param])

        # Compute update
        delta_x_t = (- lr / T.sqrt(numpy.float32(1e-5) + new_sum_squared_grad)) * grads[param]

        # Apply update
        updates[sum_square_grad] = new_sum_squared_grad
        updates[param] = param + delta_x_t
    return updates

def Adadelta(grads, decay=0.95, epsilon=1e-6):
    updates = OrderedDict()
    for param in grads.keys():
        # mean_squared_grad := E[g^2]_{t-1}
        mean_square_grad = sharedX(param.get_value() * 0.)
        # mean_square_dx := E[(\Delta x)^2]_{t-1}
        mean_square_dx = sharedX(param.get_value() * 0.)

        if param.name is not None:
            mean_square_grad.name = 'mean_square_grad_' + param.name
            mean_square_dx.name = 'mean_square_dx_' + param.name

        # Accumulate gradient
        new_mean_squared_grad = (
            decay * mean_square_grad +
            (1 - decay) * T.sqr(grads[param])
        )

        # Compute update
        rms_dx_tm1 = T.sqrt(mean_square_dx + epsilon)
        rms_grad_t = T.sqrt(new_mean_squared_grad + epsilon) 
        delta_x_t = - rms_dx_tm1 / rms_grad_t * grads[param]

        # Accumulate updates
        new_mean_square_dx = (
            decay * mean_square_dx +
            (1 - decay) * T.sqr(delta_x_t)
        )

        # Apply update
        updates[mean_square_grad] = new_mean_squared_grad
        updates[mean_square_dx] = new_mean_square_dx
        updates[param] = param + delta_x_t

    return updates

def RMSProp(grads, lr, decay=0.95, eta=0.9, epsilon=1e-6): 
    """ 
    RMSProp gradient method
    """ 
    updates = OrderedDict()
    for param in grads.keys():
        # mean_squared_grad := E[g^2]_{t-1}
        mean_square_grad = sharedX(param.get_value() * 0.)
        mean_grad = sharedX(param.get_value() * 0.)
        delta_grad = sharedX(param.get_value() * 0.)

        if param.name is None:
            raise ValueError("Model parameters must be named.")
        
        mean_square_grad.name = 'mean_square_grad_' + param.name
        
        # Accumulate gradient
        
        new_mean_grad = (decay * mean_grad + (1 - decay) * grads[param])
        new_mean_squared_grad = (decay * mean_square_grad + (1 - decay) * T.sqr(grads[param]))

        # Compute update 
        scaled_grad = grads[param] / T.sqrt(new_mean_squared_grad - new_mean_grad ** 2 + epsilon)
        new_delta_grad = eta * delta_grad - lr * scaled_grad 

        # Apply update
        updates[delta_grad] = new_delta_grad
        updates[mean_grad] = new_mean_grad
        updates[mean_square_grad] = new_mean_squared_grad
        updates[param] = param + new_delta_grad

    return updates 

class Maxout(object):
    def __init__(self, maxout_part):
        self.maxout_part = maxout_part

    def __call__(self, x):
        shape = x.shape
        if x.ndim == 2:
            shape1 = T.cast(shape[1] / self.maxout_part, 'int64')
            shape2 = T.cast(self.maxout_part, 'int64')
            x = x.reshape([shape[0], shape1, shape2])
            x = x.max(2)
        else:
            shape1 = T.cast(shape[2] / self.maxout_part, 'int64')
            shape2 = T.cast(self.maxout_part, 'int64')
            x = x.reshape([shape[0], shape[1], shape1, shape2])
            x = x.max(3)
        return x

def UniformInit(rng, sizeX, sizeY, lb=-0.01, ub=0.01):
    """ Uniform Init """
    return rng.uniform(size=(sizeX, sizeY), low=lb, high=ub).astype(theano.config.floatX)
 
def OrthogonalInit(rng, sizeX, sizeY, sparsity=-1, scale=1):
    """ 
    Orthogonal Initialization
    """

    sizeX = int(sizeX)
    sizeY = int(sizeY)

    assert sizeX == sizeY, 'for orthogonal init, sizeX == sizeY'

    if sparsity < 0:
        sparsity = sizeY
    else:
        sparsity = numpy.minimum(sizeY, sparsity)

    values = numpy.zeros((sizeX, sizeY), dtype=theano.config.floatX)
    for dx in xrange(sizeX):
        perm = rng.permutation(sizeY)
        new_vals = rng.normal(loc=0, scale=scale, size=(sparsity,))
        values[dx, perm[:sparsity]] = new_vals

    # Use SciPy:
    if sizeX*sizeY > 5000000:
        import scipy
        u,s,v = scipy.linalg.svd(values)
    else:
        u,s,v = numpy.linalg.svd(values)
    values = u * scale
    return values.astype(theano.config.floatX)

def GrabProbs(classProbs, target, gRange=None):
    if classProbs.ndim > 2:
        classProbs = classProbs.reshape((classProbs.shape[0] * classProbs.shape[1], classProbs.shape[2]))
    else:
        classProbs = classProbs
    
    if target.ndim > 1:
        tflat = target.flatten()
    else:
        tflat = target 
    return T.diag(classProbs.T[tflat])

def NormalInit(rng, sizeX, sizeY, scale=0.01, sparsity=-1):
    """ 
    Normal Initialization
    """

    sizeX = int(sizeX)
    sizeY = int(sizeY)
    
    if sparsity < 0:
        sparsity = sizeY
     
    sparsity = numpy.minimum(sizeY, sparsity)
    values = numpy.zeros((sizeX, sizeY), dtype=theano.config.floatX)
    for dx in xrange(sizeX):
        perm = rng.permutation(sizeY)
        new_vals = rng.normal(loc=0, scale=scale, size=(sparsity,))
        values[dx, perm[:sparsity]] = new_vals
        
    return values.astype(theano.config.floatX)

def ConvertTimedelta(seconds_diff): 
    hours = seconds_diff // 3600
    minutes = (seconds_diff % 3600) // 60
    seconds = (seconds_diff % 60)
    return hours, minutes, seconds

def SoftMax(x):
    x = T.exp(x - T.max(x, axis=x.ndim-1, keepdims=True))
    return x / T.sum(x, axis=x.ndim-1, keepdims=True)

# Does batch normalization of input variable
def VariableNormalization(x, mask=None, axes=0):
    if mask:
         mask = mask.dimshuffle(0, 1, 'x')
         x_masked = x*mask

         average = T.sum(x_masked, axis=axes)/T.sum(mask, axis=axes)
         if average.ndim == 1:
             x_zero_average = x_masked - average.dimshuffle('x', 'x', 0)
         else:
             x_zero_average = x_masked - average.dimshuffle('x', 0)

         x_std = T.sqrt(T.sum(x_zero_average**2)/T.sum(mask, axis=axes) + 0.0000001)
         return x_zero_average / x_std
    else:
         return (x - T.mean(x, axis=axes)) / T.sqrt(T.var(x, axis=axes) + 0.0000001)





