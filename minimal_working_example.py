import theano
import numpy

x = theano.tensor.fvector('x')
W = theano.shared(numpy.asarray([0.2, 0.7]), 'W')
y = (x * W).sum()

f = theano.function([x], y)

output = f([1.0, 1.0])
print output
