#!/usr/bin/python -tt

from pylab import *
import numpy as np
gaussian = lambda x: 3*exp(-(30-x)**2/20.)
data = gaussian(arange(100))
print type(data)
plot(data)

X = arange(data.size)
x = sum(X*data)/sum(data)
width = sqrt(abs(sum((X-x)**2*data)/sum(data)))
themax = data.max()
print x,width,themax
fit = lambda t: themax*exp(-(t-x)**2/(2*width**2))

plot(fit(X))

show()
 
