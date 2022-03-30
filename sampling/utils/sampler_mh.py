from typing import Optional

import jax.numpy
import numpy

def sample_mh(x, key, energy_fn = None, nsteps = 50, stepsize=0.01):
    """
        adapted from https://github.com/noegroup/stochastic_normalizing_flows/blob/main/snf_code/snf_code/image.py#L138

        This samples a bin with probability given by density and then perturb the coordinate of bin uniformly

        :param N: number of samples
        :param density: density or value at each pixel
        :param key: JAX needs a key for its random number generator, it is just that
        :return: samples shape=(N,2)
    """
    # normalize

    E0 = energy_fn(x).reshape(x.shape[0],1)
    E = E0
    key, subkey = jax.random.split(key)

    for i in range(nsteps):
        # proposal step
        key, subkey = jax.random.split(key)
        dx = stepsize * jax.random.normal(key,shape=x.shape)
        
        xprop = x + dx
        Eprop = energy_fn(xprop).reshape(x.shape[0],1)
        print(numpy.sum(Eprop))
    
        # acceptance step
        key, subkey = jax.random.split(key)
        rand_val = jax.random.uniform(key,shape=(x.shape[0],1))
        
        acc = ((-jax.numpy.log(rand_val)) > (Eprop - E)) 

        acc = jax.numpy.float32(acc)
        acc = acc.reshape(x.shape[0],1)
        acc_comp = 1.0 - acc 
        
        x = numpy.multiply(1.0-acc, x) + numpy.multiply(acc, xprop)
        E = numpy.multiply(1.0-acc, E) + numpy.multiply(acc, Eprop)

    dW = E - E0
    return x, dW 

