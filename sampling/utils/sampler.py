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

def kinetic_energy_hmc(v):
    energy = jax.numpy.square(v)
    energy = jax.numpy.sum(v,axis=1)
    return energy 

def hamiltonian(x, v, energy_fn):
    return energy_fn(x).reshape(x.shape[0],1) + kinetic_energy_hmc(v).reshape(x.shape[0],1)

def sample_hmc(x, key, energy_fn = None, energy_fn_grad = None, nsteps = 50, stepsize=0.01):
    # initialize velocity 
    key, subkey = jax.random.split(key)
    v = jax.random.normal(key,shape=x.shape)

    # initialize energy 
    E0 = hamiltonian(x,v,energy_fn)
    E = E0 

    vv = v - 0.5 * stepsize * energy_fn_grad(x).reshape(x.shape)

    # Initalize x to be the first step
    xx = x + stepsize * vv

    for i in range(nsteps):
        # Compute gradient of the log-posterior with respect to x
        gradient = energy_fn_grad(xx)

        # Update velocity
        vv = vv - stepsize * gradient.reshape(vv.shape)

        # Update x
        xx = xx + stepsize * vv

        Eprop = hamiltonian(xx,vv,energy_fn).reshape(x.shape[0],1)
        
    # Do a final update of the velocity for a half step
    vv = vv - 0.5 * stepsize * energy_fn_grad(xx).reshape(x.shape)

    Eprop = hamiltonian(xx,vv,energy_fn).reshape(x.shape[0],1)
    
    # acceptance step
    key, subkey = jax.random.split(key)
    rand_val = jax.random.uniform(key,shape=(x.shape[0],1))

    acc = ((-jax.numpy.log(rand_val)) > (Eprop - E0).reshape(rand_val.shape)) 

    acc = jax.numpy.float32(acc)
    acc = acc.reshape(x.shape[0],1)
    acc_comp = 1.0 - acc 
        
    x = numpy.multiply(1.0-acc, x) + numpy.multiply(acc, xx)
    v = numpy.multiply(1.0-acc, v) + numpy.multiply(acc, vv)
    E = numpy.multiply(1.0-acc, E0) + numpy.multiply(acc, Eprop)

    dW = E - E0
    return x, dW 

