from typing import Optional

import jax.numpy
import jax.numpy as np 
from jax import grad,jit,vmap 

from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


from jax.experimental import stax # neural network library
from jax.experimental.stax import Dense, Relu # neural network layers

from jax.experimental import optimizers
from jax import jit, grad
import numpy as onp

rng = jax.random.PRNGKey(0)

def sample_normal(N):
  D = 2
  return jax.random.normal(rng, (N, D))

# un-normalized negative logp loss 
def log_prob_normal(x):
  return np.sum(-np.square(x)/2.0,axis=-1)

# 
def nvp_forward(net_params, shift_and_log_scale_fn, x, flip=False):
  d = x.shape[-1]//2
  x1, x2 = x[:, :d], x[:, d:]
  if flip:
    x2, x1 = x1, x2
  shift, log_scale = shift_and_log_scale_fn(net_params, x1)
  y2 = x2*np.exp(log_scale) + shift
  if flip:
    x1, y2 = y2, x1
  y = np.concatenate([x1, y2], axis=-1)
  return y

def nvp_inverse(net_params, shift_and_log_scale_fn, y, flip=False):
  d = y.shape[-1]//2
  y1, y2 = y[:, :d], y[:, d:]
  if flip:
    y1, y2 = y2, y1
  shift, log_scale = shift_and_log_scale_fn(net_params, y1)
  x2 = (y2-shift)*np.exp(-log_scale)
  if flip:
    y1, x2 = x2, y1
  x = np.concatenate([y1, x2], axis=-1)
  return x, log_scale

def sample_nvp(net_params, shift_log_scale_fn, base_sample_fn, N, flip=False):
  x = base_sample_fn(N)
  return nvp_forward(net_params, shift_log_scale_fn, x, flip=flip)

def log_prob_nvp(net_params, shift_log_scale_fn, base_log_prob_fn, y, flip=False):
  x, log_scale = nvp_inverse(net_params, shift_log_scale_fn, y, flip=flip)
  ildj = -np.sum(log_scale, axis=-1)
  return base_log_prob_fn(x) + ildj

def init_nvp():
  D = 2
  net_init, net_apply = stax.serial(
    Dense(64), 
    Relu, 
    Dense(64), 
    Relu, 
    Dense(64),
    Relu,
    Dense(D)
  )
  in_shape = (-1, D//2)
  out_shape, net_params = net_init(rng, in_shape)
  def shift_and_log_scale_fn(net_params, x1):
    s = net_apply(net_params, x1)
    return np.split(s, 2, axis=1)
  return net_params, shift_and_log_scale_fn

def init_nvp_chain(n=2):
  flip = False
  ps, configs = [], []
  for i in range(n):
    p, f = init_nvp()
    ps.append(p), configs.append((f, flip))
    flip = not flip
  return ps, configs

def make_log_prob_fn(p, log_prob_fn, config):
  shift_log_scale_fn, flip = config
  return lambda x: log_prob_nvp(p, shift_log_scale_fn, log_prob_fn, x, flip=flip)

def log_prob_nvp_chain(ps, configs, base_log_prob_fn, y):
  log_prob_fn = base_log_prob_fn
  for p, config in zip(ps, configs):
    log_prob_fn = make_log_prob_fn(p, log_prob_fn, config)
  return log_prob_fn(y)

def generate(
    X,
    log_prob=log_prob_normal,
    nsteps=2e3,
    step_size=1e-3,
    train_energy=False,
    energy_fn=None): 
    ps, cs = init_nvp_chain(5)

    def loss(params, batch):
      return -np.mean(log_prob_nvp_chain(params, cs, log_prob, batch))

    opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)

    @jit
    def step(i, opt_state, batch):
      params = get_params(opt_state)
      tmp_loss = loss(params,batch)
      g = grad(loss)(params, batch)
      return tmp_loss,opt_update(i, g, opt_state)

    data_scale = 100.0 
    data_mean = 350.0 
    X = (X - data_mean) / data_scale
    iters = int(nsteps)
    data_generator = (X[onp.random.choice(X.shape[0], 250)] for _ in range(iters))
    opt_state = opt_init(ps)

    losses = []

    for i in range(iters):
      loss, opt_state = step(i, opt_state, next(data_generator))
      losses.append(loss)
      if (i%100==0):
        print("Step %d: "%(i),onp.mean(losses))
    ps = get_params(opt_state)

    from matplotlib import animation, rc
    from IPython.display import HTML, Image

    x = sample_normal(100000)
    values = [x*data_scale+data_mean]
    for p, config in zip(ps, cs):
      shift_log_scale_fn, flip = config
      x = nvp_forward(p, shift_log_scale_fn, x, flip=flip)
      values.append(x*data_scale+data_mean)

    # First set up the figure, the axis, and the plot element we want to animate
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('equal')
    ax.set(xlim=(0, 700), ylim=(0, 700))

    y = values[0]
    paths = ax.scatter(y[:, 0], y[:, 1], s=0.5, alpha=0.5)
    values.append(values[-1])

    def animate(i):
      l = int(i)//48
      t = (float(i%48))/48
      y = (1-t)*values[l] + t*values[l+1]
      paths.set_offsets(y)
      return (paths,)

    anim = animation.FuncAnimation(fig, animate, frames=48*len(cs)+48, interval=1)
    anim.save('anim.mp4',fps=60)

    return values[-1]


if __name__ == "__main__":
    n_samples = 2000
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    X, y = noisy_moons
    X = StandardScaler().fit_transform(X)


