import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad, jvp, vjp, jacrev
from jax.flatten_util import ravel_pytree
from flax.training import train_state
import optax
from flax import linen as nn
from mpi4py import MPI
import matplotlib.pyplot as plt

from time import time

# Model
class NNFunc(nn.Module):
    alpha: float
    eps: float = 0.25
    D: int = 100
    N: int = 500

    def setup(self):
        self.W = self.param('W', random.normal, (self.N, self.D))

    def __call__(self, X):
        h = jnp.dot(self.W, X) / jnp.sqrt(self.D)
        f = self.alpha * jnp.mean(h + 0.5 * self.eps * h**2, axis=0)
        return f

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Data
D, P, N = 100, 450, 500
X = random.normal(random.PRNGKey(0), (D, P))
Xt = random.normal(random.PRNGKey(1), (D, 1000))
beta = random.normal(random.PRNGKey(2), (D,))

def target_fn(beta, X):
    return (X.T @ beta / jnp.sqrt(D))**2

y = target_fn(beta, X)
yt = target_fn(beta, Xt)

# Optimizer
eta = 0.5 * N
tx = optax.sgd(learning_rate=eta)

# Alphas and distribution
alphas = jnp.array([2**(-5), 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16, 32])[:4]
local_alphas = alphas[rank::size]
iterations = 100 #00

local_results = []

for i, alpha in enumerate(local_alphas):
    key = random.PRNGKey(1)
    model = NNFunc(alpha=alpha, eps=0.25, D=D, N=N)
    params = model.init(key, X)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    init_params = state.params
    apply_fn = state.apply_fn

    @jit
    def loss_fn(params, x, y, init_params):
        pred = apply_fn(params, x) - apply_fn(init_params, x) - y
        return jnp.mean(pred**2 / alpha**2)



    @jit
    def train_step(state, init_params):
        loss, grads = value_and_grad(loss_fn)(state.params, X, y, init_params)

        lamb = 1e-1 / alpha
        _, jvp_output = jvp(lambda p: apply_fn(p, X), (state.params,), (grads,))
        jacobian_fn = jacrev(lambda p: apply_fn(p, X))(state.params)
        flat_jacobian = jnp.concatenate([v.reshape(P, -1) for v in jacobian_fn['params'].values()], axis=1)
        mat = flat_jacobian @ flat_jacobian.T
        out = jnp.linalg.solve(jnp.eye(mat.shape[0]) + 1 / lamb * mat, jvp_output / lamb**2)
        _, vjp_fn = vjp(lambda p: apply_fn(p, X), state.params)
        adjustment = vjp_fn(out)[0]
        grads = jax.tree.map(lambda a, b: 1 / lamb * a - b, grads, adjustment)

        return state.apply_gradients(grads=grads), loss


    tr_losses = []
    te_losses = []

    tic = time()
    for _ in range(iterations):
        state, loss = train_step(state, init_params)
        tr_losses.append(alpha**2 * loss_fn(state.params, X, y, init_params))
        te_losses.append(alpha**2 * loss_fn(state.params, Xt, yt, init_params))
    elapsed_t = time()-tic
    out = f'train loss: {tr_losses[-1]:1.2e} | test loss: {te_losses[-1]:1.2e} | time = {elapsed_t:.2f}'
    print(f"rank={rank}: alpha={alpha:1.2e}, {out}", flush=True)
    local_results.append((float(alpha), jnp.stack(tr_losses), jnp.stack(te_losses)))

# Ggther all to root
all_results = comm.gather(local_results, root=0)

if rank == 0:
    flat = [r for sublist in all_results for r in sublist]
    flat.sort(key=lambda x: x[0])  # sort by alpha
    alphas_sorted, tr_losses_all, te_losses_all = zip(*flat)
    tr_losses_all = jnp.stack(tr_losses_all)
    te_losses_all = jnp.stack(te_losses_all)

    # Plot
    plt.rcParams.update({'font.size': 10})
    plt.figure()
    for i, alpha in enumerate(alphas_sorted[:-1]):
        plt.plot(jnp.arange(1, len(tr_losses_all[i]) + 1),
                 tr_losses_all[i] / tr_losses_all[i][0], '--', color=f'C{i}')
        plt.plot(jnp.arange(1, len(te_losses_all[i]) + 1),
                 te_losses_all[i] / te_losses_all[i][0], color=f'C{i}',
                 label=rf'$\alpha = 2^{{{int(jnp.log2(alpha))}}}$')

    plt.xscale('log')
    plt.xlabel(r'$t$')
    plt.ylabel('Normalized Loss')
    plt.legend(loc='upper right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("grokk2.pdf")
    plt.show()

