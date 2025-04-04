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

if rank==0:
    t_start = time()

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
iterations = 300 #0 #00



def collect_data(preconditioner):

    local_results = []

    for i, alpha in enumerate(local_alphas):
        key = random.PRNGKey(1)
        model = NNFunc(alpha=alpha, eps=0.25, D=D, N=N)
        params = model.init(key, X)
        state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
        init_params = state.params
        apply_fn = state.apply_fn

        init_output = apply_fn(init_params, X)
        init_outputt = apply_fn(init_params, Xt)


        @jit
        def loss_fn(params, x, y, init_out):
            pred = apply_fn(params, x) - init_out - y
            return jnp.mean(pred**2 / alpha**2)

        @jit
        def train_step_smw(state, init_out):
            loss, grads = value_and_grad(loss_fn)(state.params, X, y, init_out)

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


        @jit
        def train_step_gn_exact(state, init_out):
            loss, grads = value_and_grad(loss_fn)(state.params, X, y, init_out)

            flat_jacobian = []
            jacobian_fn = jax.jacrev(
                lambda params, x: state.apply_fn(params, x)
            )(state.params, X)
            for layer in jacobian_fn['params'].values():
                flat_jacobian.append(layer.reshape(450, -1))
            jacobian_fn = jnp.concatenate(flat_jacobian, axis=1)
            u, s, vh = jnp.linalg.svd(jacobian_fn, full_matrices=False)

            # Compute inverse
            threshold = 1e-3
            s_inv = jnp.where(s > 1e-3, s ** -2, threshold)

            # Apply to grads
            flattened, back = jax.flatten_util.ravel_pytree(grads)
            my_change =  vh.T @ jnp.diag(s_inv) @ vh @ flattened
            grads = back(alpha * my_change)

            return state.apply_gradients(grads=grads), loss


        train_step = train_step_smw if precond == "smw" else train_step_gn_exact

        @jit
        def scan_body_fn(carry, _):
            state = carry
            state, loss = train_step(state, init_output)
            tr_loss = alpha**2 * loss_fn(state.params, X, y, init_output)
            te_loss = alpha**2 * loss_fn(state.params, Xt, yt, init_outputt)
            return state, (tr_loss, te_loss)

        @jit
        def run_training(state):
            state, (tr_losses, te_losses) = jax.lax.scan(scan_body_fn, state, None, length=iterations)
            return state, jnp.stack(tr_losses), jnp.stack(te_losses)

        tic = time()
        state, tr_losses, te_losses = run_training(state)
        _ = tr_losses.block_until_ready()
        elapsed_t = time() - tic

        hh, rem = divmod(int(elapsed_t), 3600)
        mm, ss = divmod(rem, 60)
        t_str = f"{hh:02}:{mm:02}:{ss:02}"
        print(f"rank={rank}: alpha={alpha:1.2e}, "
              f"train loss: {tr_losses[-1]:1.2e} | test loss: {te_losses[-1]:1.2e} | time = {t_str}", flush=True)
        local_results.append((float(alpha), tr_losses, te_losses))

    return comm.gather(local_results, root=0)


for precond in ['smw']:

    all_results = collect_data(precond)

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
        plt.savefig(f"grokk2_{precond}.pdf")
        #plt.show()


comm.Barrier()
if rank == 0:
    t_total = time() - t_start
    hh, rem = divmod(int(t_total), 3600)
    mm, ss = divmod(rem, 60)
    print(f"total_time={hh:02}:{mm:02}:{ss:02}")

MPI.Finalize()

