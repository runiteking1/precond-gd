from functools import partial
from typing import Union

import os

# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ['JAX_PLATFORM_NAME'] = 'cpu'
# os.environ['JAX_PLATFORMS']='cpu'

import chex
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState  # Useful dataclass to keep train state
from jax.tree import map as tree_map
from tqdm import tqdm

from cg import conjugate_gradient
from data_utils import batch_data
from jacobian_tools import flatten_jacobian
from model_utils import create_train_state
from MLP import MLP_1D
from plotting import plot_results
from matplotlib import pyplot as plt
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'highest')

@chex.dataclass(frozen=True)
class PrecondData:
    "Structure to store parameters for the precond"
    method: str = 'smw'  # Default Sherman Morrison Williams
    epoch: int = 0  # Default apply from the 0th iteration
    lamb: Union[float, str] = 0.1  # LM dynamics
    line_search: bool = False  # Line search or not

    # For GN
    thresh: float = 1e-3

    # For CG:
    max_iters: int = 100


def train_model(state: TrainState, x, y, num_iterations:int = 1_000, 
        obtain_matrices: bool=False, 
        lm_schedule: PrecondData = None, batch_size=64
    ): 
    @jax.jit
    def model_output(params, x):
        # f(\theta, x); jax defaults to derivs of first component
        return state.apply_fn(params, x)

    # Explicitly computes \partial f/\partial \theta to compute gradient of error
    @jax.jit
    def compute_manual_gradient(state, x, y):
        # For MSE
        n = x.shape[0]

        # Compute the value of f(\theta, x) and the function for VJP
        preds, vjp_fn = jax.vjp(model_output, state.params, x)
        residuals = (preds - y)

        # Explicitly calculate loss for metrics
        loss = 1/n * jnp.sum(residuals ** 2)
        
        # residuals^T jacobian (Might need transpose here in 2D)
        manual_grad = vjp_fn(residuals * 2 / n)[0] # First term is wrt params

        return manual_grad, loss, vjp_fn

    @partial(jax.jit, static_argnames='lm_info')
    def train_step(state, x, y, lm_info: PrecondData = None):
        # Manually calculate the gradient to demonstrate
        grads, loss, vjp_fn = compute_manual_gradient(state, x, y)

        # If lm information is passed in
        if lm_info:
            print(f'Running {lm_info.method}')
            if lm_info.method == 'exp': 
                r = 3
                # keys = jax.random.split(lm_info.key, r)
                key = jax.random.PRNGKey(0)
                keys = jax.random.split(key, r)
                
                temp, back = jax.flatten_util.ravel_pytree(grads)
                n = len(temp)
                m = len(x)

                # Generate 
                Omega = jnp.stack([jax.random.normal(keys[i], (n,)) for i in range(r + 1)], axis=1)
            
                Q = jnp.zeros((m, 0))
                for j in range(1, r+1):   
                    _, y_j = jax.jvp(lambda p:  state.apply_fn(p, x), (state.params,), (back(Omega[:, j]),)) 
                    y_j = jnp.squeeze(y_j)
                    # print(y_j, j)
                    # y_j = jnp.dot(A, Omega[:, j])      
                    # print(y_j, j)
            
                    y_j = y_j - jnp.dot(Q, jnp.dot(Q.T, y_j))
                    y_j = y_j / jnp.linalg.norm(y_j)
                    Q = jnp.hstack((Q, y_j[:, None]))
                    
                def apply_vjp_and_flatten(column): # Define a function that applies vjp_fn to a single column and flattens the result
                    column = column[:, jnp.newaxis] 
                    flat, _ = jax.flatten_util.ravel_pytree(vjp_fn(column)[0])
                    return flat

                # Use vmap to vectorize the function over the columns of Q
                B = jax.vmap(apply_vjp_and_flatten, in_axes=1, out_axes=0)(Q)    
                
                # Step 5
                Uhat, s_est, vh_est = jnp.linalg.svd(B)
                
                # Step 6
                # U_est = Q @ Uhat

                # Compute inverse
                threshold = 1e-3
                s_inv = jnp.where(s_est > lm_info.thresh, s_est**-2, threshold)
                
                # Apply to grads
                # print(flatten_grad(grads))
                flattened, back = jax.flatten_util.ravel_pytree(grads)
                my_change = vh_est[0:len(s_inv), :].T @ jnp.diag(s_inv) @ vh_est[0:len(s_inv), :] @ flattened
                grads = back(my_change)
                lm_data = None
                
            elif lm_info.method == 'poly': 
                # P = tr(B)I - B where B is J^TJ ; don't solve anything
                jacobian_fn = flatten_jacobian(jax.jacrev(model_output)(state.params, x), x)
                B = jacobian_fn.T @ jacobian_fn
                # B2 = B @ B
    
                flattened, back = jax.flatten_util.ravel_pytree(grads)
                grads = back(
                    (jnp.trace(B) * jnp.eye(B.shape[0]) - B) @ flattened
                    # (.5 * ((jnp.trace(B) ** 2) - jnp.trace(B2)) * jnp.eye(B.shape[0]) - jnp.trace(B) * B + B2) @ flattened
                )
                lm_data = None
            elif lm_info.method == 'gn-exact': 
                # TODO: try https://arxiv.org/pdf/1905.11675
                # algorithim 1
                # move *beforeeee* 
                
                jacobian_fn = flatten_jacobian(jax.jacrev(model_output)(state.params, x), x)
                # print(jacobian_fn[0, :])
                # Compute SVD
                u, s, vh = jnp.linalg.svd(jacobian_fn, full_matrices=False) 
    
                # Compute inverse
                threshold = 1e-3
                s_inv = jnp.where(s > lm_info.thresh, s**-2, threshold)
                # print(f'{s_inv=}')
    
                # Apply to grads
                # print(flatten_grad(grads))
                flattened, back = jax.flatten_util.ravel_pytree(grads)
                # print(flattened)
                my_change = vh.T @ jnp.diag(s_inv) @ vh @ flattened
                grads = back(my_change)
                lm_data = None
                
            elif lm_info.method == 'smw':            
                # For ease of access
                lamb = lm_info.lamb
    
                # J \vec k
                _, jvp_output = jax.jvp(lambda p: state.apply_fn(p, x), (state.params,), (grads,)) 
    
                # Construct and compute (I + 1/lambda JJ^T)^{-1} (Jv)
                jacobian_fn = flatten_jacobian(jax.jacrev(model_output)(state.params, x), x)
                mat = jacobian_fn @ jacobian_fn.T
                out = jnp.linalg.solve(jnp.eye(mat.shape[0]) + 1 / lamb * mat, jvp_output / (lamb ** 2))
    
                adjustment = vjp_fn(out)[0]
    
                grads = tree_map(lambda a, b: 1 / lamb * a - b, grads, adjustment)
                lm_data = None
            elif lm_info.method == 'cg':
                # Define mat-vec for (\lambda I + J^T J)
                # Matrix-free implementation; we use vjp_fn computed above and jvp is cheap 
                def matvec(k, state, x):
                    # Compute J^T J k using jvp and vjp
                    _, jvp_output = jax.jvp(lambda p: state.apply_fn(p, x), (state.params,), (k,))
                    Av = vjp_fn(jvp_output)[0]
        
                    # Add the regularization term lambda * k
                    Av = tree_map(lambda a, b: a + lm_info.lamb * b, Av, k)
                    return Av
                    
                grads, lm_data = conjugate_gradient(matvec, grads, state, x, max_iter=lm_info.max_iters)
            else: 
                raise Exception('Not a method')
        else:
            lm_data = None
            
        # Apply the gradient update
        state = state.apply_gradients(grads=grads)

        return state, loss, lm_data

    @jax.jit
    def get_eigs(state, x, return_matrix: bool=True):
        """
        Only really for diagnostics
        """
        # Calculate explicitly 
        jacobian_fn = flatten_jacobian(jax.jacrev(model_output)(state.params, x), x)
        mat = jacobian_fn.T @ jacobian_fn
        eigs = jnp.linalg.eigvalsh(mat)

        if return_matrix:
            return {'eigs': eigs, 'mat': jacobian_fn}
        else:
            return {'eigs': eigs}
    
    metrics_history = {
        'train_loss': [], 'eigs': [], 'mat': [],
        'lm_data': [],
        'prediction': []
    }

    rng_key = jax.random.PRNGKey(0)

    for epoch in tqdm(range(num_iterations), leave=True):
        # Jax randomness kinda sucks; but this ensures that each epoch the batches are diff
        rng_key, subkey = jax.random.split(rng_key)

        epoch_loss = 0
        num_batches = 0
        
        for x_batch, y_batch in batch_data(x, y, batch_size, subkey):
            # print(x_batch, y_batch)
            if lm_schedule and epoch >= lm_schedule.epoch:
                state, loss, lm_data = train_step(state, x_batch, y_batch, lm_schedule)
            else: 
                state, loss, lm_data = train_step(state, x_batch, y_batch)
            epoch_loss += loss
            num_batches += 1
        # break
        # Compute the average loss for the epoch
        avg_loss = epoch_loss / num_batches
        metrics_history['train_loss'].append(avg_loss)
        metrics_history['lm_data'].append(lm_data)


        if (epoch % 100) == 0: 
            if obtain_matrices:
                out = get_eigs(state, x)
                
                for key in out.keys():
                    metrics_history[key].append(out[key])

            metrics_history['prediction'].append(
                state.apply_fn(state.params, x)
            )


    return state, metrics_history
    
def run_exp():
    print(jax.default_backend())

    # --------------------- SGD -----------------
    # x = jnp.linspace(0, 1, 100).reshape(-1, 1)
    # y = jnp.sin(1 * jnp.pi * x)

    # model = MLP_1D(num_layers=4, hidden_dim=10)
    # key = jax.random.PRNGKey(0)
    # lm_info = PrecondData(
    #     method='gn-exact', epoch=0, lamb=1e-1,
    #     # thresh=1e-1
    #     # key=key
    # )

    # # print(model.tabulate(
    # #     jax.random.key(0), x, compute_flops=True,
    # #     compute_vjp_flops=True)
    # # )

    # state = create_train_state(model, learning_rate=1e-2, momentum=0, optimizer='sgd')
    # state, metrics_history_sgd_low = train_model(state, x, y, num_iterations=1000, obtain_matrices=False,
    #                                      lm_schedule=None,
    #                                      )
    # plot_results(state, metrics_history_sgd_low, x, y)

    # x = jnp.linspace(0, 1, 100).reshape(-1, 1)
    # y = jnp.sin(8 * jnp.pi * x)
    # model = MLP_1D(num_layers=4, hidden_dim=10)
    # state = create_train_state(model, learning_rate=1e-2, momentum=0, optimizer='sgd')
    # state, metrics_history_sgd_high = train_model(state, x, y, num_iterations=1000, obtain_matrices=False,
    #                                      lm_schedule=None,
    #                                      )
    # plot_results(state, metrics_history_sgd_high, x, y)

    # plt.figure()
    # plt.plot(metrics_history_sgd_high['train_loss'])
    # plt.plot(metrics_history_sgd_low['train_loss'])
    # --------------------- END SGD ----------------- 

    # ------------------- gn-exact 
    # x = jnp.linspace(0, 1, 100).reshape(-1, 1)
    # y = jnp.sin(1 * jnp.pi * x) #+ 5 * jnp.sin(5 * jnp.pi * x + 2) + 7 * jnp.sin(7 * jnp.pi * x - 1)
    # # model = MLP(num_layers=8,hidden_dim=24)
    # model = MLP_1D(num_layers=6,hidden_dim=8)
    # key = jax.random.PRNGKey(0)
    lm_info = PrecondData(
        method='gn-exact', epoch=0, lamb=1e-1,
        # thresh=1e-1
        # key=key
    )
    # state = create_train_state(model, learning_rate=1e-2, momentum=0, optimizer='sgd')
    # state, metrics_history_gn_low = train_model(state, x, y, num_iterations=5000, obtain_matrices=False, 
    #                                     lm_schedule=lm_info,
    #                                     )
    # plot_results(state, metrics_history_gn_low, x, y)

    # n = len(y)
    # all_fft = []
    # for est in metrics_history_gn_low['prediction']:
    #     fft_err = jnp.fft.fft(y - est) / n
    #     fft_err = fft_err[:n // 2]
    #     fft_err = jnp.abs(fft_err)
    #     all_fft.append(fft_err[::5].flatten())
    
    # for i, series in enumerate(all_fft):
    #     plt.semilogy(series, label=f'Series {i * 5}')  # Label with index

    # plt.title('Error in frequency')
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Place legend outside
    # plt.show()

    # x = jnp.linspace(0, 1, 100).reshape(-1, 1)
    # y =jnp.sin(1 * jnp.pi * x) + 2.1 * jnp.sin(5 * jnp.pi * x)
    # model = MLP_1D(num_layers=6, hidden_dim=8)
    # state = create_train_state(model, learning_rate=1e-2, momentum=0, optimizer='sgd')
    # state, metrics_history_gn_high = train_model(state, x, y, num_iterations=5000, obtain_matrices=False,
    #                                      lm_schedule=lm_info,
    #                                      )
    # plot_results(state, metrics_history_gn_high, x, y)

    x = jnp.linspace(0, 1, 100).reshape(-1, 1)
    y =jnp.sin(8 * jnp.pi * x)
    model = MLP_1D(num_layers=6, hidden_dim=8)
    state = create_train_state(model, learning_rate=1e-3, momentum=0, optimizer='sgd')
    state, metrics_history_gn_simple = train_model(state, x, y, num_iterations=5000, obtain_matrices=False,
                                         lm_schedule=lm_info,
                                         )
    plot_results(state, metrics_history_gn_simple, x, y)


    plt.figure()
    # plt.semilogy(metrics_history_gn_high['train_loss'])
    # plt.semilogy(metrics_history_gn_low['train_loss'])
    plt.semilogy(metrics_history_gn_simple['train_loss'])
    # ------------------- gn-exact 

    plt.show()



if __name__ == '__main__':
    print('Running experiment 1.')
    run_exp()
