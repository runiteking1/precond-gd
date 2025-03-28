from functools import partial
from typing import Callable, Union

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
from data_utils import batch_data_pinns, generate_training_data
from jacobian_tools import flatten_jacobian
from model_utils import create_train_state
from MLP import MLP_1D
from plotting import plot_results_pinns
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


def train_model(state: TrainState, x, num_iterations:int = 1_000, 
                forcing_function=lambda x: 2 * jnp.pi * jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1]),
                obtain_matrices: bool=False, 
                lm_schedule: PrecondData = None, batch_size=64
    ): 
    @jax.jit
    def model_single(params, x): 
        # Define eval of u
        u_fn = lambda params, x: jnp.squeeze(state.apply_fn(params, x))

        # Grad wrt x
        grad_u = jax.grad(u_fn, argnums=1)

        # Hessian wrt x TODO: is this efficient?
        hessian_u = jax.jacfwd(
            jax.jacrev(u_fn, argnums=1), argnums=1
        )

        # Get the grad and Hessian
        u_grad = grad_u(params, x)
        u_hessian = hessian_u(params, x)

        # Extract Laplacian
        u_xx = u_hessian[0, 0]
        u_yy = u_hessian[1, 1]

        return -u_xx - u_yy - forcing_function(x)
    
    # Create R(u(\theta, x)) for the PDE residual 
    pde_output = jax.vmap(
        model_single, in_axes=(None, 0) # Only stack xs, and not params
    )

    @jax.jit
    def bnd_output(params, x):
        return jnp.squeeze(state.apply_fn(params, x))
    
    @jax.jit
    def compute_manual_gradient_bnd(state, x):
        # For MSE
        n = x.shape[0]

        # Compute the value of f(\theta, x) and the function for VJP
        preds, vjp_fn = jax.vjp(bnd_output, state.params, x)
        residuals = preds

        # Explicitly calculate loss for metrics
        loss = 1/n * jnp.sum(residuals ** 2)
        
        # residuals^T jacobian (Might need transpose here in 2D)
        manual_grad = vjp_fn(residuals * 2 / n)[0] # First term is wrt params

        return manual_grad, loss, vjp_fn

    @jax.jit
    def compute_manual_gradient_pde(state, x):
        # For MSE
        n = x.shape[0]

        # Compute the value of f(\theta, x) and the function for VJP
        preds, vjp_fn = jax.vjp(pde_output, state.params, x)
        residuals = preds

        # Explicitly calculate loss for metrics
        loss = 1/n * jnp.sum(residuals ** 2)
        
        # residuals^T jacobian (Might need transpose here in 2D)
        manual_grad = vjp_fn(residuals * 2 / n)[0] # First term is wrt params

        return manual_grad, loss, vjp_fn

    @partial(jax.jit, static_argnames='lm_info')
    def train_step(state, interior_points, boundary_points, lm_info: PrecondData = None):
        print(interior_points.shape, boundary_points.shape)
        # Manually calculate the gradient to demonstrate
        grads_bnd, loss_bnd, vjp_fn_bnd = compute_manual_gradient_bnd(state, boundary_points)
        grads_pde, loss_pde, vjp_fn_pde = compute_manual_gradient_pde(state, interior_points)

        # Combine grads
        grads = jax.tree_util.tree_map(lambda x,y: x + y, grads_bnd, grads_pde)

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
                jacobian_fn_pde = flatten_jacobian(
                    jax.jacrev(pde_output)(state.params, interior_points), interior_points
                )
                jacobian_fn_bnd = flatten_jacobian(
                    jax.jacrev(bnd_output)(state.params, boundary_points), boundary_points
                )
                jacobian_fn = jnp.concat((jacobian_fn_pde, jacobian_fn_bnd))
              
                # jacobian_fn = flatten_jacobian(jax.jacrev(model_output)(state.params, x), x)

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
                _, jvp_output_pde = jax.jvp(
                    lambda p: pde_output(p, interior_points), (state.params,), (grads,)
                ) 
                _, jvp_output_bnd = jax.jvp(
                    lambda p: bnd_output(p, boundary_points), (state.params,), (grads,)
                ) 
                jvp_output = jnp.concat((jvp_output_pde, jvp_output_bnd))
    
                # Construct and compute (I + 1/lambda JJ^T)^{-1} (Jv)
                jacobian_fn_pde = flatten_jacobian(
                    jax.jacrev(pde_output)(state.params, interior_points), interior_points
                )
                jacobian_fn_bnd = flatten_jacobian(
                    jax.jacrev(bnd_output)(state.params, boundary_points), boundary_points
                )
                jacobian_fn = jnp.concat((jacobian_fn_pde, jacobian_fn_bnd))
                mat = jacobian_fn @ jacobian_fn.T
                out = jnp.linalg.solve(jnp.eye(mat.shape[0]) + 1 / lamb * mat, jvp_output / (lamb ** 2))
    
                adjustment_pde = vjp_fn_pde(out[0:len(interior_points)])[0]
                adjustment_bnd = vjp_fn_bnd(out[len(interior_points):])[0]
    
                grads = tree_map(lambda a, b, c: 1 / lamb * a - (b + c), grads, adjustment_pde, adjustment_bnd)
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

        return state, loss_bnd + loss_pde, lm_data
    
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
        
        for x_batch_int, x_batch_bnd in batch_data_pinns(x, batch_size, subkey):
            if lm_schedule and epoch >= lm_schedule.epoch:
                state, loss, lm_data = train_step(state, x_batch_int, x_batch_bnd, lm_schedule)
            else: 
                state, loss, lm_data = train_step(state, x_batch_int, x_batch_bnd)
            epoch_loss += loss
            num_batches += 1

        # Compute the average loss for the epoch
        avg_loss = epoch_loss / num_batches
        metrics_history['train_loss'].append(avg_loss)
        metrics_history['lm_data'].append(lm_data)


        if (epoch % 100) == 0: 
            if obtain_matrices:
                pass
                # out = get_eigs(state, x)
                
                # for key in out.keys():
                #     metrics_history[key].append(out[key])

            # metrics_history['prediction'].append(
            #     state.apply_fn(state.params, x_interio)
            # )
            pass


    return state, metrics_history

def run_pinns():
    model = MLP_1D(
        1, 128
    )
    state = create_train_state(model, learning_rate=1e-3, momentum=0, optimizer='adams', shape=jnp.ones((1, 2)))

    num_interior = 80 * 4
    num_boundary = 20 * 4
    interior_points, boundary_points = generate_training_data(num_interior, num_boundary)

    lm_info = PrecondData(
        # method='smw',epoch=0, lamb=1e-1,
        method='gn-exact',epoch=0
    )

    state, metrics_history = train_model(state,
        (interior_points, boundary_points), 
        num_iterations=2000,
        lm_schedule=lm_info,
        batch_size=100            
    )

    plot_results_pinns(state, metrics_history, model, (interior_points, boundary_points) )
    plt.show()


if __name__ == "__main__":
    run_pinns()
